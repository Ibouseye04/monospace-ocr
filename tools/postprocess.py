#!/usr/bin/env python3
"""Post-processing / constraint-based decoding for OCR output.

Two strategies implemented:
  1. Confusion-cost weighted correction against an allowed charset or regex
  2. Viterbi decoding over per-position probabilities with character constraints

Usage:
    # Apply charset constraints (e.g. base64 positions can only be A-Za-z0-9+/=)
    python3 tools/postprocess.py \\
        --preds runs/preds.jsonl \\
        --out runs/preds_constrained.jsonl \\
        --charset 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='

    # Or use a per-position regex pattern
    python3 tools/postprocess.py \\
        --preds runs/preds.jsonl \\
        --out runs/preds_constrained.jsonl \\
        --pattern '[A-Za-z0-9+/=]'
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from metrics import AMBIG_SETS, GlyphRow, read_jsonl


# Confusion cost map: cost of substituting char a with char b.
# Lower cost = more plausible confusion (model often mixes these up).
DEFAULT_CONFUSION_COSTS: Dict[Tuple[str, str], float] = {}
for a, b in AMBIG_SETS:
    DEFAULT_CONFUSION_COSTS[(a, b)] = 0.2
    DEFAULT_CONFUSION_COSTS[(b, a)] = 0.2


def build_allowed_set(charset: Optional[str], pattern: Optional[str]) -> Optional[Set[str]]:
    """Build the set of allowed characters from a charset string or regex pattern."""
    if charset is not None:
        return set(charset)
    if pattern is not None:
        allowed = set()
        # Test all printable ASCII
        for code in range(32, 127):
            ch = chr(code)
            if re.fullmatch(pattern, ch):
                allowed.add(ch)
        return allowed
    return None


def constrain_prediction(
    row: GlyphRow,
    allowed: Set[str],
    confusion_costs: Dict[Tuple[str, str], float],
) -> Tuple[str, float]:
    """If the current prediction is not in the allowed set, find the best
    allowed alternative from the top-k list using confusion-aware scoring."""
    if row.pred in allowed:
        return row.pred, row.pred_score

    # Try top-k candidates that are allowed
    best_char = row.pred
    best_score = -1.0
    for char, score in row.topk:
        if char in allowed:
            # Boost score by confusion plausibility
            cost = confusion_costs.get((row.pred, char), 1.0)
            adjusted = score * (1.0 - 0.5 * cost)  # slight penalty for unlikely swaps
            if adjusted > best_score:
                best_score = adjusted
                best_char = char

    if best_score < 0:
        # No allowed character in top-k; keep original
        return row.pred, row.pred_score
    return best_char, best_score


def viterbi_line(
    line_rows: List[GlyphRow],
    allowed: Optional[Set[str]] = None,
    bigram_log_probs: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[str]:
    """Simple Viterbi over per-position top-k scores with optional bigram LM.

    dp[t][c] = best log-prob ending with char c at position t
    """
    import math

    if not line_rows:
        return []

    n = len(line_rows)
    # Collect candidate chars per position
    candidates: List[List[Tuple[str, float]]] = []
    for r in line_rows:
        cands = [(c, s) for c, s in r.topk if (allowed is None or c in allowed)]
        if not cands:
            cands = [(r.pred, r.pred_score)]
        candidates.append(cands)

    # Forward pass
    # dp[c] = (log_prob, backpointer_char_or_None)
    dp_prev: Dict[str, Tuple[float, Optional[str]]] = {}
    for c, s in candidates[0]:
        lp = math.log(max(s, 1e-12))
        dp_prev[c] = (lp, None)

    dp_chain: List[Dict[str, Tuple[float, Optional[str]]]] = [dp_prev]

    for t in range(1, n):
        dp_cur: Dict[str, Tuple[float, Optional[str]]] = {}
        for c, s in candidates[t]:
            vis_lp = math.log(max(s, 1e-12))
            best_lp = -float("inf")
            best_prev = None
            for pc, (plp, _) in dp_prev.items():
                trans = 0.0
                if bigram_log_probs and (pc, c) in bigram_log_probs:
                    trans = bigram_log_probs[(pc, c)]
                total = plp + vis_lp + trans
                if total > best_lp:
                    best_lp = total
                    best_prev = pc
            dp_cur[c] = (best_lp, best_prev)
        dp_prev = dp_cur
        dp_chain.append(dp_cur)

    # Backtrace
    best_end = max(dp_prev, key=lambda c: dp_prev[c][0])
    result = [best_end]
    for t in range(n - 1, 0, -1):
        _, prev_c = dp_chain[t][result[-1]]
        result.append(prev_c if prev_c else result[-1])
    result.reverse()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True, help="Input JSONL predictions")
    ap.add_argument("--out", required=True, help="Output corrected JSONL")
    ap.add_argument(
        "--charset",
        default=None,
        help="String of allowed characters (e.g. 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')",
    )
    ap.add_argument(
        "--pattern",
        default=None,
        help="Regex pattern for allowed characters per position",
    )
    ap.add_argument(
        "--viterbi",
        action="store_true",
        help="Use Viterbi decoding over each line instead of per-glyph correction",
    )
    args = ap.parse_args()

    allowed = build_allowed_set(args.charset, args.pattern)
    if allowed is None:
        print("Provide --charset or --pattern to constrain predictions.")
        return

    rows = read_jsonl(args.preds)

    if args.viterbi:
        # Group rows by (page_id, line_id)
        lines: Dict[Tuple[str, int], List[Tuple[int, GlyphRow]]] = defaultdict(list)
        for i, r in enumerate(rows):
            lines[(r.page_id, r.line_id)].append((i, r))

        corrected = [r.pred for r in rows]
        for key in sorted(lines):
            entries = sorted(lines[key], key=lambda x: x[1].glyph_idx)
            line_rows = [e[1] for e in entries]
            decoded = viterbi_line(line_rows, allowed=allowed)
            for (orig_idx, _), new_char in zip(entries, decoded):
                corrected[orig_idx] = new_char

        n_changed = sum(
            1 for r, c in zip(rows, corrected) if r.pred != c
        )
        with open(args.out, "w", encoding="utf-8") as f_out:
            for r, new_pred in zip(rows, corrected):
                out_row = {
                    "page_id": r.page_id,
                    "line_id": r.line_id,
                    "glyph_idx": r.glyph_idx,
                    "pred": new_pred,
                    "pred_score": r.pred_score,
                    "topk": [[c, s] for c, s in r.topk],
                }
                if r.gt is not None:
                    out_row["gt"] = r.gt
                if r.bbox is not None:
                    out_row["bbox"] = r.bbox
                if r.img_path is not None:
                    out_row["img_path"] = r.img_path
                if new_pred != r.pred:
                    out_row["constrained"] = True
                    out_row["original_pred"] = r.pred
                f_out.write(json.dumps(out_row) + "\n")
    else:
        # Per-glyph constrained correction
        n_changed = 0
        with open(args.out, "w", encoding="utf-8") as f_out:
            for r in rows:
                new_pred, new_score = constrain_prediction(
                    r, allowed, DEFAULT_CONFUSION_COSTS
                )
                if new_pred != r.pred:
                    n_changed += 1
                out_row = {
                    "page_id": r.page_id,
                    "line_id": r.line_id,
                    "glyph_idx": r.glyph_idx,
                    "pred": new_pred,
                    "pred_score": round(new_score, 6),
                    "topk": [[c, s] for c, s in r.topk],
                }
                if r.gt is not None:
                    out_row["gt"] = r.gt
                if r.bbox is not None:
                    out_row["bbox"] = r.bbox
                if r.img_path is not None:
                    out_row["img_path"] = r.img_path
                if new_pred != r.pred:
                    out_row["constrained"] = True
                    out_row["original_pred"] = r.pred
                f_out.write(json.dumps(out_row) + "\n")

    print(f"Corrected {n_changed}/{len(rows)} glyphs. Wrote {args.out}")


if __name__ == "__main__":
    main()
