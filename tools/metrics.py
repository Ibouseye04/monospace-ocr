"""Shared metric helpers for OCR ambiguity analysis.

Used by ocr_analyze.py, disambig_apply.py, and benchmark scripts.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Character pairs known to be visually ambiguous in monospace fonts.
AMBIG_SETS: List[Tuple[str, str]] = [
    ("1", "l"),
    ("1", "I"),
    ("l", "I"),
    ("0", "O"),
    ("5", "S"),
    ("2", "Z"),
    ("8", "B"),
]

AMBIG_GROUPS: Dict[str, frozenset] = {
    "1lI": frozenset({"1", "l", "I"}),
    "0O": frozenset({"0", "O"}),
    "5S": frozenset({"5", "S"}),
    "2Z": frozenset({"2", "Z"}),
    "8B": frozenset({"8", "B"}),
}


@dataclass(frozen=True)
class GlyphRow:
    """One row from the per-glyph JSONL predictions file."""

    page_id: str
    line_id: int
    glyph_idx: int
    pred: str
    pred_score: float
    topk: List[Tuple[str, float]]
    gt: Optional[str] = None
    img_path: Optional[str] = None
    bbox: Optional[List[int]] = None


def read_jsonl(path: str) -> List[GlyphRow]:
    """Read a per-glyph JSONL file into a list of GlyphRow objects."""
    rows: List[GlyphRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            topk = [(str(c), float(s)) for c, s in d.get("topk", [])]
            rows.append(
                GlyphRow(
                    page_id=str(d.get("page_id", "")),
                    line_id=int(d.get("line_id", 0)),
                    glyph_idx=int(d.get("glyph_idx", 0)),
                    pred=str(d["pred"]),
                    pred_score=float(
                        d.get("pred_score", topk[0][1] if topk else 0.0)
                    ),
                    topk=topk,
                    gt=str(d["gt"]) if "gt" in d and d["gt"] is not None else None,
                    img_path=d.get("img_path"),
                    bbox=d.get("bbox"),
                )
            )
    return rows


def confusion_matrix(rows: List[GlyphRow]) -> Dict[Tuple[str, str], int]:
    """Build a confusion count dict from (gt, pred) pairs."""
    cm: Counter = Counter()
    for r in rows:
        if r.gt is not None:
            cm[(r.gt, r.pred)] += 1
    return dict(cm)


def topk_margin(r: GlyphRow) -> float:
    """Score margin between rank-1 and rank-2 predictions."""
    if len(r.topk) < 2:
        return float("inf")
    return float(r.topk[0][1] - r.topk[1][1])


def topk_score_for(r: GlyphRow, char: str) -> float:
    """Look up the probability for *char* in the top-k list (0.0 if absent)."""
    for c, s in r.topk:
        if c == char:
            return s
    return 0.0


def is_ambiguous(r: GlyphRow, margin_threshold: float = 0.10) -> bool:
    """Return True if the top-2 predictions fall into any ambiguity group
    and the margin is below *margin_threshold*."""
    if len(r.topk) < 2:
        return False
    top2 = {r.topk[0][0], r.topk[1][0]}
    margin = topk_margin(r)
    if margin >= margin_threshold:
        return False
    for group in AMBIG_GROUPS.values():
        if top2 <= group:
            return True
    return False


def uncertainty_score(margin: float, tau: float, k: float = 20.0) -> float:
    """Smooth uncertainty in [0, 1]:  1 - sigmoid(k * (margin - tau))."""
    x = k * (margin - tau)
    return 1.0 - (1.0 / (1.0 + math.exp(-x)))


def per_class_precision_recall(
    rows: List[GlyphRow], classes: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """Compute precision and recall per class (only rows with gt)."""
    labeled = [r for r in rows if r.gt is not None]
    if not labeled:
        return {}
    gt_counts: Counter = Counter(r.gt for r in labeled)
    pred_counts: Counter = Counter(r.pred for r in labeled)
    tp: Counter = Counter()
    for r in labeled:
        if r.gt == r.pred:
            tp[r.gt] += 1

    targets = classes if classes else sorted(set(gt_counts) | set(pred_counts))
    out = {}
    for c in targets:
        p = tp[c] / pred_counts[c] if pred_counts[c] else 0.0
        r = tp[c] / gt_counts[c] if gt_counts[c] else 0.0
        out[c] = {
            "precision": round(p, 6),
            "recall": round(r, 6),
            "tp": tp[c],
            "gt_count": gt_counts[c],
            "pred_count": pred_counts[c],
        }
    return out
