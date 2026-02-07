#!/usr/bin/env python3
"""Error-analysis CLI for monospace-ocr JSONL predictions.

Produces a JSON report with:
  - Full confusion matrix (top-N off-diagonal pairs)
  - Focused confusion counts for ambiguous character sets
  - Pairwise ROC/AUC using (score_a - score_b) from top-k
  - Margin distributions for correct vs wrong predictions
  - Confusion spike analysis by margin quantile
  - Per-class precision/recall for focus characters

Usage:
    python3 tools/ocr_analyze.py --preds runs/preds_with_gt.jsonl --out runs/report.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:
    roc_auc_score = None  # type: ignore[assignment]
    roc_curve = None  # type: ignore[assignment]

from metrics import (
    AMBIG_SETS,
    GlyphRow,
    confusion_matrix,
    per_class_precision_recall,
    read_jsonl,
    topk_margin,
    topk_score_for,
)


def per_pair_stats(
    rows: List[GlyphRow], a: str, b: str
) -> Optional[Dict]:
    """Compute pairwise ROC/AUC and margin distributions for a pair (a, b)."""
    filt = [r for r in rows if r.gt in (a, b)]
    if not filt:
        return None

    y_true = np.array([1 if r.gt == a else 0 for r in filt], dtype=np.int32)
    scores = np.array(
        [topk_score_for(r, a) - topk_score_for(r, b) for r in filt],
        dtype=np.float32,
    )

    auc = None
    roc_data = None
    if roc_auc_score is not None and len(set(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, scores))
            fpr, tpr, thr = roc_curve(y_true, scores)
            roc_data = (fpr.tolist(), tpr.tolist(), thr.tolist())
        except Exception:
            auc = None

    correct_margins: List[float] = []
    wrong_margins: List[float] = []
    for r in filt:
        m = topk_margin(r)
        if r.pred == r.gt:
            correct_margins.append(m)
        else:
            wrong_margins.append(m)

    def safe_pct(lst, q):
        return float(np.percentile(lst, q)) if lst else None

    return {
        "pair": f"{a}<->{b}",
        "n": len(filt),
        "auc": auc,
        "margin_correct_p10": safe_pct(correct_margins, 10),
        "margin_correct_med": safe_pct(correct_margins, 50),
        "margin_wrong_p90": safe_pct(wrong_margins, 90),
        "margin_wrong_med": safe_pct(wrong_margins, 50),
        "roc": roc_data,
    }


def confusion_spike_analysis(
    rows: List[GlyphRow], focus: List[str]
) -> Dict:
    """Analyse how error rate spikes for low-margin predictions."""
    focused = [r for r in rows if r.gt in focus]
    if not focused:
        return {}

    margins = np.array([topk_margin(r) for r in focused], dtype=np.float32)
    wrong = np.array([r.pred != r.gt for r in focused], dtype=bool)

    spike: Dict = {}
    for q in [1, 2, 5, 10, 20]:
        thr = float(np.percentile(margins, q))
        mask = margins <= thr
        rate = float(wrong[mask].mean()) if mask.any() else None
        spike[f"bottom_{q}pct_margin_thr"] = round(thr, 6)
        spike[f"wrong_rate_at_bottom_{q}pct"] = (
            round(rate, 6) if rate is not None else None
        )
    return spike


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True, help="JSONL glyph predictions (with gt)")
    ap.add_argument("--out", default="analysis_report.json", help="Output JSON report")
    ap.add_argument(
        "--focus",
        default="1,l,I",
        help="Comma-separated focus classes for detailed analysis",
    )
    args = ap.parse_args()

    rows = read_jsonl(args.preds)
    labeled = [r for r in rows if r.gt is not None]
    if not labeled:
        print("No labeled rows (need 'gt' field in JSONL). Exiting.")
        return

    cm = confusion_matrix(labeled)
    gt_tot = Counter(r.gt for r in labeled)
    pred_tot = Counter(r.pred for r in labeled)

    focus = [c.strip() for c in args.focus.split(",") if c.strip()]

    # Focused confusion table
    focus_cm = {
        f"{g}->{p}": n
        for (g, p), n in cm.items()
        if g in focus or p in focus
    }

    # Pairwise ROC/AUC
    pair_reports = []
    for a, b in AMBIG_SETS:
        rep = per_pair_stats(labeled, a, b)
        if rep:
            pair_reports.append(rep)

    # Confusion spikes
    spike = confusion_spike_analysis(labeled, focus)

    # Per-class precision/recall for focus chars
    pr = per_class_precision_recall(labeled, focus)

    # Overall accuracy
    correct = sum(1 for r in labeled if r.gt == r.pred)
    accuracy = correct / len(labeled) if labeled else 0.0

    report = {
        "n_rows": len(rows),
        "n_labeled": len(labeled),
        "accuracy": round(accuracy, 6),
        "gt_totals": dict(gt_tot),
        "pred_totals": dict(pred_tot),
        "confusion_pairs_top": sorted(
            [{"gt": g, "pred": p, "n": n} for (g, p), n in cm.items() if g != p],
            key=lambda x: x["n"],
            reverse=True,
        )[:50],
        "focus_confusions": focus_cm,
        "focus_precision_recall": pr,
        "pair_reports": pair_reports,
        "confusion_spikes_by_margin_quantile": spike,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out}  ({len(labeled)} labeled rows, accuracy={accuracy:.4f})")


if __name__ == "__main__":
    main()
