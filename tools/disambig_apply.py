#!/usr/bin/env python3
"""Apply a trained HOG disambiguator to override uncertain base predictions.

Reads the per-glyph JSONL emitted by cluster.py, identifies glyphs where the
base model is uncertain between ambiguous characters, runs the disambiguator
on those glyphs, and writes a corrected JSONL.

Usage:
    python3 tools/disambig_apply.py \\
        --preds runs/preds.jsonl \\
        --model disambig_1lI.pkl \\
        --glyphs_dir runs/glyphs \\
        --out runs/preds_corrected.jsonl \\
        --tau 0.10
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import hog

from metrics import AMBIG_GROUPS, GlyphRow, read_jsonl, topk_margin


def load_disambig_model(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def disambiguate_glyph(
    img_path: str,
    model_data: dict,
) -> tuple[str, float]:
    """Run HOG disambiguator on a single glyph crop. Returns (pred_class, prob)."""
    clf = model_data["clf"]
    target_size = model_data["target_size"]
    hog_params = model_data["hog_params"]

    im = Image.open(img_path).convert("L").resize(target_size, Image.BILINEAR)
    arr = np.array(im, dtype=np.float64) / 255.0
    feat = hog(arr, **hog_params).reshape(1, -1)
    proba = clf.predict_proba(feat)[0]
    best_idx = int(np.argmax(proba))
    return model_data["classes"][best_idx], float(proba[best_idx])


def find_ambig_group(top2_chars: set[str]) -> str | None:
    """Return the group name if top-2 chars belong to a known ambiguity group."""
    for name, group in AMBIG_GROUPS.items():
        if top2_chars <= group:
            return name
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True, help="Input JSONL from cluster.py")
    ap.add_argument("--model", required=True, help="Pickle disambiguator model")
    ap.add_argument(
        "--glyphs_dir",
        default=None,
        help="Directory with glyph crops (if img_path not in JSONL)",
    )
    ap.add_argument("--out", required=True, help="Output corrected JSONL")
    ap.add_argument(
        "--tau",
        type=float,
        default=0.10,
        help="Margin threshold below which disambiguator is triggered",
    )
    args = ap.parse_args()

    model_data = load_disambig_model(args.model)
    model_classes = set(model_data["classes"])
    rows = read_jsonl(args.preds)

    n_total = 0
    n_triggered = 0
    n_changed = 0

    with open(args.out, "w", encoding="utf-8") as f_out:
        for r in rows:
            n_total += 1
            out_row = {
                "page_id": r.page_id,
                "line_id": r.line_id,
                "glyph_idx": r.glyph_idx,
                "pred": r.pred,
                "pred_score": r.pred_score,
                "topk": [[c, s] for c, s in r.topk],
            }
            if r.gt is not None:
                out_row["gt"] = r.gt
            if r.bbox is not None:
                out_row["bbox"] = r.bbox
            if r.img_path is not None:
                out_row["img_path"] = r.img_path

            # Check if disambiguation should trigger
            margin = topk_margin(r)
            if len(r.topk) >= 2 and margin < args.tau:
                top2 = {r.topk[0][0], r.topk[1][0]}
                group = find_ambig_group(top2)
                if group and top2 <= model_classes:
                    # Resolve img path
                    img_path = r.img_path
                    if img_path is None and args.glyphs_dir:
                        img_path = str(
                            Path(args.glyphs_dir)
                            / f"{r.page_id}_l{r.line_id}_g{r.glyph_idx}.png"
                        )

                    if img_path and Path(img_path).exists():
                        n_triggered += 1
                        new_pred, new_score = disambiguate_glyph(img_path, model_data)
                        if new_pred != r.pred:
                            n_changed += 1
                        out_row["pred"] = new_pred
                        out_row["disambig_score"] = round(new_score, 6)
                        out_row["disambig_triggered"] = True

            f_out.write(json.dumps(out_row) + "\n")

    print(
        f"Processed {n_total} glyphs: "
        f"{n_triggered} disambiguated, {n_changed} changed. "
        f"Wrote {args.out}"
    )


if __name__ == "__main__":
    main()
