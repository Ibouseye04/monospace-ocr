#!/usr/bin/env python3
"""Train a lightweight HOG + Logistic Regression disambiguator for ambiguous
character sets (1/l/I by default).

The disambiguator is meant to run *only* when the base CNN is uncertain.
It uses Histogram of Oriented Gradients features which capture stroke endings
and fine structural differences that the main classifier's distance metric
often smooths out.

Usage:
    python3 tools/train_disambig_hog.py \\
        --data_dir buckets_aug \\
        --classes 1,l,I \\
        --out_model disambig_1lI.pkl

Requires: scikit-image (for HOG), scikit-learn (for Logistic Regression).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score


HOG_PARAMS = dict(
    orientations=12,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)


def load_dataset(
    data_dir: Path, classes: list[str], target_size: tuple[int, int] = (32, 32)
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load glyph images from per-class subdirectories, compute HOG features."""
    features = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            print(f"Warning: {cls_dir} not found, skipping")
            continue
        paths = sorted(list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg")))
        for p in paths:
            im = Image.open(p).convert("L").resize(target_size, Image.BILINEAR)
            arr = np.array(im, dtype=np.float64) / 255.0
            feat = hog(arr, **HOG_PARAMS)
            features.append(feat)
            labels.append(idx)
    return np.array(features), np.array(labels), classes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data_dir",
        required=True,
        help="Directory with per-class subdirs of glyph images",
    )
    ap.add_argument("--classes", default="1,l,I", help="Comma-separated classes")
    ap.add_argument("--out_model", default="disambig_1lI.pkl", help="Output pickle path")
    ap.add_argument("--C", type=float, default=1.0, help="Logistic regression C parameter")
    ap.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    data_dir = Path(args.data_dir)

    print(f"Loading data from {data_dir} for classes {classes} ...")
    X, y, class_names = load_dataset(data_dir, classes)
    if len(X) == 0:
        print("No data found. Check --data_dir and --classes.")
        return

    print(f"Dataset: {len(X)} samples, {len(class_names)} classes")
    for i, cn in enumerate(class_names):
        print(f"  '{cn}': {(y == i).sum()} samples")

    # Cross-validation
    clf = LogisticRegression(
        C=args.C, max_iter=2000, solver="lbfgs", multi_class="multinomial"
    )
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    print(f"\n{args.cv}-fold CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Train final model on all data
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print(f"\nFull-dataset training accuracy: {(y_pred == y).mean():.4f}")
    print("\nClassification report (train set):")
    print(classification_report(y, y_pred, target_names=class_names))
    print("Confusion matrix (train set):")
    print(confusion_matrix(y, y_pred))

    # Save
    model_data = {
        "clf": clf,
        "classes": class_names,
        "hog_params": HOG_PARAMS,
        "target_size": (32, 32),
    }
    with open(args.out_model, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nSaved disambiguator to {args.out_model}")


if __name__ == "__main__":
    main()
