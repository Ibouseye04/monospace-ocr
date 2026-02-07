#!/usr/bin/env python3
"""Targeted data augmentation for ambiguous glyph classes (1/l/I, 0/O, etc.).

Reads labeled glyph crops organised in per-class directories and writes
augmented copies with randomised:
  - contrast / brightness shifts
  - Gaussian blur
  - stroke thickness (dilate / erode via min/max filter)
  - horizontal shear
  - crop jitter (kills serif / terminal cues, forces robustness)
  - Gaussian noise
  - pixel dropout (simulates broken ink)

Usage:
    python3 tools/augment_glyphs.py \\
        --in_dir buckets --out_dir buckets_aug \\
        --classes 1,l,I --n_per_img 10
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def pil_to_np(im: Image.Image) -> np.ndarray:
    return np.array(im)


def np_to_pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray(a)


def add_gaussian_noise(im: Image.Image, sigma: float) -> Image.Image:
    a = pil_to_np(im).astype(np.float32)
    noise = np.random.normal(0, sigma, a.shape).astype(np.float32)
    a = np.clip(a + noise, 0, 255).astype(np.uint8)
    return np_to_pil(a)


def dropout(im: Image.Image, p: float) -> Image.Image:
    a = pil_to_np(im).copy()
    mask = np.random.rand(*a.shape[:2]) < p
    a[mask] = 255  # white background assumption
    return np_to_pil(a)


def shear_x(im: Image.Image, shear: float) -> Image.Image:
    w, h = im.size
    m = (1, shear, 0, 0, 1, 0)
    return im.transform((w, h), Image.AFFINE, m, resample=Image.BICUBIC)


def jitter_crop(im: Image.Image, max_px: int) -> Image.Image:
    w, h = im.size
    dx1 = random.randint(0, max_px)
    dy1 = random.randint(0, max_px)
    dx2 = random.randint(0, max_px)
    dy2 = random.randint(0, max_px)
    x1, y1 = dx1, dy1
    x2, y2 = max(1, w - dx2), max(1, h - dy2)
    if x2 <= x1 or y2 <= y1:
        return im
    crop = im.crop((x1, y1, x2, y2))
    return ImageOps.pad(crop, (w, h), color=255, centering=(0.5, 0.5))


def dilate_erode(im: Image.Image, k: int, do_dilate: bool) -> Image.Image:
    if do_dilate:
        return im.filter(ImageFilter.MaxFilter(size=k))
    return im.filter(ImageFilter.MinFilter(size=k))


def augment_one(im: Image.Image) -> Image.Image:
    """Apply a random combination of augmentations to a single glyph image."""
    out = im.copy()

    # Contrast / brightness
    if random.random() < 0.8:
        out = ImageEnhance.Contrast(out).enhance(random.uniform(0.7, 1.4))
    if random.random() < 0.8:
        out = ImageEnhance.Brightness(out).enhance(random.uniform(0.8, 1.3))

    # Blur
    if random.random() < 0.5:
        out = out.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2))
        )

    # Stroke thickness
    if random.random() < 0.6:
        k = random.choice([3, 3, 5])
        out = dilate_erode(out, k=k, do_dilate=(random.random() < 0.5))

    # Horizontal shear
    if random.random() < 0.5:
        out = shear_x(out, shear=random.uniform(-0.2, 0.2))

    # Crop jitter
    if random.random() < 0.5:
        out = jitter_crop(out, max_px=2)

    # Noise / dropout
    if random.random() < 0.6:
        out = add_gaussian_noise(out, sigma=random.uniform(3, 18))
    if random.random() < 0.3:
        out = dropout(out, p=random.uniform(0.01, 0.05))

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in_dir",
        required=True,
        help="Directory with labelled glyph images in per-class subdirs (e.g. buckets/1, buckets/l)",
    )
    ap.add_argument("--out_dir", required=True, help="Output directory for augmented images")
    ap.add_argument(
        "--classes",
        default="1,l,I",
        help="Comma-separated classes to augment (must match subdir names)",
    )
    ap.add_argument("--n_per_img", type=int, default=8, help="Augmented copies per source image")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    total = 0
    for c in classes:
        (out_dir / c).mkdir(parents=True, exist_ok=True)
        src = in_dir / c
        if not src.is_dir():
            print(f"Warning: source directory {src} not found, skipping class '{c}'")
            continue
        imgs = sorted(list(src.glob("*.png")) + list(src.glob("*.jpg")))
        for p in imgs:
            im = Image.open(p).convert("L")
            stem = p.stem
            # Also copy the original
            im.save(out_dir / c / f"{stem}.png")
            for i in range(args.n_per_img):
                a = augment_one(im)
                a.save(out_dir / c / f"{stem}__aug{i}.png")
            total += 1 + args.n_per_img
    print(f"Wrote {total} images to {out_dir}")


if __name__ == "__main__":
    main()
