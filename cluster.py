#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import argparse

# --- DETERMINISM ---
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/> "
CLUSTERS = len(ALPHABET)
EXPECTED_COLS = 78
TARGET_CELL_SIZE = (32, 32)
WEIGHTS_PATH = "./weights"

def log(msg):
    print(msg, file=sys.stderr, flush=True)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

def normalize_character_soft(raw_cell, h_step):
    if raw_cell.size == 0: return np.zeros(TARGET_CELL_SIZE, dtype=np.uint8)
    clean_cell = raw_cell.copy()
    clean_cell[:, 0] = 0
    _, mask = cv2.threshold(clean_cell, 25, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    canvas = np.zeros(TARGET_CELL_SIZE, dtype=np.uint8)
    if coords is None: return canvas
    bx, by, bw, bh = cv2.boundingRect(coords)
    char_crop = clean_cell[by:by+bh, bx:bx+bw]
    scale = 28.0 / h_step
    nw, nh = max(1, int(bw * scale)), max(1, int(bh * scale))
    if nw > 30 or nh > 30:
        f = 30.0 / max(nw, nh)
        nw, nh = int(nw * f), int(nh * f)
    resized = cv2.resize(char_crop, (nw, nh), interpolation=cv2.INTER_AREA)
    rel_y_offset = by / h_step
    target_y = int(rel_y_offset * 32)
    target_x = (32 - nw) // 2
    if target_y + nh > 32: target_y = 32 - nh
    if target_y < 0: target_y = 0
    canvas[target_y:target_y+nh, target_x:target_x+nw] = resized
    return canvas

def solve_grid_2d(img, gx, gy, gw, gh, num_lines):
    def score_axis(projection, n_segments, start_guess, dim_guess):
        best_cost, best_params = float('inf'), (start_guess, dim_guess)
        for s_try in range(start_guess - 4, start_guess + 5):
            for d_try in range(dim_guess - 8, dim_guess + 9):
                step = d_try / n_segments
                gutter_ink = 0
                for i in range(n_segments + 1):
                    p = int(s_try + i * step)
                    if 0 <= p < len(projection): gutter_ink += projection[p]
                if gutter_ink < best_cost:
                    best_cost, best_params = gutter_ink, (s_try, d_try)
        return best_params
    y_proj = np.sum(img[:, gx:gx+gw], axis=1)
    y_s, h_t = score_axis(y_proj, num_lines, gy, gh)
    x_proj = np.sum(img[y_s:y_s+h_t, :], axis=0)
    x_s, w_t = score_axis(x_proj, EXPECTED_COLS, gx, gw)
    return x_s, y_s, w_t, h_t

def extract_cells(image_path, num_lines, debug=False):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    bg = np.median(img)
    inv = cv2.absdiff(img, int(bg))
    _, mask = cv2.threshold(inv, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None: return None
    gx, gy, gw, gh = cv2.boundingRect(coords)
    xs, ys, wt, ht = solve_grid_2d(inv, gx, gy, gw, gh, num_lines)
    w_step, h_step = wt / EXPECTED_COLS, ht / num_lines
    if debug:
        dbg = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        for r in range(num_lines + 1):
            y = int(ys + r * h_step)
            cv2.line(dbg, (int(xs), y), (int(xs + wt), y), (0, 255, 0), 1)
        for c in range(EXPECTED_COLS + 1):
            x = int(xs + c * w_step)
            cv2.line(dbg, (x, int(ys)), (x, int(ys + ht)), (0, 255, 0), 1)
        plt.figure(figsize=(8, 8)); plt.imshow(dbg); plt.title("Grid Overlay"); plt.show()
    cells = []
    for r in range(num_lines):
        for c in range(EXPECTED_COLS):
            y1, y2 = int(ys + r * h_step), int(ys + (r + 1) * h_step)
            x1, x2 = int(xs + c * w_step), int(xs + (c + 1) * w_step)
            cell_raw = inv[max(0,y1):y2, max(0,x1):x2]
            cells.append(normalize_character_soft(cell_raw, h_step))
    return np.array(cells)

def calculate_bucket_averages(visuals, labels):
    """Calculates the mean image for each cluster label."""
    averages = np.zeros((CLUSTERS, 32, 32), dtype=np.uint8)
    for i in range(CLUSTERS):
        mask = (labels == i)
        if np.any(mask):
            averages[i] = np.mean(visuals[mask], axis=0).astype(np.uint8)
    return averages

def show_outliers(visuals, labels, ref_averages, title, model_preds=None):
    fig, axes = plt.subplots(6, 11, figsize=(18, 11))
    ax_f = axes.flatten()
    for i, char in enumerate(ALPHABET):
        mask = (labels == i)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            ax_f[i].axis('off'); continue

        avg_img = ref_averages[i]
        subset_cells = visuals[indices].astype(float)
        ref_float = avg_img.astype(float)
        # Mean Squared Error to find most deviant
        mse = np.mean((subset_cells - ref_float)**2, axis=(1, 2))
        worst_idx_in_group = np.argmax(mse)
        global_idx = indices[worst_idx_in_group]
        outlier_img = visuals[global_idx]

        line, col = (global_idx // EXPECTED_COLS) + 1, (global_idx % EXPECTED_COLS) + 1
        meta = ""
        if model_preds is not None:
            pred_char = ALPHABET[model_preds[global_idx]]
            meta = f" P='{pred_char}'"

        combined = np.zeros((32, 65), dtype=np.uint8)
        combined[:, :32] = avg_img
        combined[:, 33:65] = outlier_img
        ax_f[i].imshow(combined, cmap='magma')
        ax_f[i].set_title(f"'{char}' ({line},{col}){meta}", fontsize=7)
        ax_f[i].axis('off')
    plt.suptitle(title, fontsize=16); plt.tight_layout(); plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("train_path", nargs='?', default=None)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--lines", type=int, default=65)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(CLUSTERS).to(device)
    visuals = extract_cells(args.image, args.lines, debug=args.debug)
    if visuals is None: return

    # This will hold the "Ground Truth" averages if we are in training mode
    ground_truth_averages = None
    gt_labels = None

    if args.train_path:
        # --- TRAINING MODE ---
        with open(args.train_path, "r", encoding="utf-8") as f:
            train_text = f.read().replace('\n', '').replace('\r', '')
        n_train = min(len(train_text), len(visuals))
        char_to_idx = {c: i for i, c in enumerate(ALPHABET)}
        gt_labels = np.array([char_to_idx.get(c, char_to_idx[' ']) for c in train_text[:n_train]])

        # Calculate averages based on Ground Truth labels
        ground_truth_averages = calculate_bucket_averages(visuals[:n_train], gt_labels)

        if args.debug:
            show_outliers(visuals[:n_train], gt_labels, ground_truth_averages,
                          "TRAINING MISTAKES CHECK: Ground Truth Avg vs Most Deviant")

        # Training Loop
        X = torch.tensor(visuals[:n_train], dtype=torch.float32).unsqueeze(1) / 255.0
        Y = torch.tensor(gt_labels, dtype=torch.long)
        train_idx, val_idx = train_test_split(np.arange(n_train), test_size=0.1, random_state=42)
        train_loader = DataLoader(TensorDataset(X[train_idx], Y[train_idx]), batch_size=64, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(40):
            model.train()
            l_sum = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(); loss = criterion(model(xb), yb)
                loss.backward(); optimizer.step(); l_sum += loss.item()
            log(f"Epoch {epoch+1:02d} | Loss: {l_sum/len(train_loader):.4f}")

        torch.save(model.state_dict(), WEIGHTS_PATH)
    else:
        # --- INFERENCE MODE ---
        if not os.path.exists(WEIGHTS_PATH):
            log("Error: Weights not found."); return
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

    # --- INFERENCE / PREDICTION PHASE ---
    model.eval()
    X_all = torch.tensor(visuals, dtype=torch.float32).unsqueeze(1).to(device) / 255.0
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_all), 128):
            all_preds.append(model(X_all[i:i+128]).argmax(1).cpu().numpy())
    pred_indices = np.concatenate(all_preds)

    if not args.quiet:
        for r in range(args.lines):
            row = pred_indices[r*EXPECTED_COLS : (r+1)*EXPECTED_COLS]
            sys.stdout.write("".join([ALPHABET[i] for i in row]) + "\n")

    # --- FINAL DEBUGGING COMPARISON ---
    if ground_truth_averages is not None:
        show_outliers(visuals, pred_indices, ground_truth_averages,
                      "Training Averages vs Inference Outliers", model_preds=pred_indices)
    else:
        # If we loaded weights, we have no GT. We compare model predictions against
        # averages derived from those same predictions (inf_averages).
        inf_averages = calculate_bucket_averages(visuals, pred_indices)
        show_outliers(visuals, pred_indices, inf_averages,
                      "Inference Bucket Deviations (Averages vs Outliers)")

if __name__ == "__main__":
    main()
