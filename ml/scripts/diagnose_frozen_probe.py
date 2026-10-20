"""Diagnostic: choose the frozen-probe regularisation honestly.

The first frozen-probe run memorised the stacking split (0.9688 AUC) and
collapsed to chance on the calibration split (0.5046). A 513-parameter head
fitted on 800 samples needs far stronger regularisation than L2=1.

This script extracts the embeddings once and sweeps the regularisation
strength, selecting on ``fusion_val`` (never on calibration or test).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.contracts import IMAGE_EXTS, discover_samples
from training.metrics import compute_auc
from train_frozen_vision_probe import extract_embeddings, load_frozen_tower

DATA_ROOT = Path(r"D:\onn_data\openfake_visual")
CACHE = Path(r"D:\onn_data\runs\release_v6\checkpoints\frozen_probe_embeddings_cache.npz")


def fit(X, y, l2_reg):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    Z = np.hstack([(X - mean) / std, np.ones((len(X), 1))])
    d = Z.shape[1]
    w = np.zeros(d, dtype=np.float64)
    reg = np.eye(d) * l2_reg
    reg[-1, -1] = 1e-6
    yf = y.astype(np.float64)
    for _ in range(80):
        logits = np.clip(Z @ w, -40, 40)
        p = 1 / (1 + np.exp(-logits))
        r = np.maximum(p * (1 - p), 1e-6)
        grad = Z.T @ (p - yf) + reg @ w
        hess = (Z.T * r) @ Z + reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.linalg.norm(step) < 1e-7:
            break
    return w[:-1], float(w[-1]), mean, std


def apply_probe(X, w, b, mean, std):
    Z = np.hstack([(X - mean) / std, np.ones((len(X), 1))])
    logits = np.clip(Z @ np.append(w, b), -40, 40)
    return 1 / (1 + np.exp(-logits))


def main():
    splits = ("stacking", "fusion_val", "calibration", "test")
    if CACHE.exists():
        print(f"[Cache] loading {CACHE}")
        with np.load(CACHE) as d:
            X = {s: d[f"X_{s}"] for s in splits}
            y = {s: d[f"y_{s}"] for s in splits}
    else:
        visual = load_frozen_tower()
        device = torch.device("cpu")
        X, y = {}, {}
        for split in splits:
            items = discover_samples(DATA_ROOT, split, IMAGE_EXTS)
            t0 = time.time()
            X[split] = extract_embeddings(visual, items, device, batch_size=8)
            y[split] = np.asarray([label for _, label in items], dtype=np.int64)
            print(f"[Tower] {split}: {len(items)} images in {time.time() - t0:.0f}s")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CACHE,
            **{f"X_{s}": X[s] for s in splits},
            **{f"y_{s}": y[s] for s in splits},
        )

    print("\n" + "=" * 96)
    print("REGULARISATION SWEEP  (fit on stacking, select on fusion_val)")
    print("=" * 96)
    print(f"{'l2_reg':>12} {'stacking':>10} {'fusion_val':>11} {'calibration':>12} {'test':>9}")
    best = None
    for l2 in (0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0):
        w, b, mean, std = fit(X["stacking"], y["stacking"], l2)
        a_stack = compute_auc(y["stacking"], apply_probe(X["stacking"], w, b, mean, std))
        a_val = compute_auc(y["fusion_val"], apply_probe(X["fusion_val"], w, b, mean, std))
        a_cal = compute_auc(y["calibration"], apply_probe(X["calibration"], w, b, mean, std))
        a_test = compute_auc(y["test"], apply_probe(X["test"], w, b, mean, std))
        print(f"{l2:12.1f} {a_stack:10.4f} {a_val:11.4f} {a_cal:12.4f} {a_test:9.4f}")
        if best is None or a_val > best[1]:
            best = (l2, a_val, a_cal, a_test)
    print(f"\nSelected l2_reg = {best[0]} by fusion_val AUC ({best[1]:.4f})")
    print(f"  -> calibration AUC {best[2]:.4f} | sealed test AUC {best[3]:.4f}")


if __name__ == "__main__":
    main()
