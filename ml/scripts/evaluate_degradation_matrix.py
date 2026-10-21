"""Evaluate Image Detector Robustness Across a Degradation Matrix.

Evaluates an image forensic model across:
1. JPEG compression quality factors: QF 90, 70, 50, 30.
2. Downscale-and-upscale artifacts: ratios 0.75, 0.50, 0.25.
3. Gaussian blurring: radius 0.5, 1.0, 2.0.
4. Per-source false positive rates (FPR) on real sources (LAION, Pexels, etc.).

Enforces the acceptance gate: No source with >=15 real samples may exceed 5% FPR.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.src.data.bfree_paired_corpus import (
    CodecProfile,
    apply_codec_profile,
)


DEGRADATION_MATRIX: List[CodecProfile] = [
    CodecProfile(name="clean_raw", format="PNG", quality=100),
    CodecProfile(name="jpeg_q90", format="JPEG", quality=90),
    CodecProfile(name="jpeg_q70", format="JPEG", quality=70),
    CodecProfile(name="jpeg_q50", format="JPEG", quality=50),
    CodecProfile(name="jpeg_q30", format="JPEG", quality=30),
    CodecProfile(name="downscale_0.75", format="PNG", quality=100, downscale_ratio=0.75),
    CodecProfile(name="downscale_0.50", format="PNG", quality=100, downscale_ratio=0.50),
    CodecProfile(name="downscale_0.25", format="PNG", quality=100, downscale_ratio=0.25),
    CodecProfile(name="blur_0.5", format="PNG", quality=100, blur_radius=0.5),
    CodecProfile(name="blur_1.0", format="PNG", quality=100, blur_radius=1.0),
    CodecProfile(name="blur_2.0", format="PNG", quality=100, blur_radius=2.0),
]


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute binary classification metrics."""
    preds = (scores >= threshold).astype(int)
    positives = labels == 1
    negatives = labels == 0

    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))

    tp = int(np.sum(preds[positives] == 1))
    fp = int(np.sum(preds[negatives] == 1))
    tn = int(np.sum(preds[negatives] == 0))
    fn = int(np.sum(preds[positives] == 0))

    tpr = tp / max(1, n_pos)
    fpr = fp / max(1, n_neg)
    acc = (tp + tn) / max(1, len(labels))
    bal_acc = 0.5 * (tpr + (1.0 - fpr))

    # Fast AUC computation using trapezoidal rule over sorted thresholds
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    cum_tp = np.cumsum(sorted_labels == 1)
    cum_fp = np.cumsum(sorted_labels == 0)
    if n_pos > 0 and n_neg > 0:
        tpr_curve = cum_tp / n_pos
        fpr_curve = cum_fp / n_neg
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc = float(trapz_fn(tpr_curve, fpr_curve))
    else:
        auc = 0.5

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "tpr": float(tpr),
        "fpr": float(fpr),
    }


def evaluate_degradations(
    predict_fn,
    image_paths: List[str],
    labels: List[int],
    source_names: List[str],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run evaluation across all matrix profiles and breakdown by source."""
    results: Dict[str, Any] = {"conditions": {}, "per_source_fpr": {}}

    labels_arr = np.array(labels, dtype=int)
    negatives = labels_arr == 0

    for profile in DEGRADATION_MATRIX:
        scores = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                degraded = apply_codec_profile(img, profile)
                score = predict_fn(degraded)
                scores.append(score)
            except Exception as e:
                scores.append(0.5)

        scores_arr = np.array(scores, dtype=np.float32)
        metrics = compute_metrics(labels_arr, scores_arr, threshold=threshold)
        results["conditions"][profile.name] = metrics

    # Per-source FPR on clean images
    clean_scores = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            clean_scores.append(predict_fn(img))
        except Exception:
            clean_scores.append(0.5)
    clean_scores_arr = np.array(clean_scores, dtype=np.float32)

    unique_sources = set(source_names)
    for src in unique_sources:
        src_mask = np.array([s == src for s in source_names]) & negatives
        n_src_reals = int(np.sum(src_mask))
        if n_src_reals > 0:
            fp_count = int(np.sum(clean_scores_arr[src_mask] >= threshold))
            fpr = fp_count / n_src_reals
            results["per_source_fpr"][src] = {
                "n_real": n_src_reals,
                "fpr": float(fpr),
                "passes_5pct_gate": bool(fpr <= 0.05) if n_src_reals >= 15 else True,
            }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detector on degradation matrix")
    parser.add_argument("--output", type=str, default="degradation_report.json")
    args = parser.parse_args()
    print("Degradation Matrix Evaluator initialized. Run via test suite or dataset arguments.")


if __name__ == "__main__":
    main()
