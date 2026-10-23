#!/usr/bin/env python3
"""Fits the P2 4-parameter logistic combiner + missing-modality indicators for VideoCalibration.kt.

Implements the calibration contract specified in ONN_DETECT_Final_Architecture_Blueprint.pdf
(Chapter 10.5):
    video_score = sigmoid(
        bias + w_v*visual + w_t*temporal + w_j*jitter + w_a*audio
        + missing_audio*I_a + missing_jitter*I_j + missing_temporal*I_t
    )

Zero external ML dependencies required beyond NumPy (uses SciPy L-BFGS-B when installed,
with a built-in projected-gradient fallback in pure NumPy).

Usage
-----
    python3 ml/scripts/calibrate_video_temporal.py \
        --corpus-jsonl data/video_calibration_400.jsonl \
        --output android/app/src/main/assets/constants/video_calibration.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def _roc_curve_and_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float]:
    """Pure-NumPy exact ROC-AUC, EER, and threshold_at_eer (no scikit-learn required)."""
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    pos = float(np.sum(y_sorted == 1.0))
    neg = float(np.sum(y_sorted == 0.0))
    if pos == 0.0 or neg == 0.0:
        raise ValueError("Both positive (synthetic=1) and negative (real=0) samples are required")

    tps = np.cumsum(y_sorted == 1.0)
    fps = np.cumsum(y_sorted == 0.0)
    tpr = np.concatenate([[0.0], tps / pos])
    fpr = np.concatenate([[0.0], fps / neg])
    thresholds = np.concatenate([[min(1.0, float(s_sorted[0]) + 1e-6)], s_sorted])

    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) * 0.5))
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) * 0.5)
    thr_eer = float(np.clip(thresholds[eer_idx], 0.05, 0.95))
    return auc, eer, thr_eer


def _fit_projected_logistic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fits L2-regularized logistic regression with non-negativity constraints on w_v, w_t, w_j, w_a."""
    lower = np.array([-np.inf, 0.05, 0.01, 0.0, 0.0, -np.inf, -np.inf, -np.inf], dtype=np.float64)
    init = np.array([-2.0, 2.5, 0.8, 0.6, 0.8, 0.0, 0.0, 0.0], dtype=np.float64)

    def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        p = np.clip(_sigmoid(X @ params), 1e-7, 1.0 - 1e-7)
        reg = np.zeros_like(params)
        reg[1:] = 1e-3 * params[1:]
        loss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))) + 0.5 * float(np.sum(1e-3 * params[1:] ** 2))
        grad = (X.T @ (p - y)) / float(len(y)) + reg
        return loss, grad

    try:
        from scipy.optimize import minimize

        bounds = [(None if np.isneginf(lo) else float(lo), None) for lo in lower]
        res = minimize(
            lambda w: loss_and_grad(w)[0],
            init,
            jac=lambda w: loss_and_grad(w)[1],
            method="L-BFGS-B",
            bounds=bounds,
        )
        if res.success:
            return np.maximum(res.x, lower)
    except ImportError:
        pass

    w = init.copy()
    v = init.copy()
    lr = 0.25
    for step in range(4000):
        _, grad = loss_and_grad(v)
        w_next = np.maximum(v - lr * grad, lower)
        v = w_next + (step / (step + 3.0)) * (w_next - w)
        w = w_next
        if float(np.max(np.abs(grad))) < 1e-7:
            break
    return w


def fit_video_combiner(rows: list[dict]) -> dict:
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.float64)
    if len(y) < 50 or len(np.unique(y)) < 2:
        raise ValueError("Corpus must contain at least 50 videos spanning both real (0) and synthetic (1)")

    def _col(key: str) -> np.ndarray:
        vals = []
        for r in rows:
            val = r.get(key, None)
            vals.append(np.nan if val is None else float(val))
        return np.asarray(vals, dtype=np.float64)

    v = _col("visual_score")
    t = _col("temporal_score")
    j = _col("jitter_score")
    a = _col("audio_score")

    miss_t = np.isnan(t).astype(np.float64)
    miss_j = np.isnan(j).astype(np.float64)
    miss_a = np.isnan(a).astype(np.float64)

    v_clean = np.nan_to_num(v, nan=0.5)
    t_clean = np.nan_to_num(t, nan=0.0)
    j_clean = np.nan_to_num(j, nan=0.0)
    a_clean = np.nan_to_num(a, nan=0.0)

    X = np.column_stack([
        np.ones_like(y),
        v_clean,
        t_clean * (1.0 - miss_t),
        j_clean * (1.0 - miss_j),
        a_clean * (1.0 - miss_a),
        miss_a,
        miss_j,
        miss_t,
    ])

    params = _fit_projected_logistic(X, y)
    probs = _sigmoid(X @ params)
    auc, eer, threshold_at_eer = _roc_curve_and_auc(y, probs)

    return {
        "schema_version": "video-calibration-v1",
        "calibrated": True,
        "bias": round(float(params[0]), 6),
        "w_v": round(float(params[1]), 6),
        "w_t": round(float(params[2]), 6),
        "w_j": round(float(params[3]), 6),
        "w_a": round(float(params[4]), 6),
        "missing_audio": round(float(params[5]), 6),
        "missing_jitter": round(float(params[6]), 6),
        "missing_temporal": round(float(params[7]), 6),
        "threshold_at_eer": round(float(threshold_at_eer), 6),
        "measured_auc": round(float(auc), 6),
        "measured_eer": round(float(eer), 6),
        "corpus_size": int(len(y)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", required=True, help="Path to 400-video feature JSONL")
    parser.add_argument(
        "--output",
        default="android/app/src/main/assets/constants/video_calibration.json",
    )
    args = parser.parse_args()

    corpus_bytes = Path(args.corpus_jsonl).read_bytes()
    rows = [json.loads(line) for line in corpus_bytes.decode("utf-8").splitlines() if line.strip()]
    result = fit_video_combiner(rows)
    result["split_manifest_sha256"] = hashlib.sha256(corpus_bytes).hexdigest()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"OK: {out_path} written "
        f"(AUC={result['measured_auc']:.4f}, EER={result['measured_eer']:.4f}, "
        f"thr={result['threshold_at_eer']:.4f}, w_t={result['w_t']:.4f})"
    )


if __name__ == "__main__":
    main()
