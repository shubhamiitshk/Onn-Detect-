#!/usr/bin/env python3
"""Fits the P2 4-parameter logistic combiner + missing-modality indicators for VideoCalibration.kt.

Input CSV/JSONL columns per video (200 real + 200 synthetic):
label (0 or 1), visual_score, temporal_score, jitter_score, audio_score
(NaN when a modality is absent, e.g. silent video or no face tracked).

Outputs: android/app/src/main/assets/constants/video_calibration.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, roc_curve


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def fit_video_combiner(rows: list[dict]) -> dict:
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.float64)
    if len(y) < 50 or len(np.unique(y)) < 2:
        raise ValueError("Corpus must contain both real (0) and synthetic (1) videos")

    v = np.asarray([float(r.get("visual_score", np.nan)) for r in rows], dtype=np.float64)
    t = np.asarray([float(r.get("temporal_score", np.nan)) for r in rows], dtype=np.float64)
    j = np.asarray([float(r.get("jitter_score", np.nan)) for r in rows], dtype=np.float64)
    a = np.asarray([float(r.get("audio_score", np.nan)) for r in rows], dtype=np.float64)

    miss_t = np.isnan(t).astype(np.float64)
    miss_j = np.isnan(j).astype(np.float64)
    miss_a = np.isnan(a).astype(np.float64)

    v_clean = np.nan_to_num(v, nan=0.5)
    t_clean = np.nan_to_num(t, nan=0.0)
    j_clean = np.nan_to_num(j, nan=0.0)
    a_clean = np.nan_to_num(a, nan=0.0)

    # Design matrix matching VideoCalibration.kt:
    # logit = bias + w_v*visual + w_t*temporal + w_j*jitter + w_a*audio
    #         + missing_audio*I_a + missing_jitter*I_j + missing_temporal*I_t
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

    # Constrain modality weights (w_v, w_t, w_j, w_a) >= 0 so VideoCalibration's
    # fail-closed non-negative weight check always passes.
    bounds = [
        (None, None),  # bias
        (0.05, None),  # w_v (strictly positive)
        (0.01, None),  # w_t (lifts TEMPORAL_WEIGHT > 0)
        (0.0, None),   # w_j
        (0.0, None),   # w_a
        (None, None),  # missing_audio
        (None, None),  # missing_jitter
        (None, None),  # missing_temporal
    ]
    init = np.array([-2.0, 2.5, 0.8, 0.6, 0.8, 0.0, 0.0, 0.0], dtype=np.float64)

    def loss(params: np.ndarray) -> float:
        p = np.clip(_sigmoid(X @ params), 1e-7, 1.0 - 1e-7)
        l2 = 1e-3 * float(np.sum(params[1:] ** 2))
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))) + l2

    res = minimize(loss, init, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    params = res.x
    probs = _sigmoid(X @ params)
    auc = float(roc_auc_score(y, probs))
    fpr, tpr, thresholds = roc_curve(y, probs)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    threshold_at_eer = float(np.clip(thresholds[eer_idx], 0.05, 0.95))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

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
        "threshold_at_eer": round(threshold_at_eer, 6),
        "measured_auc": round(auc, 6),
        "measured_eer": round(eer, 6),
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
        f"OK: video_calibration.json written (AUC={result['measured_auc']:.4f}, "
        f"EER={result['measured_eer']:.4f}, thr={result['threshold_at_eer']:.4f})"
    )


if __name__ == "__main__":
    main()
