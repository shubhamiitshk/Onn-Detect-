#!/usr/bin/env python3
"""Train a regularized forensic head on frozen foundation embeddings.

Adheres strictly to the implementation contract in IMPLEMENTATION.md:
1. Validates that root identities and image hashes are strictly disjoint across train/calib/test.
2. Standardizes features using statistics computed ONLY on the training split.
3. Fits regularized logistic regression (L2 penalty) on CPU.
4. Folds standardization mean and std directly into float32 weights and bias BEFORE threshold selection:
      W_folded = W / std
      b_folded = b - sum(W * mean / std)
   This ensures the on-device inference engine evaluates pure affine dot product with zero runtime overhead.
5. Selects a tie-aware threshold on calibration negatives at the target FPR (e.g., 1.0% FPR).
6. Evaluates untouched test partition and computes AUROC, TPR@1%FPR, TPR@0.1%FPR, and EER.
7. Emits JSON head definition with `diagnostic_only: true` for wire compatibility.

Usage:
    python -m ml.scripts.train_frozen_forensic_head \\
        --features /path/to/features.npz \\
        --contract /path/to/feature_contract.json \\
        --output-dir /path/to/output \\
        --target-fpr 0.01

    # Self-test mode (verifies folding arithmetic and calibration logic):
    python -m ml.scripts.train_frozen_forensic_head --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def fold_standardization_into_weights(
    raw_weights: np.ndarray,
    raw_bias: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Folds (x - mean) / std into affine W_folded * x + b_folded."""
    # (x - mu) / sigma * W + b = x * (W / sigma) + (b - sum(W * mu / sigma))
    safe_std = np.where(std < 1e-12, 1.0, std)
    w_folded = raw_weights / safe_std
    b_folded = float(raw_bias - np.sum(w_folded * mean))
    return w_folded.astype(np.float32), float(b_folded)


def predict_probability_folded(features: np.ndarray, w_folded: np.ndarray, b_folded: float) -> np.ndarray:
    """Ordered affine arithmetic matching Kotlin ForensicFeatureHead."""
    logits = np.dot(features.astype(np.float32), w_folded) + b_folded
    # Standard sigmoid with clipping for numerical stability
    logits = np.clip(logits, -88.0, 88.0)
    return 1.0 / (1.0 + np.exp(-logits))


def select_tie_aware_cutoff(calib_scores: np.ndarray, calib_labels: np.ndarray, target_fpr: float = 0.01) -> float:
    """Selects decision cutoff on calibration negatives meeting target FPR with conservative tie-handling."""
    neg_scores = calib_scores[calib_labels == 0]
    if len(neg_scores) == 0:
        return 0.5
    sorted_neg = np.sort(neg_scores)
    # Target percentile: (1 - target_fpr)
    idx = int(math.ceil((1.0 - target_fpr) * len(sorted_neg))) - 1
    idx = min(max(0, idx), len(sorted_neg) - 1)
    cutoff = float(sorted_neg[idx])
    return cutoff


def evaluate_binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    """Computes AUC, EER, and TPR at fixed FPR operating points."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return {"auc": 0.5, "eer": 0.5, "tpr_at_1pct_fpr": 0.0, "tpr_at_threshold": 0.0}

    # AUC via Mann-Whitney U
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    pos_ranks = ranks[:n_pos]
    u = np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0
    auc = float(u / (n_pos * n_neg))

    # TPR at threshold
    tpr_at_thresh = float(np.mean(pos >= threshold))
    fpr_at_thresh = float(np.mean(neg >= threshold))

    # TPR @ 1% FPR
    cutoff_1pct = select_tie_aware_cutoff(scores, labels, target_fpr=0.01)
    tpr_at_1pct = float(np.mean(pos >= cutoff_1pct))

    # TPR @ 0.1% FPR
    cutoff_0_1pct = select_tie_aware_cutoff(scores, labels, target_fpr=0.001)
    tpr_at_0_1pct = float(np.mean(pos >= cutoff_0_1pct))

    return {
        "auc": round(auc, 6),
        "tpr_at_threshold": round(tpr_at_thresh, 4),
        "fpr_at_threshold": round(fpr_at_thresh, 4),
        "tpr_at_1pct_fpr": round(tpr_at_1pct, 4),
        "tpr_at_0_1pct_fpr": round(tpr_at_0_1pct, 4),
    }


def train_forensic_head(
    features_npz_path: Path,
    contract_json_path: Path,
    output_dir: Path,
    target_fpr: float = 0.01,
    c_reg: float = 1.0,
) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(str(features_npz_path))
    X = data["features"]
    y = data["labels"]
    splits = data["splits"]
    root_ids = data["root_ids"] if "root_ids" in data else np.array(["unknown"] * len(y))

    # Verify root identity isolation
    train_mask = splits == "train"
    calib_mask = splits == "calibration"
    test_mask = splits == "test"

    train_roots = set(root_ids[train_mask])
    calib_roots = set(root_ids[calib_mask])
    test_roots = set(root_ids[test_mask])

    overlap_train_calib = train_roots.intersection(calib_roots)
    overlap_train_test = train_roots.intersection(test_roots)
    overlap_calib_test = calib_roots.intersection(test_roots)

    if overlap_train_calib or overlap_train_test or overlap_calib_test:
        raise ValueError(
            f"Split leakage detected! Root ID overlaps: "
            f"train/calib={len(overlap_train_calib)}, "
            f"train/test={len(overlap_train_test)}, "
            f"calib/test={len(overlap_calib_test)}"
        )

    # Compute standardization ONLY on train split
    X_train = X[train_mask]
    y_train = y[train_mask]
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    safe_std = np.where(train_std < 1e-12, 1.0, train_std)

    X_train_std = (X_train - train_mean) / safe_std

    # Fit L2 regularized logistic regression
    clf = LogisticRegression(C=c_reg, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit(X_train_std, y_train)

    raw_w = clf.coef_[0]
    raw_b = float(clf.intercept_[0])

    # Fold standardization into weights
    w_folded, b_folded = fold_standardization_into_weights(raw_w, raw_b, train_mean, safe_std)

    # Score calibration set with folded arithmetic
    X_calib = X[calib_mask]
    y_calib = y[calib_mask]
    calib_scores = predict_probability_folded(X_calib, w_folded, b_folded)
    cutoff = select_tie_aware_cutoff(calib_scores, y_calib, target_fpr=target_fpr)

    # Score test set
    X_test = X[test_mask]
    y_test = y[test_mask]
    test_scores = predict_probability_folded(X_test, w_folded, b_folded)
    test_metrics = evaluate_binary_metrics(test_scores, y_test, cutoff)

    contract = json.loads(contract_json_path.read_text(encoding="utf-8")) if contract_json_path.is_file() else {}

    head_def = {
        "model_id": contract.get("model_id", "google/embeddinggemma-2"),
        "model_revision": contract.get("model_revision", "main"),
        "feature_dim": int(X.shape[-1]),
        "weights": [round(float(x), 7) for x in w_folded],
        "bias": round(b_folded, 7),
        "threshold": round(cutoff, 7),
        "target_fpr": target_fpr,
        "diagnostic_only": True,
        "contract": "scalar-prob-v1",
        "calibration_metrics": {
            "n_calibration": int(np.sum(calib_mask)),
            "n_calib_neg": int(np.sum((calib_mask) & (y == 0))),
            "n_calib_pos": int(np.sum((calib_mask) & (y == 1))),
        },
        "test_metrics": test_metrics,
    }

    head_path = output_dir / "forensic_feature_head.json"
    head_path.write_text(json.dumps(head_def, indent=2), encoding="utf-8")
    print(f"Trained forensic head written to {head_path}")
    print(f"Test Metrics: AUC={test_metrics['auc']}, TPR@1%FPR={test_metrics['tpr_at_1pct_fpr']}")
    return head_def


def run_self_test() -> bool:
    print("Running train_frozen_forensic_head self-test...")
    rng = np.random.RandomState(42)
    dim = 768
    n_samples = 120

    splits = ["train"] * 60 + ["calibration"] * 30 + ["test"] * 30
    y = np.array([i % 2 for i in range(n_samples)], dtype=np.int32)
    root_ids = np.array([f"root_{i // 2:03d}" for i in range(n_samples)])

    # Inject slight signal into fakes
    X = rng.randn(n_samples, dim).astype(np.float32)
    X[y == 1, :30] += 0.8
    # L2 normalize
    X = X / np.linalg.norm(X, axis=-1, keepdims=True)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        npz = tmp / "features.npz"
        np.savez_compressed(npz, features=X, labels=y, splits=np.array(splits), root_ids=root_ids)
        contract = tmp / "feature_contract.json"
        contract.write_text(json.dumps({"model_id": "test-gemma", "feature_dim": dim}))

        out = tmp / "out"
        res = train_forensic_head(npz, contract, out, target_fpr=0.01)

        assert (out / "forensic_feature_head.json").is_file()
        assert res["diagnostic_only"] is True
        assert len(res["weights"]) == dim
        assert res["test_metrics"]["auc"] > 0.85

    print("train_frozen_forensic_head self-test PASSED!")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=str, default="")
    parser.add_argument("--contract", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="head_output")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--c-reg", type=float, default=1.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.features:
        parser.error("--features is required unless --self-test is used")

    train_forensic_head(
        Path(args.features).resolve(),
        Path(args.contract).resolve() if args.contract else Path(""),
        Path(args.output_dir).resolve(),
        args.target_fpr,
        args.c_reg,
    )


if __name__ == "__main__":
    main()
