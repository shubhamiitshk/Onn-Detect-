"""
Calibrate the MLLM scorer on a labeled calibration split.

This script fits Platt scaling parameters to the MLLM's raw probability
outputs so that the final scores are true calibrated probabilities.

Usage:
    python -m ml.scripts.calibrate_mllm \
        --scores releases/onn-20261020-v5/mllm_scores_calibration.json \
        --dataset-manifest datasets/openfake/manifest.json \
        --split calibration \
        --output releases/onn-20261020-v5/mllm_calibration.json

Output format:
    {
        "metrics": {
            "n_samples": 400,
            "raw_auc": 0.847,
            "calibrated_auc": 0.847,
            "raw_brier": 0.142,
            "calibrated_brier": 0.098,
            "platt_a": 1.234,
            "platt_b": -0.056
        },
        "platt_scaler": {
            "a": 1.234,
            "b": -0.056,
            "fitted": true
        },
        "model_name": "google/gemma-3n-E2B-it"
    }
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ml.src.models.mllm_forensic_scorer import (
    MllmForensicScorer,
    PlattScaler,
    load_mllm_scores,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate MLLM probability outputs via Platt scaling"
    )
    parser.add_argument(
        "--scores",
        required=True,
        help="Path to MLLM scores JSON file",
    )
    parser.add_argument(
        "--dataset-manifest",
        required=True,
        help="Path to dataset manifest JSON",
    )
    parser.add_argument(
        "--split",
        default="calibration",
        help="Dataset split to use for calibration",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for calibration parameters",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load MLLM scores
    logger.info(f"Loading MLLM scores from {args.scores}")
    scores = load_mllm_scores(args.scores)
    logger.info(f"Loaded {len(scores)} MLLM scores")

    # Load dataset manifest to get labels
    logger.info(f"Loading dataset manifest from {args.dataset_manifest}")
    with open(args.dataset_manifest) as f:
        manifest = json.load(f)

    split_entries = manifest["splits"][args.split]

    # Match scores to labels
    matched_scores = []
    matched_labels = []

    for entry in split_entries:
        image_path = Path(entry["path"])
        label = entry["label"]  # 0=real, 1=AI
        filename = image_path.name

        if filename in scores:
            matched_scores.append(scores[filename].probability)
            matched_labels.append(label)

    if not matched_scores:
        logger.error("No MLLM scores matched to dataset entries!")
        return 1

    raw_scores = np.array(matched_scores)
    labels = np.array(matched_labels)

    logger.info(f"Matched {len(matched_scores)} scores to labels")
    logger.info(f"Label distribution: {np.sum(labels)} AI, {len(labels) - np.sum(labels)} real")

    # Fit Platt scaling
    logger.info("Fitting Platt scaling...")
    scaler = PlattScaler()
    scaler.fit(raw_scores, labels)

    # Compute calibration metrics
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

    calibrated = np.array([scaler.transform(s) for s in raw_scores])

    metrics = {
        "n_samples": len(labels),
        "raw_auc": float(roc_auc_score(labels, raw_scores)),
        "calibrated_auc": float(roc_auc_score(labels, calibrated)),
        "raw_brier": float(brier_score_loss(labels, raw_scores)),
        "calibrated_brier": float(brier_score_loss(labels, calibrated)),
        "raw_log_loss": float(log_loss(labels, raw_scores)),
        "calibrated_log_loss": float(log_loss(labels, calibrated)),
        "platt_a": scaler.a,
        "platt_b": scaler.b,
    }

    # Expected Calibration Error
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (calibrated >= bin_edges[i]) & (calibrated < bin_edges[i + 1])
        if mask.sum() > 0:
            avg_confidence = calibrated[mask].mean()
            avg_accuracy = labels[mask].mean()
            ece += (mask.sum() / len(labels)) * abs(avg_confidence - avg_accuracy)
    metrics["expected_calibration_error"] = float(ece)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "platt_scaler": scaler.to_dict(),
            "model_name": "google/gemma-3n-E2B-it",
        }, f, indent=2)

    logger.info("=" * 60)
    logger.info("MLLM Calibration Results")
    logger.info("=" * 60)
    logger.info(f"Samples: {metrics['n_samples']}")
    logger.info(f"Raw AUC: {metrics['raw_auc']:.4f}")
    logger.info(f"Calibrated AUC: {metrics['calibrated_auc']:.4f}")
    logger.info(f"Raw Brier: {metrics['raw_brier']:.4f}")
    logger.info(f"Calibrated Brier: {metrics['calibrated_brier']:.4f}")
    logger.info(f"Expected Calibration Error: {metrics['expected_calibration_error']:.4f}")
    logger.info(f"Platt a={scaler.a:.4f}, b={scaler.b:.4f}")
    logger.info(f"Output saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
