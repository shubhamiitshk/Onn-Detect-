"""
Score a dataset with the MLLM forensic scorer for fusion training.

This script processes all images in a dataset split (train/val/calibration/test)
with the Gemma 3n E2B MLLM and saves the results to JSON for use by
fusion_trainer_v2.py.

Usage:
    python -m ml.scripts.score_with_mllm \
        --dataset-manifest datasets/openfake/manifest.json \
        --split calibration \
        --model-id google/gemma-3n-E2B-it \
        --output releases/onn-20261020-v5/mllm_scores_calibration.json \
        --batch-size 10

The output JSON format:
    [
        {
            "path": "datasets/openfake/images/000001.png",
            "probability": 0.723,
            "raw_probability": 0.651,
            "confidence": "high",
            "anomalies": ["unnatural skin texture", "asymmetric eyes"],
            "reasoning": "The image shows typical diffusion model artifacts...",
            "latency_ms": 2340,
            "model_name": "google/gemma-3n-E2B-it",
            "num_tokens": 156,
            "calibrated": false
        },
        ...
    ]
"""

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

from ml.src.models.mllm_forensic_scorer import MllmForensicScorer, MllmScore, score_dataset_with_mllm

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Score dataset images with MLLM for fusion training"
    )
    parser.add_argument(
        "--dataset-manifest",
        required=True,
        help="Path to dataset manifest JSON",
    )
    parser.add_argument(
        "--split",
        default="calibration",
        choices=["train", "val", "calibration", "test", "fusion_val"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--model-id",
        default="google/gemma-3n-E2B-it",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit quantization for memory efficiency",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for MLLM scores",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of images between incremental saves",
    )
    parser.add_argument(
        "--calibration-params",
        default=None,
        help="Path to fitted Platt scaling parameters JSON",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load dataset manifest
    manifest_path = Path(args.dataset_manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Get image paths for the specified split
    split_key = args.split
    if split_key not in manifest["splits"]:
        logger.error(f"Split '{split_key}' not found in manifest. "
                     f"Available: {list(manifest['splits'].keys())}")
        return 1

    image_entries = manifest["splits"][split_key]
    image_paths = [Path(entry["path"]) for entry in image_entries]

    logger.info(f"Loaded {len(image_paths)} images from split '{split_key}'")

    # Initialize MLLM scorer
    logger.info(f"Loading MLLM: {args.model_id}")
    scorer = MllmForensicScorer.from_pretrained(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )

    # Load calibration parameters if available
    if args.calibration_params:
        from ml.src.models.mllm_forensic_scorer import PlattScaler
        with open(args.calibration_params) as f:
            calib_data = json.load(f)
        scorer.platt_scaler = PlattScaler.from_dict(calib_data)
        logger.info(f"Loaded Platt scaling: a={calib_data['a']:.4f}, b={calib_data['b']:.4f}")

    # Score all images
    logger.info(f"Scoring {len(image_paths)} images...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = score_dataset_with_mllm(
        image_paths=image_paths,
        scorer=scorer,
        output_path=output_path,
        batch_size=args.batch_size,
    )

    # Compute summary statistics
    import numpy as np

    probabilities = [r["probability"] for r in results]
    latencies = [r["latency_ms"] for r in results]
    anomaly_counts = [len(r["anomalies"]) for r in results]

    logger.info("=" * 60)
    logger.info("MLLM Scoring Summary")
    logger.info("=" * 60)
    logger.info(f"Total images: {len(results)}")
    logger.info(f"Mean P(ai): {np.mean(probabilities):.4f}")
    logger.info(f"Std P(ai): {np.std(probabilities):.4f}")
    logger.info(f"Mean latency: {np.mean(latencies):.0f}ms")
    logger.info(f"Mean anomalies: {np.mean(anomaly_counts):.2f}")
    logger.info(f"Output saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
