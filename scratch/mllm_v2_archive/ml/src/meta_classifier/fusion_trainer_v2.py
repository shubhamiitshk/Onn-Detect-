"""
Fusion Trainer v2 — Expanded 16-feature schema with MLLM integration.

This module extends the existing XGBoost fusion meta-classifier with
6 new features derived from the on-device MLLM (Gemma 3n E2B) scorer:

  [10] mllm_score         — MLLM P(ai_generated) from token logprobs
  [11] mllm_confidence    — MLLM self-confidence (0.0=low, 0.5=med, 1.0=high)
  [12] mllm_anomaly_count — Number of semantic anomalies detected
  [13] modality_agreement  — Cross-model agreement score [0, 1]
  [14] srm_entropy         — SRM prediction entropy (uncertainty)
  [15] lcnn_entropy        — LCNN prediction entropy (uncertainty)

The fusion pipeline:
  1. Score all images with SRM CNN + MLLM
  2. Extract expanded 16-feature vectors
  3. Calibrate each model's output (Platt scaling)
  4. Train XGBoost on calibrated features
  5. Export to Kotlin for Android inference

Research basis:
  - CMLG (IWBF 2025): Calibrated Gaussian score fusion
  - MoA-DF (ACM MM 2025): Multi-MLLM ensemble probability combination
  - AWARE-NET (2025): Adaptive weighted averaging
  - DFBench: 540K image benchmark validating MLLM detection

Usage:
    python -m ml.src.meta_classifier.fusion_trainer_v2 \
        --config ml/configs/fusion_v2.yaml \
        --mllm-scores path/to/mllm_scores.json \
        --output-dir releases/onn-20261020-v5
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import xgboost as xgb

from .feature_extractor import extract_visual_features, FEATURE_NAMES

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Extended Feature Schema
# ──────────────────────────────────────────────────────────────────────

EXTENDED_FEATURE_NAMES = FEATURE_NAMES + [
    "mllm_score",
    "mllm_confidence",
    "mllm_anomaly_count",
    "modality_agreement",
    "srm_entropy",
    "lcnn_entropy",
]

MLLM_SCORE_FEATURES = EXTENDED_FEATURE_NAMES[10:]

# ──────────────────────────────────────────────────────────────────────
# MLLM Score Loader
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MllmScoreEntry:
    """Single MLLM score entry loaded from JSON."""
    path: str
    probability: float
    raw_probability: float
    confidence: str
    anomalies: list[str]
    reasoning: str
    latency_ms: float
    model_name: str
    num_tokens: int
    calibrated: bool

    @property
    def confidence_value(self) -> float:
        return {"low": 0.0, "medium": 0.5, "high": 1.0}.get(self.confidence, 0.5)

    @property
    def anomaly_count(self) -> float:
        return float(len(self.anomalies))


def load_mllm_scores(path: str | Path) -> dict[str, MllmScoreEntry]:
    """Load MLLM scores from JSON file into a path-keyed dict."""
    with open(path) as f:
        data = json.load(f)

    scores: dict[str, MllmScoreEntry] = {}
    for entry in data:
        score = MllmScoreEntry(**entry)
        # Use filename as key for matching
        scores[Path(score.path).name] = score
    return scores


# ──────────────────────────────────────────────────────────────────────
# Entropy Computation
# ──────────────────────────────────────────────────────────────────────

def binary_entropy(p: float) -> float:
    """Compute binary entropy of a probability value.

    H(p) = -p*log2(p) - (1-p)*log2(1-p)

    Returns 0 for p=0 or p=1 (certain), 1 for p=0.5 (maximally uncertain).
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def compute_modality_agreement(srm_score: float, mllm_score: float) -> float:
    """Compute agreement between SRM CNN and MLLM scores.

    Returns a value in [0, 1] where:
      1.0 = both models agree perfectly
      0.0 = models completely disagree (one says 0, other says 1)

    This is computed as 1 - |srm_score - mllm_score|, smoothed
    to emphasize agreement in the decision-relevant region.
    """
    # Raw agreement
    raw_agreement = 1.0 - abs(srm_score - mllm_score)

    # Emphasize agreement when both are confident
    confidence = 1.0 - binary_entropy(srm_score) * binary_entropy(mllm_score)

    return float(raw_agreement * confidence)


# ──────────────────────────────────────────────────────────────────────
# Expanded Feature Extractor
# ──────────────────────────────────────────────────────────────────────

def extract_extended_features(
    visual_features: np.ndarray,
    mllm_score: Optional[MllmScoreEntry] = None,
) -> np.ndarray:
    """Extract the full 16-feature vector for fusion v2.

    Args:
        visual_features: The existing 10-feature vector from
            extract_visual_features()
        mllm_score: Optional MLLM score entry (zeros if None)

    Returns:
        np.ndarray of shape (16,) with the extended feature vector
    """
    if mllm_score is None:
        # Use neutral values when MLLM is not available
        mllm_prob = 0.5
        mllm_conf = 0.0
        mllm_anomalies = 0.0
        srm_entropy = binary_entropy(float(visual_features[0]))
        modality_agreement = 0.5  # Neutral when no MLLM
    else:
        mllm_prob = mllm_score.probability
        mllm_conf = mllm_score.confidence_value
        mllm_anomalies = mllm_score.anomaly_count
        srm_entropy = binary_entropy(float(visual_features[0]))
        modality_agreement = compute_modality_agreement(
            float(visual_features[0]), mllm_prob
        )

    # LCNN entropy is 0 for image-only analysis (no audio)
    lcnn_entropy = 0.0

    extended = np.array([
        *visual_features[:10],
        mllm_prob,
        mllm_conf,
        mllm_anomalies,
        modality_agreement,
        srm_entropy,
        lcnn_entropy,
    ], dtype=np.float32)

    assert extended.shape == (16,), f"Expected (16,), got {extended.shape}"
    return extended


# ──────────────────────────────────────────────────────────────────────
# Calibrated XGBoost Fusion Model
# ──────────────────────────────────────────────────────────────────────

class CalibratedFusionV2:
    """XGBoost fusion with per-model calibration.

    This extends the original XGBoostMetaClassifier with:
    1. Platt scaling per input model (SRM, MLLM)
    2. Expanded 16-feature schema
    3. Calibrated probability output

    The calibration ensures that the fused output is a true probability
    (not just a score), which is critical for the VerdictPolicy threshold.
    """

    def __init__(
        self,
        srm_calibration: Optional[dict] = None,
        mllm_calibration: Optional[dict] = None,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        seed: int = 42,
    ):
        self.srm_calibration = srm_calibration or {"a": 1.0, "b": 0.0}
        self.mllm_calibration = mllm_calibration or {"a": 1.0, "b": 0.0}
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.seed = seed
        self.model: Optional[xgb.Booster] = None
        self.base_margin: float = 0.0

    def _calibrate_score(self, score: float, params: dict) -> float:
        """Apply Platt scaling to a raw score."""
        a, b = params["a"], params["b"]
        epsilon = 1e-7
        clipped = np.clip(score, epsilon, 1 - epsilon)
        logit = np.log(clipped / (1 - clipped))
        z = a * logit + b
        return float(1.0 / (1.0 + np.exp(-z)))

    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Apply calibration to individual model scores before fusion."""
        processed = features.copy()

        # Calibrate SRM score (feature 0)
        processed[0] = self._calibrate_score(
            float(features[0]), self.srm_calibration
        )

        # Calibrate MLLM score (feature 10)
        processed[10] = self._calibrate_score(
            float(features[10]), self.mllm_calibration
        )

        return processed

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "CalibratedFusionV2":
        """Train the fusion model.

        Args:
            X: Training features, shape (N, 16)
            y: Training labels (0=real, 1=AI), shape (N,)
            X_val: Optional validation features
            y_val: Optional validation labels
        """
        # Preprocess: calibrate individual scores
        X_processed = np.array([self._preprocess_features(row) for row in X])

        dtrain = xgb.DMatrix(X_processed, label=y)

        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            X_val_processed = np.array([
                self._preprocess_features(row) for row in X_val
            ])
            dval = xgb.DMatrix(X_val_processed, label=y_val)
            evals.append((dval, "val"))

        params = {
            "objective": "binary:logistic",
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "seed": self.seed,
            "eval_metric": ["auc", "logloss"],
        }

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False,
        )

        # Detect base margin intercept
        self._detect_intercept(X_processed[:10], y[:10])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict P(ai_generated) for feature vectors."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_processed = np.array([self._preprocess_features(row) for row in X])
        dmatrix = xgb.DMatrix(X_processed)
        return self.model.predict(dmatrix)

    def _detect_intercept(self, X: np.ndarray, y: np.ndarray) -> None:
        """Detect the base margin intercept for Kotlin export."""
        raw_margin = self.model.predict(
            xgb.DMatrix(X), output_margin=True
        )
        # Approximate intercept by comparing to zero-margin prediction
        self.base_margin = float(np.mean(raw_margin))

    def export_kotlin(self, output_path: str | Path) -> str:
        """Export the trained model to Kotlin for Android inference.

        Generates a Kotlin class with the same structure as
        XGBoostMetaClassifier.kt but with 16 features.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        kotlin_code = self._generate_kotlin_source()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(kotlin_code, encoding="utf-8")
        return kotlin_code

    def _generate_kotlin_source(self) -> str:
        """Generate Kotlin source code for the fusion model."""
        # Dump trees as JSON
        tree_dump = self.model.get_dump(dump_format="json")

        lines = [
            "// Auto-generated by fusion_trainer_v2.py — DO NOT MODIFY",
            "// Fusion v2: 16-feature calibrated XGBoost with MLLM integration",
            f"// Features: {', '.join(EXTENDED_FEATURE_NAMES)}",
            "package com.onndetect.detection.fusion",
            "",
            "object FusionV2Classifier {",
            f"    const val RELEASE_ID = \"onn-20261020-v5\"",
            "    const val N_FEATURES = 16",
            "",
            "    fun predict(features: FloatArray): Float {",
            "        require(features.size == N_FEATURES)",
            f"        var margin = {self.base_margin:.10e}f",
        ]

        # Parse and emit trees
        import json as json_mod
        for tree_json in tree_dump:
            tree = json_mod.loads(tree_json)
            lines.extend(self._emit_tree(tree))

        lines.extend([
            "        return sigmoid(margin)",
            "    }",
            "",
            "    private fun sigmoid(x: Float): Float {",
            "        return 1f / (1f + kotlin.math.exp(-x))",
            "    }",
            "}",
        ])

        return "\n".join(lines)

    def _emit_tree(self, tree: dict) -> list[str]:
        """Emit Kotlin if/else blocks for a single tree."""
        lines = []
        self._emit_node(tree, 1, lines)
        return lines

    def _emit_node(self, node: dict, indent: int, lines: list[str]) -> None:
        """Recursively emit tree nodes."""
        prefix = "        " + "    " * indent

        if "leaf" in node:
            lines.append(f"{prefix}margin += {node['leaf']:.10e}f")
            return

        fid = node["split"]
        thr = node["split_condition"]
        missing = node.get("missing", node["yes"])

        # Determine missing branch handling
        if missing == node["yes"]:
            condition = f"features[{fid}].isNaN() || features[{fid}] < {thr}f"
        else:
            condition = f"!features[{fid}].isNaN() && features[{fid}] < {thr}f"

        lines.append(f"{prefix}if ({condition}) {{")
        self._emit_node(node["children"][0], indent + 1, lines)
        lines.append(f"{prefix}}} else {{")
        self._emit_node(node["children"][1], indent + 1, lines)
        lines.append(f"{prefix}}}")

    def save_model(self, output_path: str | Path) -> None:
        """Save model to JSON for later loading."""
        if self.model is None:
            raise RuntimeError("Model not trained.")

        save_data = {
            "srm_calibration": self.srm_calibration,
            "mllm_calibration": self.mllm_calibration,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "seed": self.seed,
            "base_margin": self.base_margin,
        }
        Path(output_path).write_text(json.dumps(save_data, indent=2))
        self.model.save_model(str(output_path).replace(".json", ".xgb"))


# ──────────────────────────────────────────────────────────────────────
# Training Pipeline Entry Point
# ──────────────────────────────────────────────────────────────────────

def train_fusion_v2(
    config: dict[str, Any],
    mllm_scores_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train the v2 fusion model with MLLM features.

    This is the main entry point for the training pipeline. It:
    1. Loads the dataset manifest and MLLM scores
    2. Extracts 16-feature vectors for all splits
    3. Trains calibrated XGBoost
    4. Evaluates on test set
    5. Exports Kotlin for Android

    Args:
        config: Training configuration dict
        mllm_scores_path: Path to MLLM scores JSON
        output_dir: Directory to write outputs

    Returns:
        Training results dict with metrics
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load MLLM scores
    mllm_scores = load_mllm_scores(mllm_scores_path)
    logger.info(f"Loaded {len(mllm_scores)} MLLM scores")

    # TODO: Load dataset manifests and extract features
    # This integrates with the existing train_xgboost_meta.py pipeline
    # For now, return the structure for integration

    results = {
        "feature_schema": EXTENDED_FEATURE_NAMES,
        "mllm_features": MLLM_SCORE_FEATURES,
        "n_mllm_scores": len(mllm_scores),
        "calibration": {
            "srm": {"a": 1.0, "b": 0.0},  # Placeholder
            "mllm": {"a": 1.0, "b": 0.0},  # Placeholder
        },
        "status": "initialized",
    }

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train fusion v2 with MLLM features")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--mllm-scores", required=True, help="MLLM scores JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    results = train_fusion_v2(config, args.mllm_scores, args.output_dir)
    print(json.dumps(results, indent=2))
