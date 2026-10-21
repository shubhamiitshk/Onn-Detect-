"""Train and Evaluate Orthogonal-Subspace Probe with DCPT Losses.

Fits the Effort-style orthogonal subspace projector to strip high-variance
dataset-provenance shortcuts from frozen foundation embeddings, then trains
the lightweight probe classifier with DCPT paired consistency losses.

Records and validates:
- teacher_test_auc
- student_test_auc
- distillation_gap_auc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.src.models.orthogonal_subspace_probe import (
    OrthogonalProbeClassifier,
    OrthogonalSubspaceProjector,
)
from ml.src.training.dcpt_losses import (
    dcpt_consistency_loss,
    dcpt_feature_consistency_loss,
)


def compute_binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC from binary labels and predicted scores."""
    pos = labels == 1
    neg = labels == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    cum_tp = np.cumsum(sorted_labels == 1)
    cum_fp = np.cumsum(sorted_labels == 0)
    tpr = cum_tp / n_pos
    fpr = cum_fp / n_neg
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz_fn(tpr, fpr))


def train_probe_epoch(
    model: OrthogonalProbeClassifier,
    clean_embeddings: torch.Tensor,
    degraded_embeddings: torch.Tensor,
    labels: torch.Tensor,
    optimizer: optim.Optimizer,
    dcpt_weight: float = 0.5,
    dcpt_feature_weight: float = 0.2,
) -> float:
    """Train probe for one epoch with paired DCPT consistency losses."""
    model.train()
    optimizer.zero_grad()

    clean_logits, z_clean = model(clean_embeddings)
    degraded_logits, z_degraded = model(degraded_embeddings)

    bce_loss = nn.functional.binary_cross_entropy_with_logits(clean_logits, labels)
    loss = bce_loss

    if dcpt_weight > 0:
        c_loss = dcpt_consistency_loss(clean_logits, degraded_logits)
        loss = loss + dcpt_weight * c_loss

    if dcpt_feature_weight > 0:
        f_loss = dcpt_feature_consistency_loss(z_clean, z_degraded)
        loss = loss + dcpt_feature_weight * f_loss

    loss.backward()
    optimizer.step()
    return float(loss.item())


def fit_and_evaluate_probe(
    train_clean: np.ndarray,
    train_degraded: np.ndarray,
    train_labels: np.ndarray,
    test_clean: np.ndarray,
    test_labels: np.ndarray,
    teacher_test_scores: np.ndarray,
    embed_dim: int = 768,
    top_k_remove: int = 16,
    epochs: int = 10,
    lr: float = 1e-3,
) -> Dict[str, float]:
    """Fit projector, train probe, and evaluate distillation gap."""
    model = OrthogonalProbeClassifier(
        embed_dim=embed_dim,
        hidden_dim=128,
        top_k_remove=top_k_remove,
    )

    # 1. Fit projector on training clean features
    train_tensor = torch.from_numpy(train_clean).float()
    model.projector.fit_from_covariance(train_tensor)

    # 2. Train classifier head
    deg_tensor = torch.from_numpy(train_degraded).float()
    lbl_tensor = torch.from_numpy(train_labels).float()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        train_probe_epoch(
            model,
            train_tensor,
            deg_tensor,
            lbl_tensor,
            optimizer,
            dcpt_weight=0.5,
            dcpt_feature_weight=0.2,
        )

    # 3. Evaluate student probe on test set
    model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(test_clean).float()
        student_scores = model.predict_proba(test_tensor).cpu().numpy()

    teacher_auc = compute_binary_auc(test_labels, teacher_test_scores)
    student_auc = compute_binary_auc(test_labels, student_scores)
    distillation_gap = teacher_auc - student_auc

    return {
        "teacher_test_auc": float(teacher_auc),
        "student_test_auc": float(student_auc),
        "distillation_gap_auc": float(distillation_gap),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Orthogonal Subspace Probe")
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--top-k", type=int, default=16)
    args = parser.parse_args()
    print(f"Orthogonal Subspace Probe trainer initialized (dim={args.embed_dim}, top_k={args.top_k})")


if __name__ == "__main__":
    main()
