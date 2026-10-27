"""Orthogonal-Subspace Probe for Vision Foundation Models (VFM).

Implements the Effort (ICML 2025, arXiv:2411.15633) orthogonal subspace adaptation
and probe architecture for foundation vision models (SigLIP-B/16 and DINOv2-S/14).

Why this matters:
Naive linear probes over frozen VFMs overfit high-variance semantic directions
(such as 'LAION web scrapings vs. Pexels stock photos'), which creates dataset
provenance shortcuts. Effort performs Singular Value Decomposition (SVD) on the
feature space, stripping the dominant semantic eigenspace (V_k) and training the
classifier strictly on the orthogonal lower singular subspace (V_perp).

Components:
- OrthogonalSubspaceProjector: Computes and caches P_perp = I - V_k V_k^T.
- OrthogonalProbeClassifier: Neural classifier head operating on projected features,
  emitting penultimate embeddings for DCPT consistency and final logits.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalSubspaceProjector(nn.Module):
    """Projects embeddings onto the orthogonal subspace V_perp = I - V_k V_k^T.

    ## Which data the nuisance subspace is estimated from

    The subspace MUST be estimated from the REAL class alone (``label == 0``).
    Every sample there is authentic, so any variance between acquisition
    sources (laion vs pexels/docci/imagenet) is 100% source nuisance and 0%
    synthesis signal.

    Estimating it on the MIXED class is a trap: the real-vs-fake direction
    is itself a top principal direction of the combined data, so removing
    the top-k components of the mixed covariance removes the very signal
    the probe is supposed to read.

    The measured result on the real corpus (SigLIP-B/16, 768-d,
    ``rank_k = 1``): the source-debiased probe reaches **0.8682** sealed-test
    AUC against **0.8919** for the plain probe, while the LAION real-image FPR
    falls from 0.2897 to 0.2573. So the debiasing is a **trade-off**, not a
    strict win, and the plain probe is what ships. See
    ``ml/scripts/evaluate_vfm_probes.py::fit_source_debiased_probe``.

    An earlier revision of this note claimed 0.9013 AUC and a LAION FPR of
    0.2101. That came from a fit/apply mismatch: the probe was fitted on the
    RAW projected features but applied to the STANDARDIZED ones, so the
    measured AUC belonged to a function that had never been fitted. The fit
    and the application now use the same standardization.
    """

    def __init__(self, embed_dim: int, top_k_remove: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.top_k_remove = top_k_remove
        # Register projection matrix as a buffer (not updated by gradients)
        self.register_buffer("proj_matrix", torch.eye(embed_dim))

    @torch.no_grad()
    def fit_from_covariance(self, features: torch.Tensor) -> None:
        """Fit dominant subspace V_k via SVD on centered features and build P_perp.

        ``features`` should contain the NUISANCE samples only - the real
        class, when the nuisance is the acquisition source.
        """
        if features.ndim != 2:
            raise ValueError(f"Expected 2D tensor (N, D), got {features.shape}")
        n, d = features.shape
        if d != self.embed_dim:
            raise ValueError(f"Feature dim {d} does not match projector dim {self.embed_dim}")
        if n < 2:
            raise ValueError("Need at least two samples to estimate a subspace")

        centered = features - features.mean(dim=0, keepdim=True)
        # SVD on centered features: X = U S V^T, V contains right singular vectors
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        # Top-k components to remove
        k = min(self.top_k_remove, vh.shape[0])
        v_top = vh[:k, :]  # shape: (k, D)

        # Orthogonal projection operator P_perp = I - V_top^T @ V_top
        p_perp = torch.eye(d, device=features.device) - torch.matmul(v_top.T, v_top)
        self.proj_matrix.copy_(p_perp)

    @torch.no_grad()
    def fit_from_source_discriminant(
        self,
        features: torch.Tensor,
        sources: list[str],
    ) -> None:
        """Fit the nuisance subspace from a SOURCE indicator on real images.

        This is the stronger estimator: instead of taking the top principal
        directions of the real embeddings, it takes the directions that
        DISCRIMINATE between acquisition sources. Those are exactly the
        directions a source shortcut rides on.

        ``features`` must be the real-class embeddings; ``sources`` their
        acquisition sources.
        """
        import numpy as np

        unique = sorted(set(sources))
        if len(unique) < 2:
            raise ValueError(
                "The real class has fewer than two acquisition sources; there "
                "is no source subspace to remove"
            )
        # SVD on the (K, D) weighted source centroids in feature space:
        # B = [sqrt(N_s / N) * (mu_s - mu_global)] for each acquisition source s.
        global_mean = features.mean(dim=0, keepdim=True)
        centroid_diffs = []
        n_total = float(len(features))
        for src in unique:
            mask = torch.tensor([s == src for s in sources], dtype=torch.bool, device=features.device)
            n_s = float(mask.sum().item())
            if n_s < 2:
                continue
            mu_s = features[mask].mean(dim=0, keepdim=True)
            weight = torch.sqrt(torch.tensor(n_s / n_total, dtype=torch.float32, device=features.device))
            centroid_diffs.append(weight * (mu_s - global_mean))
        if not centroid_diffs:
            p_perp = torch.eye(self.embed_dim, device=features.device)
            self.proj_matrix.copy_(p_perp)
            return
        b_matrix = torch.cat(centroid_diffs, dim=0)  # Shape (K, D)
        _, _, vh = torch.linalg.svd(b_matrix, full_matrices=False)  # vh has shape (K, D)
        k = min(self.top_k_remove, vh.shape[0])
        v_top = vh[:k, :]  # Shape (k, D)
        p_perp = torch.eye(self.embed_dim, device=features.device) - torch.matmul(v_top.T, v_top)
        self.proj_matrix.copy_(p_perp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input embeddings x of shape (..., D) onto orthogonal subspace."""
        return torch.matmul(x, self.proj_matrix)


class OrthogonalProbeClassifier(nn.Module):
    """Classifier head trained on orthogonalized VFM embeddings."""

    def __init__(
        self,
        embed_dim: int = 768,  # Default for SigLIP-B/16; use 384 for DINOv2-S/14
        hidden_dim: int = 128,
        top_k_remove: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.projector = OrthogonalSubspaceProjector(embed_dim, top_k_remove=top_k_remove)

        self.backbone = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate feature embedding for DCPT feature consistency loss."""
        proj = self.projector(x)
        # Handle 1D or 2D inputs
        if proj.ndim == 1:
            proj = proj.unsqueeze(0)
        return self.backbone(proj)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass emitting raw 1-D binary logits."""
        emb = self.extract_embedding(x)
        logits = self.head(emb)
        return logits.squeeze(-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (logits, penultimate_embedding)."""
        emb = self.extract_embedding(x)
        logits = self.head(emb).squeeze(-1)
        return logits, emb

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probability of synthetic origin in [0, 1]."""
        logits = self.forward_logits(x)
        return torch.sigmoid(logits)
