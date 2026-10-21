"""DCPT: Degradation-Consistent Paired Training losses.

Paper: arXiv:2604.10102 - "Degradation-Consistent Paired Training for Robust
AI-Generated Image Detection".

Two terms, both zero-overhead at inference (the degraded view exists only
during training):

  * a FEATURE-level cosine consistency term on the penultimate embedding, and
  * a PREDICTION-level symmetric KL term on the output distribution.

Why the KL term is written out by hand
--------------------------------------
``torch.nn.functional.kl_div(input, target)`` computes
``target * (log target - input)`` POINTWISE. Handed a single probability it
therefore evaluates only the positive-class term ``p * log(p/q)`` and
silently drops the negative-class term ``(1 - p) * log((1 - p)/(1 - q))``.

For a binary classifier that is not a divergence at all, and it weights the
gradient by ``p`` instead of by ``1 - p``. On real images (``p_clean ~
0.01``) the consistency signal is then roughly 30x weaker than on fakes
(``p_deg ~ 0.40``) - exactly backwards for a term whose job is to make the
model treat a degraded view like the clean one.

The full Bernoulli divergence is computed explicitly below.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-7


def dcpt_consistency_loss(
    clean_logits: torch.Tensor,
    degraded_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Symmetric BINARY KL divergence between two Bernoulli predictions.

    ``clean_logits`` and ``degraded_logits`` are raw logits of the same
    shape. The divergence is computed over both outcomes ``(p, 1 - p)``.

    The ``temperature ** 2`` factor restores the gradient scale that the
    soft-target temperature scaling removed, matching the standard
    knowledge-distillation convention.
    """
    if clean_logits.shape != degraded_logits.shape:
        raise ValueError(
            "clean and degraded logits must have the same shape, got "
            f"{tuple(clean_logits.shape)} and {tuple(degraded_logits.shape)}"
        )
    p = torch.sigmoid(clean_logits / temperature).clamp(_EPS, 1.0 - _EPS)
    q = torch.sigmoid(degraded_logits / temperature).clamp(_EPS, 1.0 - _EPS)
    kl_pq = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
    kl_qp = q * torch.log(q / p) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p))
    return 0.5 * (kl_pq + kl_qp).mean() * (temperature ** 2)


def dcpt_feature_consistency_loss(
    z_clean: torch.Tensor,
    z_degraded: torch.Tensor,
) -> torch.Tensor:
    """DCPT's feature-level term: ``1 - cosine_similarity`` of the embeddings.

    The penultimate embedding carries the representation the classifier
    reads. Constraining it to agree across a clean and a degraded view is
    what makes the model rely on structure rather than on the exact
    high-frequency fingerprint that compression destroys.
    """
    if z_clean.shape != z_degraded.shape:
        raise ValueError(
            "clean and degraded embeddings must have the same shape, got "
            f"{tuple(z_clean.shape)} and {tuple(z_degraded.shape)}"
        )
    return (1.0 - F.cosine_similarity(z_clean, z_degraded, dim=-1)).mean()


def dcpt_total_loss(
    clean_logits: torch.Tensor,
    degraded_logits: torch.Tensor,
    z_clean: torch.Tensor | None = None,
    z_degraded: torch.Tensor | None = None,
    logit_weight: float = 1.0,
    feature_weight: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Both DCPT terms, weighted."""
    total = logit_weight * dcpt_consistency_loss(clean_logits, degraded_logits, temperature)
    if (z_clean is None) != (z_degraded is None):
        raise ValueError("z_clean and z_degraded must be supplied together")
    if z_clean is not None and z_degraded is not None:
        total = total + feature_weight * dcpt_feature_consistency_loss(z_clean, z_degraded)
    return total
