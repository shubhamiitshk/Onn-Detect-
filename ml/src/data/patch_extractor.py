"""Native-resolution patch extraction, Multiple Instance Learning (MIL) and localization.

Bilinear downsampling of high-resolution images acts as a low-pass filter
that destroys high-frequency 5x5 SRM residuals and VAE checkerboard artifacts.
This module extracts native-resolution patches at scale 1.0 (without resize)
prior to forensic filtering, and provides calibrated multi-instance aggregation
and localization evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


@dataclass
class MultiInstancePatches:
    """Container for multi-instance extracted native patches and spatial metadata."""
    patches: List[Image.Image]
    boxes: List[Tuple[int, int, int, int]]  # (y0, x0, y1, x1) in native image coordinates
    energies: List[float]
    coverage_ratio: float  # Fraction of original image area covered by union of patches
    image_size: Tuple[int, int]  # (width, height) of native image


def compute_texture_energy(gray: np.ndarray, patch_size: int = 128, stride: int = 64) -> Tuple[int, int]:
    """Find (top, left) coordinates of the highest-texture region using Laplacian variance."""
    h, w = gray.shape
    if h <= patch_size or w <= patch_size:
        return 0, 0

    best_var = -1.0
    best_coord = (0, 0)

    # 3x3 Laplacian kernel
    lap = (
        -4 * gray[1:-1, 1:-1].astype(np.float32)
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            sub_lap = lap[y : y + patch_size - 2, x : x + patch_size - 2]
            var = float(np.var(sub_lap))
            if var > best_var:
                best_var = var
                best_coord = (y, x)

    return best_coord


def extract_native_patches(
    image: Image.Image, patch_size: int = 224, num_patches: int = 2
) -> List[Image.Image]:
    """Extract native-resolution patches without downsampling (backward-compatible).

    Returns:
        num_patches PIL Images of size (patch_size, patch_size).
        Patch 0: highest-texture patch (richest high-frequency forensic signal).
        Patch 1: center patch (subject/focal region).
    """
    rgb = image.convert("RGB")
    w, h = rgb.size

    # If image is smaller than target, pad reflectively rather than downsampling
    if w < patch_size or h < patch_size:
        pad_w = max(0, patch_size - w)
        pad_h = max(0, patch_size - h)
        padded = np.pad(
            np.array(rgb),
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
            mode="reflect",
        )
        rgb = Image.fromarray(padded)
        w, h = rgb.size

    # Shared integer BT.601 luma for texture energy calculation
    arr = np.asarray(rgb, dtype=np.int32)
    gray = ((arr[..., 0] * 299 + arr[..., 1] * 587 + arr[..., 2] * 114) // 1000).astype(np.uint8)

    patches = []

    # 1. Richest texture patch
    top_t, left_t = compute_texture_energy(gray, patch_size=patch_size)
    patches.append(rgb.crop((left_t, top_t, left_t + patch_size, top_t + patch_size)))

    if num_patches > 1:
        # 2. Central patch
        top_c = max(0, (h - patch_size) // 2)
        left_c = max(0, (w - patch_size) // 2)
        patches.append(rgb.crop((left_c, top_c, left_c + patch_size, top_c + patch_size)))

    return patches


def extract_candidate_patches(
    image: Image.Image,
    patch_size: int = 224,
    stride: int = 112,
    max_patches: int = 8,
    ranking_metric: str = "gradient",
) -> MultiInstancePatches:
    """Extract native-resolution candidate patches budgeted for MIL detection.

    Args:
        image: Native PIL Image at scale 1.0.
        patch_size: Receptive field size (default 224 for SigLIP / SRM compatibility).
        stride: Stride between candidate windows (default 112 for 50% overlap).
        max_patches: Maximum number of patches to select (budgeted by device compute).
        ranking_metric: Cues to rank patches:
            "gradient": Integer BT.601 luma L1 gradient energy (bit-compatible with PatchCraft).
            "laplacian": 3x3 Laplacian variance (high-frequency residual variance).

    Returns:
        MultiInstancePatches containing cropped images, bounding boxes, rankings,
        and canvas coverage ratio.
    """
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if max_patches <= 0:
        raise ValueError(f"max_patches must be positive, got {max_patches}")

    rgb = image.convert("RGB")
    orig_w, orig_h = rgb.size

    # Pad reflectively if smaller than patch_size
    pad_w = max(0, patch_size - orig_w)
    pad_h = max(0, patch_size - orig_h)
    if pad_w > 0 or pad_h > 0:
        padded = np.pad(
            np.asarray(rgb),
            ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
            mode="reflect",
        )
        working_rgb = Image.fromarray(padded)
    else:
        working_rgb = rgb

    w, h = working_rgb.size
    arr = np.asarray(working_rgb, dtype=np.int32)
    luma = (arr[..., 0] * 299 + arr[..., 1] * 587 + arr[..., 2] * 114) // 1000

    candidates: List[Tuple[float, int, int, int, int]] = []  # (energy, y0, x0, y1, x1)

    y_steps = list(range(0, h - patch_size + 1, stride))
    if not y_steps or y_steps[-1] != h - patch_size:
        y_steps.append(h - patch_size)

    x_steps = list(range(0, w - patch_size + 1, stride))
    if not x_steps or x_steps[-1] != w - patch_size:
        x_steps.append(w - patch_size)

    for y0 in y_steps:
        y1 = y0 + patch_size
        for x0 in x_steps:
            x1 = x0 + patch_size
            sub = luma[y0:y1, x0:x1]

            if ranking_metric == "gradient":
                # L1 gradient energy
                gx = np.abs(sub[:, 1:] - sub[:, :-1]).sum()
                gy = np.abs(sub[1:, :] - sub[:-1, :]).sum()
                energy = float(gx + gy)
            elif ranking_metric == "laplacian":
                lap = (
                    -4 * sub[1:-1, 1:-1].astype(np.float32)
                    + sub[:-2, 1:-1]
                    + sub[2:, 1:-1]
                    + sub[1:-1, :-2]
                    + sub[1:-1, 2:]
                )
                energy = float(np.var(lap))
            else:
                raise ValueError(f"Unknown ranking_metric {ranking_metric!r}")

            candidates.append((energy, y0, x0, y1, x1))

    # Deterministic sorting: highest energy first, breaking ties with y0, x0
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    selected = candidates[:max_patches]

    # Compute canvas coverage ratio (union mask of selected boxes)
    mask = np.zeros((h, w), dtype=bool)
    patches: List[Image.Image] = []
    boxes: List[Tuple[int, int, int, int]] = []
    energies: List[float] = []

    for energy, y0, x0, y1, x1 in selected:
        patches.append(working_rgb.crop((x0, y0, x1, y1)))
        boxes.append((y0, x0, y1, x1))
        energies.append(energy)
        mask[y0:y1, x0:x1] = True

    coverage = float(mask[:orig_h, :orig_w].sum()) / float(orig_w * orig_h)

    return MultiInstancePatches(
        patches=patches,
        boxes=boxes,
        energies=energies,
        coverage_ratio=coverage,
        image_size=(orig_w, orig_h),
    )


def npr_residual(image_tensor: torch.Tensor) -> torch.Tensor:
    """Neighboring Pixel Relationships (NPR) operator (Tan et al., CVPR 2024).

    Computes local pairwise 2x2 pixel differences that expose deconvolution
    and flow-matching upsampling grid periodicities.
    Args:
        image_tensor: (B, C, H, W) normalized float tensor in [0, 1].
    Returns:
        (B, C, H-1, W-1) NPR difference map.
    """
    if not _TORCH_AVAILABLE or not isinstance(image_tensor, torch.Tensor):
        raise RuntimeError("npr_residual requires PyTorch tensor input")
    diff_h = torch.abs(image_tensor[:, :, 1:, :-1] - image_tensor[:, :, :-1, :-1])
    diff_w = torch.abs(image_tensor[:, :, :-1, 1:] - image_tensor[:, :, :-1, :-1])
    diff_d = torch.abs(image_tensor[:, :, 1:, 1:] - image_tensor[:, :, :-1, :-1])
    return (diff_h + diff_w + diff_d) / 3.0


# ============================================================================
# Multi-Instance Learning (MIL) Aggregation Operators
# ============================================================================


def top_k_pooling(
    logits: Union[torch.Tensor, np.ndarray, Sequence[float]],
    k: int = 3,
    dim: int = -1,
) -> Union[torch.Tensor, float, np.ndarray]:
    """Top-k logit pooling over candidate patches.

    Averages the top-k highest logits in the bag:
        pooled = (1 / k) * sum_{i=1}^k z_(i)
    Avoids uncalibrated max-pooling (which explodes FPR as bag size grows)
    while remaining sensitive to localized forensic manipulations.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if _TORCH_AVAILABLE and isinstance(logits, torch.Tensor):
        eff_k = min(k, logits.shape[dim])
        vals, _ = torch.topk(logits, k=eff_k, dim=dim)
        return vals.mean(dim=dim)

    arr = np.asarray(logits, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Cannot pool empty logits")
    eff_k = min(k, arr.shape[dim])
    # Partition top k elements
    top_vals = np.take(np.partition(arr, -eff_k, axis=dim), range(-eff_k, 0), axis=dim)
    return float(np.mean(top_vals)) if arr.ndim == 1 else np.mean(top_vals, axis=dim)


def log_sum_exp_pooling(
    logits: Union[torch.Tensor, np.ndarray, Sequence[float]],
    tau: float = 2.0,
    dim: int = -1,
) -> Union[torch.Tensor, float, np.ndarray]:
    """Smooth Log-Sum-Exp (LSE) logit pooling with finite temperature tau.

    Interpolates smoothly between mean pooling (tau -> inf) and max pooling (tau -> 0):
        pooled = tau * ln( (1/N) * sum_{i=1}^N exp(z_i / tau) )
    Uses numerically stable max-subtraction:
        pooled = m + tau * ln( (1/N) * sum_{i=1}^N exp((z_i - m) / tau) )
    """
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError(f"tau must be positive and finite, got {tau}")

    if _TORCH_AVAILABLE and isinstance(logits, torch.Tensor):
        n = logits.shape[dim]
        m, _ = torch.max(logits, dim=dim, keepdim=True)
        shifted = (logits - m) / tau
        pooled = m + tau * torch.log(torch.mean(torch.exp(shifted), dim=dim, keepdim=True))
        return pooled.squeeze(dim)

    arr = np.asarray(logits, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Cannot pool empty logits")
    n = arr.shape[dim]
    m = np.max(arr, axis=dim, keepdims=True)
    shifted = (arr - m) / tau
    pooled = m + tau * np.log(np.mean(np.exp(shifted), axis=dim, keepdims=True))
    res = np.squeeze(pooled, axis=dim)
    return float(res) if res.ndim == 0 else res


def noisy_or_pooling(
    probs: Union[torch.Tensor, np.ndarray, Sequence[float]],
    calibrate_for_n: bool = True,
    dim: int = -1,
) -> Union[torch.Tensor, float, np.ndarray]:
    """Noisy-OR aggregation of per-patch probabilities.

    Standard Noisy-OR: 1 - prod_{i=1}^N (1 - p_i).
    If calibrate_for_n=True, applies N-dependent calibration penalty to prevent
    the false-positive rate from scaling uncontrollably with the number of patches N:
        p_cal = 1 - (prod (1 - p_i)) ** (1 / N)
    """
    eps = 1e-7
    if _TORCH_AVAILABLE and isinstance(probs, torch.Tensor):
        p = torch.clamp(probs, eps, 1.0 - eps)
        log_comp = torch.log(1.0 - p)
        sum_log = torch.sum(log_comp, dim=dim)
        n = probs.shape[dim]
        if calibrate_for_n and n > 0:
            sum_log = sum_log / float(n)
        return 1.0 - torch.exp(sum_log)

    arr = np.clip(np.asarray(probs, dtype=np.float32), eps, 1.0 - eps)
    if arr.size == 0:
        raise ValueError("Cannot pool empty probabilities")
    sum_log = np.sum(np.log(1.0 - arr), axis=dim)
    n = arr.shape[dim]
    if calibrate_for_n and n > 0:
        sum_log = sum_log / float(n)
    res = 1.0 - np.exp(sum_log)
    return float(res) if res.ndim == 0 else res


if _TORCH_AVAILABLE:
    class GatedAttentionMIL(nn.Module):
        """Attention-based Deep Multiple Instance Learning (Ilse et al., ICML 2018).

        Computes data-dependent soft attention weights over candidate patch embeddings:
            a_i = softmax( w^T (tanh(V h_i) * sigmoid(U h_i)) )
        Returns both the aggregated bag representation/logit and the spatial attention
        heatmap weights for forensic interpretability.
        """

        def __init__(self, in_features: int, hidden_dim: int = 64):
            super().__init__()
            self.v = nn.Linear(in_features, hidden_dim)
            self.u = nn.Linear(in_features, hidden_dim)
            self.w = nn.Linear(hidden_dim, 1)

        def forward(
            self,
            patch_embeddings: torch.Tensor,
            patch_logits: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                patch_embeddings: (B, N, in_features)
                patch_logits: (B, N, 1) or (B, N)
            Returns:
                bag_logit: (B, 1)
                attention_weights: (B, N) summing to 1.0 per bag
            """
            if patch_logits.ndim == 2:
                patch_logits = patch_logits.unsqueeze(-1)
            # Gated attention mechanism
            a_v = torch.tanh(self.v(patch_embeddings))
            a_u = torch.sigmoid(self.u(patch_embeddings))
            gated = self.w(a_v * a_u)  # (B, N, 1)
            weights = F.softmax(gated, dim=1)  # (B, N, 1)
            bag_logit = torch.sum(weights * patch_logits, dim=1)  # (B, 1)
            return bag_logit, weights.squeeze(-1)


# ============================================================================
# Localization Evaluation & Partial Manipulation Metrics
# ============================================================================


def patch_mask_iou(
    boxes: Sequence[Tuple[int, int, int, int]],
    scores: Sequence[float],
    gt_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute Intersection-over-Union (IoU) between patch detections and ground-truth mask.

    Args:
        boxes: List of (y0, x0, y1, x1) bounding boxes.
        scores: Predicted forgery probabilities per patch.
        gt_mask: Binary 2D numpy array (H, W) where True/1 indicates manipulated pixels.
        threshold: Score threshold above which a patch is considered positive detection.

    Returns:
        IoU in [0.0, 1.0].
    """
    h, w = gt_mask.shape[:2]
    pred_mask = np.zeros((h, w), dtype=bool)

    for box, score in zip(boxes, scores):
        if score >= threshold:
            y0, x0, y1, x1 = box
            y0_c = max(0, min(h, y0))
            y1_c = max(0, min(h, y1))
            x0_c = max(0, min(w, x0))
            x1_c = max(0, min(w, x1))
            pred_mask[y0_c:y1_c, x0_c:x1_c] = True

    gt_binary = gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask, gt_binary).sum()
    union = np.logical_or(pred_mask, gt_binary).sum()

    if union == 0:
        return 1.0 if not np.any(gt_binary) and not np.any(pred_mask) else 0.0
    return float(intersection) / float(union)


def evaluate_small_manipulations(
    records: Sequence[Dict[str, Union[float, np.ndarray, bool]]],
    area_bins: Tuple[Tuple[float, float], ...] = (
        (0.01, 0.05),   # 1-5% micro-inpainting / splicing
        (0.05, 0.20),   # 5-20% localized region edits
        (0.20, 1.00),   # >20% extensive edits
    ),
    score_key: str = "score",
    mask_key: str = "gt_mask",
    threshold: float = 0.5,
) -> Dict[str, Dict[str, Union[int, float]]]:
    """Evaluate detector sensitivity stratified by partial edit area fraction.

    Per Research Upgrade Audit section 3 & 11: prevents global metrics from masking
    failures on localized 1-20% partial edits.
    """
    results: Dict[str, Dict[str, Union[int, float]]] = {}

    for low, high in area_bins:
        bin_label = f"{int(low * 100)}%-{int(high * 100)}%"
        total = 0
        detected = 0

        for rec in records:
            mask = rec.get(mask_key)
            if mask is None:
                continue
            gt_arr = np.asarray(mask, dtype=bool)
            area_frac = float(gt_arr.sum()) / float(gt_arr.size)

            if low <= area_frac < high:
                total += 1
                score = float(rec.get(score_key, 0.0))
                if score >= threshold:
                    detected += 1

        tpr = (float(detected) / float(total)) if total > 0 else float("nan")
        results[bin_label] = {
            "total_samples": total,
            "detected_samples": detected,
            "tpr": tpr,
        }

    return results
