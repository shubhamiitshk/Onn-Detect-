"""Native-resolution patch extraction and Neighboring Pixel Relationships (NPR).

Bilinear downsampling of high-resolution images acts as a low-pass filter
that destroys high-frequency 5x5 SRM residuals and VAE checkerboard artifacts.
This module extracts native-resolution patches at scale 1.0 (without resize)
prior to forensic filtering.
"""

from typing import Tuple, List
import numpy as np
import torch
from PIL import Image


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
    """Extract native-resolution patches without downsampling.

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

    # BT.601 luma for texture energy calculation
    gray = (
        np.array(rgb, dtype=np.float32)[..., 0] * 0.299
        + np.array(rgb, dtype=np.float32)[..., 1] * 0.587
        + np.array(rgb, dtype=np.float32)[..., 2] * 0.114
    ).astype(np.uint8)

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


def npr_residual(image_tensor: torch.Tensor) -> torch.Tensor:
    """Neighboring Pixel Relationships (NPR) operator (Tan et al., CVPR 2024).

    Computes local pairwise 2x2 pixel differences that expose deconvolution
    and flow-matching upsampling grid periodicities.
    Args:
        image_tensor: (B, C, H, W) normalized float tensor in [0, 1].
    Returns:
        (B, C, H-1, W-1) NPR difference map.
    """
    diff_h = torch.abs(image_tensor[:, :, 1:, :-1] - image_tensor[:, :, :-1, :-1])
    diff_w = torch.abs(image_tensor[:, :, :-1, 1:] - image_tensor[:, :, :-1, :-1])
    diff_d = torch.abs(image_tensor[:, :, 1:, 1:] - image_tensor[:, :, :-1, :-1])
    return (diff_h + diff_w + diff_d) / 3.0
