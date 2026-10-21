"""2D Shift-Invariant Phase-Only Spectral Reconstruction.

Implements the phase-spectrum forensic feature extractor (Li et al., CVPR 2026).
Unlike power-spectrum slopes which degrade rapidly under JPEG re-quantization,
phase angles remain invariant to amplitude quantization.

Crucial math details:
1. DC component is centered via np.fft.fftshift before radial masks are applied.
2. Evaluates spatial statistics (kurtosis, HF energy ratio) over the phase-only
   spatial reconstruction to guarantee translation equivariance.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
from PIL import Image


def to_gray_u8(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to uint8 grayscale using ITU-R BT.601 integer math."""
    arr = np.asarray(img.convert("RGB"), dtype=np.uint32)
    # R * 77 + G * 150 + B * 29 >> 8
    gray = (arr[:, :, 0] * 77 + arr[:, :, 1] * 150 + arr[:, :, 2] * 29) >> 8
    return gray.astype(np.uint8)


def phase_spectrum_features(img: Image.Image, size: int = 256) -> Tuple[float, float]:
    """Extract shift-invariant 2D phase-only spatial residual features.

    Returns:
        kurtosis: Kurtosis of the full phase-only spatial reconstruction.
        hf_ratio: Ratio of high-frequency phase energy to total phase energy.
    """
    scaled = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    gray = to_gray_u8(scaled).astype(np.float64) / 255.0
    gray = gray - gray.mean()

    # 2D Fourier transform
    spectrum = np.fft.fft2(gray)
    # Unit-magnitude phase: e^{i * theta}
    unit_phase = np.exp(1j * np.angle(spectrum))

    # Phase-only reconstruction in spatial domain
    phase_img = np.fft.ifft2(unit_phase).real

    # High-frequency band: DC centered first via fftshift
    shifted = np.fft.fftshift(unit_phase)
    n = size
    cy, cx = n // 2, n // 2
    y, x = np.ogrid[:n, :n]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # High-frequency annular mask: r >= 0.25 * (n / 2)
    hf_mask = (r >= 0.25 * (n / 2.0)).astype(np.float64)
    hf_phase_img = np.fft.ifft2(np.fft.ifftshift(shifted * hf_mask)).real

    # Kurtosis of the phase-only reconstruction
    p2 = np.mean(phase_img ** 2)
    p4 = np.mean(phase_img ** 4)
    kurtosis = float(p4 / (p2 ** 2 + 1e-12))

    # High-frequency energy ratio
    hf_ratio = float(np.std(hf_phase_img) / (np.std(phase_img) + 1e-12))

    return kurtosis, hf_ratio
