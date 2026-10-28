"""Phase-spectrum feature candidates.

JPEG quantization destroys high-frequency AMPLITUDE but leaves the PHASE
angles intact, so a phase-derived feature should degrade far less under
re-encoding than the power-spectrum slope the fusion currently uses
(the shipped ``fft_score`` measures 0.5140 standalone AUC - chance - and
the CVPR 2026 phase-spectrum result is the standard to beat).

Several candidate summaries are provided because a SCALAR summary of the
phase spectrum is not obviously the right instrument: Li et al. feed the
phase into a cross-modal transformer, and a scalar hand-summary may or
may not carry the signal. ``evaluate_analytic_features.py`` measures every
candidate on the sealed split, clean and laundered, so the choice is made
on evidence rather than on plausibility.

Conventions that must be respected (each one was a bug at least once):

  * the DC term must be moved to the centre with ``fftshift`` before any
    radial statistic is computed, otherwise the radii are centred on the
    Nyquist corner rather than on DC;
  * the phase must be turned back into an IMAGE (the phase-only
    reconstruction) rather than reduced to ``cos(phase)`` directly, because
    a translated image acquires a linear phase ramp that ``cos(phase)``
    does not tolerate;
  * a "high frequency" band must be an OUTER annulus. A band that starts
    near DC keeps almost the whole spectrum, the reconstruction is then
    proportional to the full one, and the ratio between them is constant
    to floating-point precision - i.e. the feature carries no information
    at all.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# The outer band used for the high-frequency statistics. 0.5 of Nyquist
# keeps a genuine annulus rather than "everything except the middle".
HF_INNER_FRACTION = 0.5

# Analysis resolution. Matches `fft_features`, so the phase candidates are
# measured on the same footing as the power-spectrum features they are
# meant to replace.
ANALYSIS_SIZE = 256


def _grey01(img: Image.Image) -> np.ndarray:
    """Square, mean-removed greyscale in [0, 1].

    Uses the same ITU-R BT.601 luma as ``feature_extractor.to_gray_u8``
    so phase candidates are directly comparable to the power-spectrum features.
    """
    resized = img.convert("RGB").resize((ANALYSIS_SIZE, ANALYSIS_SIZE), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float64)
    arr = (rgb[..., 0] * 299.0 + rgb[..., 1] * 587.0 + rgb[..., 2] * 114.0) / 1000.0 / 255.0
    return arr - arr.mean()


def phase_only_image(grey: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The phase-only spatial reconstruction and its unit-magnitude phase.

    Returns ``(reconstruction, unit_phase)`` where ``unit_phase`` is
    fftshift-ed so that DC sits at the centre.
    """
    spectrum = np.fft.fft2(grey)
    unit_phase = np.exp(1j * np.angle(spectrum))
    reconstruction = np.fft.ifft2(unit_phase).real
    return reconstruction, np.fft.fftshift(unit_phase)


def _radial_profile(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthal average of a square, centred power spectrum."""
    if power.ndim != 2 or power.shape[0] != power.shape[1]:
        raise ValueError(f"Power spectrum must be square, got {power.shape}")
    n = power.shape[0]
    centre = n // 2
    max_radius = int(n / 2.0 * np.sqrt(2.0))
    y, x = np.indices((n, n))
    radius = np.sqrt((x - centre) ** 2 + (y - centre) ** 2).astype(int)
    mask = radius <= max_radius
    sums = np.zeros(max_radius + 1)
    counts = np.zeros(max_radius + 1, dtype=np.int64)
    np.add.at(sums, radius[mask], power[mask])
    np.add.at(counts, radius[mask], 1)
    profile = np.zeros(max_radius + 1)
    nonzero = counts > 0
    profile[nonzero] = sums[nonzero] / counts[nonzero]
    return profile, counts


def _loglog_slope(profile: np.ndarray, counts: np.ndarray) -> float:
    """OLS slope of the radial profile in log-log space."""
    freqs, powers = [], []
    for i in range(2, len(profile)):
        if counts[i] > 0 and profile[i] > 1e-10:
            freqs.append(np.log(i))
            powers.append(np.log(profile[i]))
    if len(freqs) < 2:
        return 0.0
    fx = np.asarray(freqs)
    fy = np.asarray(powers)
    denominator = float(np.sum(fx ** 2) - len(fx) * fx.mean() ** 2)
    if denominator <= 0:
        return 0.0
    return float((np.sum(fx * fy) - len(fx) * fx.mean() * fy.mean()) / denominator)


def phase_candidates(img: Image.Image) -> dict[str, float]:
    """Every phase-derived candidate summary.

    All of them are computed from ONE phase-only reconstruction, so the
    whole set costs a single FFT.
    """
    grey = _grey01(img)
    reconstruction, unit_phase = phase_only_image(grey)
    n = grey.shape[0]
    centre = n // 2

    # Radial profile of the PHASE-ONLY reconstruction's power spectrum.
    power = np.abs(np.fft.fft2(reconstruction)) ** 2
    profile, counts = _radial_profile(power)
    slope = _loglog_slope(profile, counts)

    # Genuine outer annulus: radius from HF_INNER_FRACTION * Nyquist out.
    y, x = np.indices((n, n))
    radius = np.sqrt((x - centre) ** 2 + (y - centre) ** 2)
    hf_mask = radius >= HF_INNER_FRACTION * (n / 2.0)
    hf_unit = np.fft.ifftshift(unit_phase * hf_mask)
    hf_reconstruction = np.fft.ifft2(hf_unit).real

    total_energy = float(np.sum(reconstruction ** 2))
    hf_energy = float(np.sum(hf_reconstruction ** 2))
    hf_ratio = hf_energy / (total_energy + 1e-12)

    # Kurtosis of the phase-only reconstruction.
    variance = float(np.mean(reconstruction ** 2))
    kurtosis = float(np.mean(reconstruction ** 4) / (variance ** 2 + 1e-12))

    # Std ratio between the outer-band reconstruction and the full one.
    std_ratio = float(np.std(hf_reconstruction) / (np.std(reconstruction) + 1e-12))

    # Circular statistics of the phase in the outer band: a generated image
    # tends to have a more structured (less uniform) phase distribution.
    outer_phase = np.angle(np.fft.ifftshift(unit_phase * hf_mask))
    resultant = float(np.abs(np.mean(np.exp(1j * outer_phase))))

    return {
        "phase_radial_slope": slope,
        "phase_hf_energy_ratio": hf_ratio,
        "phase_kurtosis": kurtosis,
        "phase_std_ratio": std_ratio,
        "phase_outer_resultant": resultant,
    }
