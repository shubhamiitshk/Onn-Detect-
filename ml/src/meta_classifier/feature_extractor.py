"""Canonical ten-signal feature extraction for fusion experiments.

RGB resize/feature equations are versioned with the checkpoint and release.
Slot 6 carries the measured semantic-probe score (Engine C, "Lite C") and is
``NaN`` (unavailable) when no validated probe ran. Slots 7-9 keep fixed
unsupported placeholders; unverified provenance or temporal measurements are
rejected. Python equations alone do not establish Android decoder parity.

Slot 6 history
--------------
The slot was reserved for a semantic vision backbone from the start
(``train_xgboost_meta.py --semantic-tflite``) but was held at the neutral
constant ``NEUTRAL_EXIF = 0.1`` because no second engine existed. A neutral
constant is only honest while nothing is ever measured there: ``0.1`` reads
as "low anomaly" to any tree that splits on it. Now that a real measurement
can occupy the slot, "probe did not run" must be *unavailable* (``NaN``), not
a plausible-looking number. The XGBoost tree learns a missing branch for
that case and the Kotlin transpiler emits it.
"""

import json
import os

import numpy as np
from PIL import Image
from data.image_frontend import decode_image, SOURCE_SIZE_KEY, SOURCE_FORMAT_KEY

# Bin edges identical to Android KLCooccurrenceAnalyzer: diffs in [-10, 10]
KL_BINS = 21
KL_CLAMP = 10
# Android maps score = kl / 0.3f coerced to [0,1]; keep identical here.
KL_SCORE_SCALE = 0.3

# Neutral authentic-side constants. MUST stay identical to
# DetectionPipeline.Companion.NEUTRAL_* in the Android app.
NEUTRAL_EXIF = 0.1
NEUTRAL_C2PA = 0.05
NEUTRAL_LIP_SYNC = 0.9
NEUTRAL_JITTER = 0.1

# Fusion slot indices. Slot 6 is the optional semantic-probe measurement.
SEMANTIC_PROBE_INDEX = 6
# The only slot permitted to be NaN ("evidence unavailable"). Every other slot
# must carry a finite measurement or a fixed compatibility constant.
OPTIONAL_FEATURE_INDICES = (SEMANTIC_PROBE_INDEX,)

DEFAULT_BASELINE = [
    0.005, 0.007, 0.012, 0.018, 0.025, 0.038, 0.055, 0.082, 0.120, 0.150,
    0.180,
    0.150, 0.120, 0.082, 0.055, 0.038, 0.025, 0.018, 0.012, 0.007, 0.005,
]

FEATURE_NAMES = [
    "srm_score", "fft_score", "kl_score", "flatness_score",
    "resolution_norm", "is_compressed",
    "semantic_probe_score", "c2pa_present", "lip_sync_corr", "landmark_jitter"
]


def to_gray_u8(img: Image.Image) -> np.ndarray:
    """ITU-R BT.601 luma, matching Android's (r*299 + g*587 + b*114)/1000."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float64)
    gray = (rgb[..., 0] * 299 + rgb[..., 1] * 587 + rgb[..., 2] * 114) / 1000.0
    return np.clip(gray, 0, 255).astype(np.uint8)


def diff_histogram(gray: np.ndarray) -> np.ndarray:
    """21-bin normalized histogram of horizontal adjacent pixel differences.

    Mirrors Android KLCooccurrenceAnalyzer.analyze(): clamp diffs to [-10, 10],
    bin index = diff + 10, normalize, add 1e-6 epsilon, renormalize.

    No adjacent pixel pair is UNAVAILABLE evidence, never a uniform
    distribution. Returning a constant vector here would sum to 21e-6 rather
    than 1.0 and would then be consumed by ``kl_score`` as if it were a
    probability distribution; Android refuses the same input instead
    (KLCooccurrenceAnalyzer: "Insufficient pixel differences").
    """
    if gray.ndim != 2 or gray.shape[1] < 2:
        raise ValueError(f"No adjacent pixel pairs in grayscale array of shape {gray.shape}")
    diffs = gray[:, :-1].astype(np.int32) - gray[:, 1:].astype(np.int32)
    clipped = np.clip(diffs, -KL_CLAMP, KL_CLAMP) + KL_CLAMP
    hist = np.bincount(clipped.ravel(), minlength=KL_BINS).astype(np.float64)
    total = hist.sum()
    if total == 0:
        raise ValueError("Empty adjacent-pixel difference histogram")
    hist = hist / total + 1e-6
    return hist / hist.sum()


def load_baseline(path: str | None) -> np.ndarray:
    """Load calibrated natural-diff histogram JSON; fall back to hardcoded prior."""
    if path and os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        arr = np.asarray(data["bins"] if isinstance(data, dict) else data, dtype=np.float64)
        if arr.shape == (KL_BINS,) and arr.sum() > 0:
            return arr / arr.sum()
    base = np.asarray(DEFAULT_BASELINE, dtype=np.float64)
    return base / base.sum()


def kl_score(gray: np.ndarray, baseline: np.ndarray) -> tuple[float, float]:
    """Returns (score, raw_kl). Mirrors the Android analyzer mapping."""
    p = diff_histogram(gray)
    q = baseline / baseline.sum()
    kl = float(np.sum(p * np.log(p / q)))
    score = min(max(kl / KL_SCORE_SCALE, 0.0), 1.0)
    return score, kl


def calibrate_baseline(image_paths: list[str], sample_limit: int | None = None) -> dict:
    """Average normalized diff histogram over real photographs -> JSON-ready dict."""
    paths = image_paths[:sample_limit] if sample_limit else image_paths
    acc = np.zeros(KL_BINS, dtype=np.float64)
    used = 0
    for p in paths:
        try:
            img = decode_image(p).image
            acc += diff_histogram(to_gray_u8(img))
            used += 1
        except Exception as exc:
            raise RuntimeError(f"Unable to decode baseline sample {p}") from exc
    if used == 0:
        raise RuntimeError("No images available for baseline calibration")
    acc /= acc.sum()
    return {"bins": [float(x) for x in acc], "num_images": used}


def fft2_power_spectrum(gray01: np.ndarray) -> np.ndarray:
    """Centered log-scaled 2D power spectrum of a square grayscale array in [0,1]."""
    f = np.fft.fft2(gray01)
    fshift = np.fft.fftshift(f)
    ps = np.abs(fshift) ** 2
    return ps


def azimuthal_average(ps: np.ndarray) -> np.ndarray:
    """Radial average identical to Android computeAzimuthalAverage()."""
    n = ps.shape[0]
    center = n // 2
    max_radius = int(n / 2.0 * np.sqrt(2.0))
    radial_sum = np.zeros(max_radius + 1)
    radial_count = np.zeros(max_radius + 1, dtype=np.int64)
    y, x = np.indices((n, n))
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2).astype(int)
    mask = r <= max_radius
    np.add.at(radial_sum, r[mask], ps[mask])
    np.add.at(radial_count, r[mask], 1)
    avg = np.zeros_like(radial_sum)
    nz = radial_count > 0
    avg[nz] = radial_sum[nz] / radial_count[nz]
    return avg


def power_law_slope(radial: np.ndarray) -> float:
    """OLS log-log slope, skipping DC and r<2 - mirrors Android implementation.

    Matches FFTSpectralAnalyzer.computePowerLawSlope, which raises
    ("Insufficient spectral evidence" / "Degenerate spectrum") rather than
    inventing a slope.

    An earlier version returned the natural slope -2.0 here. That value
    reaches the feature vector through ``abs(slope - (-2.0)) / 2`` and becomes
    fft_score = 0.0, i.e. MAXIMALLY AUTHENTIC under real=0_fake=1, for an
    image that produced no usable spectrum at all. Missing evidence is
    unavailable, not zero.
    """
    freqs, powers = [], []
    for i in range(2, len(radial)):
        if radial[i] > 1e-10:
            freqs.append(np.log(i))
            powers.append(np.log(radial[i]))
    if len(freqs) < 2:
        raise ValueError(f"Insufficient spectral evidence: {len(freqs)} usable radial bins")
    fx, fy = np.asarray(freqs), np.asarray(powers)
    denom = np.sum(fx**2) - len(fx) * fx.mean() ** 2
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("Degenerate spectrum: OLS denominator is not positive")
    slope = float((np.sum(fx * fy) - len(fx) * fx.mean() * fy.mean()) / denom)
    if not np.isfinite(slope):
        raise ValueError("Degenerate spectrum: fitted slope is not finite")
    return slope


def wiener_entropy(ps: np.ndarray) -> float:
    """Geometric mean / arithmetic mean of the power spectrum."""
    vals = ps[ps > 1e-10]
    if vals.size == 0 or vals.sum() == 0:
        return 0.0
    return float(np.exp(np.log(vals).mean()) / vals.mean())


def fft_features(img: Image.Image, size: int = 256) -> tuple[float, float]:
    """(fft_score, flatness_score) mirroring Android FFTSpectralAnalyzer.

    Both use bilinear filtering; native codec/resize parity must be measured.
    Score = min(|slope - (-2)| / 2, 1).
    """
    scaled = img.convert("RGB").resize((size, size), Image.BILINEAR)
    gray = to_gray_u8(scaled).astype(np.float64) / 255.0
    ps = fft2_power_spectrum(gray)
    radial = azimuthal_average(ps)
    slope = power_law_slope(radial)
    deviation = abs(slope - (-2.0))
    score = min(deviation / 2.0, 1.0)
    flatness = wiener_entropy(ps)
    return score, flatness


def quality_features(img: Image.Image, path: str, source_size=None) -> tuple[float, float]:
    """(resolution_norm, is_compressed) from real file properties."""
    w, h = source_size or img.info.get(SOURCE_SIZE_KEY, img.size)
    resolution_norm = min(min(w, h) / 1024.0, 1.0)
    encoding = img.info.get(SOURCE_FORMAT_KEY) or img.format
    if not encoding:
        # File signatures, not suffixes, determine the compressed feature.
        with Image.open(path) as source:
            encoding = source.format
    compressed = 1.0 if encoding == 'JPEG' else 0.0
    return resolution_norm, compressed


def phase_spectrum_features(img: Image.Image, size: int = 256) -> tuple[float, float]:
    """Shift-invariant 2D phase-only spatial residual.

    JPEG quantization heavily damps high-frequency AMPLITUDE but leaves the
    phase angles intact, so a phase-only reconstruction survives re-encoding
    where a power-spectrum slope does not. The shipped detector's
    ``fft_score`` measures exactly that amplitude slope, and it collapses to
    chance under laundering (``ml/scripts/llm_signal_analysis_results.json``:
    standalone AUC 0.5013).

    Two details matter and are easy to get wrong:

    * the DC term must be moved to the centre with ``fftshift`` before any
      radial statistic is computed, otherwise the radii are centred on the
      Nyquist corner rather than on DC;
    * the phase must be turned back into an IMAGE (the phase-only
      reconstruction) rather than reduced to ``cos(phase)`` directly,
      because a translated image acquires a linear phase ramp that
      ``cos(phase)`` does not tolerate.

    Following Li et al., CVPR 2026 ("Detecting Compressed AI-Generated
    Images via Phase Spectrum Robustness").

    The implementation lives in :mod:`meta_classifier.phase_features` so
    there is exactly ONE version of this maths in the repository. An earlier
    copy lived here and used a "high frequency" band that started at 0.25
    of Nyquist, which keeps ~94% of the spectrum: the reconstruction was
    then proportional to the full one and the ratio between them was
    constant to floating-point precision, i.e. the feature carried no
    information at all.

    Returns ``(kurtosis, high_frequency_ratio)``.
    """
    from meta_classifier.phase_features import phase_candidates

    candidates = phase_candidates(img)
    return candidates["phase_kurtosis"], candidates["phase_hf_energy_ratio"]


def extract_visual_features(
    img_path: str,
    srm_model,
    device,
    baseline: np.ndarray,
    provenance: dict | None = None,
    input_size: int = 32,
    semantic_probe_score: float | None = None,
    frontend_id: str | None = None,
) -> np.ndarray:
    """Full canonical feature vector for one image file.

    ``provenance`` is reserved and rejects supplied measurements. Unsupported
    features stay at the fixed constants shared with the Android pipeline.

    ``semantic_probe_score`` is the Engine C ("Lite C") measurement for slot 6.
    ``None`` means the probe did not run and the slot is UNAVAILABLE, which
    is encoded as ``NaN`` - never as a neutral constant. A neutral value
    would be read by the fusion tree as "low anomaly" evidence that was
    never measured.
    """
    # Reject supplied provenance BEFORE any decode, import or inference
    # work, so an unverified measurement can never reach a feature vector
    # even transiently.
    #
    # This guard must run BEFORE the torch/data imports below. The import
    # used to come first, so a contract test that asserts this rejection
    # crashed with ModuleNotFoundError in environments without PyTorch,
    # instead of raising the ValueError it is testing for.
    if provenance:
        raise ValueError("Unverified provenance/temporal features must not influence the image verdict")

    from data.image_dataset import image_tensor
    import torch

    decoded = decode_image(img_path)
    img = decoded.image

    # Match the checkpoint/mobile frontend contract (PatchCraft mosaic vs legacy bilinear).
    if frontend_id and "-patchcraft-mosaic-tile32" in frontend_id:
        from train_srm_dcpt_patchcraft import extract_patchcraft_mosaic
        tile = 32 if input_size >= 64 and input_size % 32 == 0 else max(8, input_size // 4)
        srm_img = extract_patchcraft_mosaic(img, mosaic_size=input_size, tile_size=tile, select="rich")
        arr = np.asarray(srm_img.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(device)
    else:
        tensor = image_tensor(img, input_size).unsqueeze(0).to(device)
    with torch.no_grad():
        srm = float(srm_model(tensor).item())

    fscore, flatness = fft_features(img)
    kscore, _ = kl_score(to_gray_u8(img), baseline)
    res_norm, comp = quality_features(img, img_path, decoded.source_size)

    # Slot 6: measured semantic-probe score, or NaN when unavailable.
    if semantic_probe_score is None:
        probe = float("nan")
    else:
        probe = float(semantic_probe_score)
        if not np.isfinite(probe):
            raise ValueError("Semantic probe score must be finite when supplied")
        if not 0.0 <= probe <= 1.0:
            raise ValueError(f"Semantic probe score out of range: {probe}")

    # Features 7-9 are FIXED unsupported neutral values, not measurements.
    prov = {}
    c2pa_present = float(prov.get("c2pa_present", NEUTRAL_C2PA))
    lip_sync_corr = float(prov.get("lip_sync_corr", NEUTRAL_LIP_SYNC))
    landmark_jitter = float(prov.get("landmark_jitter", NEUTRAL_JITTER))

    return np.array([
        srm, fscore, kscore, flatness, res_norm, comp,
        probe, c2pa_present, lip_sync_corr, landmark_jitter
    ], dtype=np.float32)
