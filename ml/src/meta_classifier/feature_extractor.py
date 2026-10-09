"""Canonical ten-signal feature extraction for fusion experiments.

RGB resize/feature equations are versioned with the checkpoint and release.
Features 6-9 use fixed unsupported placeholders; unverified provenance or
temporal measurements are rejected. Python equations alone do not establish
Android decoder parity.
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

DEFAULT_BASELINE = [
    0.005, 0.007, 0.012, 0.018, 0.025, 0.038, 0.055, 0.082, 0.120, 0.150,
    0.180,
    0.150, 0.120, 0.082, 0.055, 0.038, 0.025, 0.018, 0.012, 0.007, 0.005,
]

FEATURE_NAMES = [
    "srm_score", "fft_score", "kl_score", "flatness_score",
    "resolution_norm", "is_compressed",
    "exif_anomaly", "c2pa_present", "lip_sync_corr", "landmark_jitter"
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


def extract_visual_features(
    img_path: str,
    srm_model,
    device,
    baseline: np.ndarray,
    provenance: dict | None = None,
    input_size: int = 32,
) -> np.ndarray:
    """Full canonical feature vector for one image file.

    ``provenance`` is reserved and rejects supplied measurements. Unsupported
    features stay at the fixed constants shared with the Android pipeline.
    """
    from data.image_dataset import image_tensor
    import torch

    # Reject supplied provenance BEFORE any decode or inference work, so an
    # unverified measurement can never reach a feature vector even transiently.
    if provenance:
        raise ValueError("Unverified provenance/temporal features must not influence the image verdict")

    decoded = decode_image(img_path)
    img = decoded.image

    # Match the checkpoint/mobile square resize contract.
    tensor = image_tensor(img, input_size).unsqueeze(0).to(device)
    with torch.no_grad():
        srm = float(srm_model(tensor).item())

    fscore, flatness = fft_features(img)
    kscore, _ = kl_score(to_gray_u8(img), baseline)
    res_norm, comp = quality_features(img, img_path, decoded.source_size)

    # Features 6-9 are FIXED unsupported neutral values, not measurements.
    prov = {}
    exif_anomaly = float(prov.get("exif_anomaly", NEUTRAL_EXIF))
    c2pa_present = float(prov.get("c2pa_present", NEUTRAL_C2PA))
    lip_sync_corr = float(prov.get("lip_sync_corr", NEUTRAL_LIP_SYNC))
    landmark_jitter = float(prov.get("landmark_jitter", NEUTRAL_JITTER))

    return np.array([
        srm, fscore, kscore, flatness, res_norm, comp,
        exif_anomaly, c2pa_present, lip_sync_corr, landmark_jitter
    ], dtype=np.float32)
