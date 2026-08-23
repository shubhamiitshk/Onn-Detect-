"""Real meta-classifier feature extraction.

Every handcrafted feature computed here mirrors, step for step, its on-device
counterpart in the Android app (see android/app/src/main/java/com/onndetect/
detection/visual/). This guarantees that the distribution the XGBoost
meta-classifier is trained on matches what it will see at inference time.

Canonical feature vector (order is a cross-platform contract):

    0: srm_score        - SRMForensicCNN probability of AI-generated
    1: fft_score        - spectral slope deviation from 1/f^2 power law
    2: kl_score         - KL divergence of adjacent-pixel-diff histogram vs baseline
    3: flatness_score   - Wiener spectral entropy of the power spectrum
    4: resolution_norm  - min(w,h)/1024 clamped to [0,1]
    5: is_compressed    - 1.0 if JPEG-compressed source, else 0.0

Audio/video/temporal signals are intentionally NOT part of this vector:
the image pipeline fuses only signals it can actually compute; the audio path
thresholds its LCNN score directly with an EER-calibrated constant.
"""

import json
import os

import numpy as np
from PIL import Image

# Bin edges identical to Android KLCooccurrenceAnalyzer: diffs in [-10, 10]
KL_BINS = 21
KL_CLAMP = 10
# Android maps score = kl / 0.3f coerced to [0,1]; keep identical here.
KL_SCORE_SCALE = 0.3

DEFAULT_BASELINE = [
    0.005, 0.007, 0.012, 0.018, 0.025, 0.038, 0.055, 0.082, 0.120, 0.150,
    0.180,
    0.150, 0.120, 0.082, 0.055, 0.038, 0.025, 0.018, 0.012, 0.007, 0.005,
]

FEATURE_NAMES = [
    "srm_score", "fft_score", "kl_score", "flatness_score",
    "resolution_norm", "is_compressed",
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
    """
    diffs = gray[:, :-1].astype(np.int32) - gray[:, 1:].astype(np.int32)
    clipped = np.clip(diffs, -KL_CLAMP, KL_CLAMP) + KL_CLAMP
    hist = np.bincount(clipped.ravel(), minlength=KL_BINS).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return np.full(KL_BINS, 1e-6)
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
            img = Image.open(p).convert("RGB")
            acc += diff_histogram(to_gray_u8(img))
            used += 1
        except Exception:
            continue
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
    """OLS log-log slope, skipping DC and r<2 — mirrors Android implementation."""
    freqs, powers = [], []
    for i in range(2, len(radial)):
        if radial[i] > 1e-10:
            freqs.append(np.log(i))
            powers.append(np.log(radial[i]))
    if len(freqs) < 2:
        return -2.0
    fx, fy = np.asarray(freqs), np.asarray(powers)
    denom = np.sum(fx**2) - len(fx) * fx.mean() ** 2
    if denom == 0:
        return -2.0
    return float((np.sum(fx * fy) - len(fx) * fx.mean() * fy.mean()) / denom)


def wiener_entropy(ps: np.ndarray) -> float:
    """Geometric mean / arithmetic mean of the power spectrum."""
    vals = ps[ps > 1e-10]
    if vals.size == 0 or vals.sum() == 0:
        return 0.0
    return float(np.exp(np.log(vals).mean()) / vals.mean())


def fft_features(img: Image.Image, size: int = 256) -> tuple[float, float]:
    """(fft_score, flatness_score) mirroring Android FFTSpectralAnalyzer.

    Bitmap.createScaledBitmap uses bilinear filtering; Pillow BILINEAR matches.
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


def quality_features(img: Image.Image, path: str) -> tuple[float, float]:
    """(resolution_norm, is_compressed) from real file properties."""
    w, h = img.size
    resolution_norm = min(min(w, h) / 1024.0, 1.0)
    compressed = 1.0 if path.lower().endswith((".jpg", ".jpeg")) else 0.0
    return resolution_norm, compressed


def extract_visual_features(
    img_path: str,
    srm_model,
    device,
    baseline: np.ndarray,
) -> np.ndarray:
    """Full canonical feature vector for one image file."""
    from torchvision import transforms
    import torch

    img = Image.open(img_path).convert("RGB")

    # SRM-CNN at native resolution (no resize): same as training/eval pipeline
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        srm = float(srm_model(tensor).item())

    fscore, flatness = fft_features(img)
    kscore, _ = kl_score(to_gray_u8(img), baseline)
    res_norm, comp = quality_features(img, img_path)

    return np.array([srm, fscore, kscore, flatness, res_norm, comp], dtype=np.float64)
