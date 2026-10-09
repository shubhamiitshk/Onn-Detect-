"""Verify cross-platform FFT parity.

1. Implements the EXACT Kotlin FFTUtil algorithm (bit-reversal, butterfly
   passes, row-column passes, fftshift) in Python.
2. Checks it against numpy's reference FFT.
3. Runs the full Android analyzer chain (resize->gray->spectrum->azimuthal->
   OLS slope->score/flatness) against ml/src/meta_classifier/
   feature_extractor.py on real dataset images and reports max deltas.
"""
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "ml", "src"))
sys.path.append(os.path.join(ROOT, "ml"))

# ---------------- exact port of FFTUtil.kt ----------------

def kt_fft_1d(re, im):
    n = len(re)
    assert n & (n - 1) == 0 and n > 0
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]
    ln = 2
    while ln <= n:
        ang = -2.0 * np.pi / ln
        w_step_re, w_step_im = np.cos(ang), np.sin(ang)
        start = 0
        half = ln >> 1
        while start < n:
            w_re, w_im = 1.0, 0.0
            for k in range(half):
                ie, io = start + k, start + k + half
                u_re, u_im = re[ie], im[ie]
                v_re = re[io] * w_re - im[io] * w_im
                v_im = re[io] * w_im + im[io] * w_re
                re[ie] = u_re + v_re
                im[ie] = u_im + v_im
                re[io] = u_re - v_re
                im[io] = u_im - v_im
                nre = w_re * w_step_re - w_im * w_step_im
                w_im = w_re * w_step_im + w_im * w_step_re
                w_re = nre
            start += ln
        ln <<= 1


def kt_power_spectrum(pixels, n):
    re = list(map(float, pixels))
    im = [0.0] * (n * n)
    # row transforms (Kotlin copies out per row via the offset overload)
    tmp_re, tmp_im = [0.0] * n, [0.0] * n
    for y in range(n):
        off = y * n
        for i in range(n):
            tmp_re[i] = re[off + i]
            tmp_im[i] = im[off + i]
        kt_fft_1d(tmp_re, tmp_im)
        for i in range(n):
            re[off + i] = tmp_re[i]
            im[off + i] = tmp_im[i]
    col_re, col_im = [0.0] * n, [0.0] * n
    for x in range(n):
        for y in range(n):
            col_re[y] = re[y * n + x]
            col_im[y] = im[y * n + x]
        kt_fft_1d(col_re, col_im)
        for y in range(n):
            re[y * n + x] = col_re[y]
            im[y * n + x] = col_im[y]
    spec = [0.0] * (n * n)
    half = n // 2
    for y in range(n):
        sy = (y + half) % n
        for x in range(n):
            sx = (x + half) % n
            spec[sy * n + sx] = re[y * n + x] ** 2 + im[y * n + x] ** 2
    return np.asarray(spec)


# ---- step 1+2: algorithm vs numpy reference on random input ----
rng = np.random.default_rng(0)
x = rng.random(64 * 64)
kt = kt_power_spectrum(x, 64)
ref = np.abs(np.fft.fftshift(np.fft.fft2(x.reshape(64, 64)))) ** 2
err = np.abs(np.sort(kt) - np.sort(ref.ravel())).max()  # order-insensitive
direct = np.abs(kt - ref.ravel()).max()
print(f"FFT algorithm vs numpy: direct max diff {direct:.2e} "
      f"(order-insensitive {err:.2e})")
if not np.allclose(kt, ref.ravel(), rtol=1e-10, atol=1e-7):
    raise SystemExit("FFT algorithm parity failed")

# ---- step 3: full analyzer chain parity on REAL images ----
from meta_classifier.feature_extractor import fft_features, to_gray_u8, azimuthal_average, power_law_slope, wiener_entropy

def android_chain(img_path, size=256):
    img = Image.open(img_path).convert("RGB")
    scaled = img.resize((size, size), Image.BILINEAR)
    gray01 = np.asarray(scaled, dtype=np.float64)
    g = (gray01[..., 0] * 299 + gray01[..., 1] * 587 + gray01[..., 2] * 114) / 1000.0
    g = np.clip(g, 0, 255).astype(np.uint8).astype(np.float64) / 255.0
    ps = kt_power_spectrum(g.ravel(), size).reshape(size, size)
    radial = azimuthal_average(ps)
    slope = power_law_slope(radial)
    score = min(abs(slope + 2.0) / 2.0, 1.0)
    flatness = wiener_entropy(ps)
    return score, flatness

val_dir = os.path.join(ROOT, "datasets", "genimage", "val", "real")
if os.path.exists(val_dir) and os.listdir(val_dir):
    files = [os.path.join(val_dir, f) for f in sorted(os.listdir(val_dir))[:5]]
else:
    print("\nNo local dataset found in datasets/genimage/val/real; testing with synthetic images.")
    # Create temporary in-memory PIL images for verification
    files = []
    np.random.seed(42)
    for i in range(5):
        arr = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        files.append(img)

max_ds, max_df = 0.0, 0.0
for i, item in enumerate(files):
    if isinstance(item, str):
        py_score, py_flat = fft_features(Image.open(item))
        kt_score, kt_flat = android_chain(item)
        name = os.path.basename(item)
    else:
        py_score, py_flat = fft_features(item)
        # In-memory android chain
        scaled = item.resize((256, 256), Image.BILINEAR)
        gray01 = np.asarray(scaled, dtype=np.float64)
        g = (gray01[..., 0] * 299 + gray01[..., 1] * 587 + gray01[..., 2] * 114) / 1000.0
        g = np.clip(g, 0, 255).astype(np.uint8).astype(np.float64) / 255.0
        ps = kt_power_spectrum(g.ravel(), 256).reshape(256, 256)
        radial = azimuthal_average(ps)
        slope = power_law_slope(radial)
        kt_score = min(abs(slope + 2.0) / 2.0, 1.0)
        kt_flat = wiener_entropy(ps)
        py_score, py_flat = fft_features(item)
        name = f"synthetic_img_{i+1}.png"

    max_ds = max(max_ds, abs(py_score - kt_score))
    max_df = max(max_df, abs(py_flat - kt_flat))
    print(f"{name}: python(score={py_score:.6f}, flat={py_flat:.6e}) "
          f"android(score={kt_score:.6f}, flat={kt_flat:.6e})")
print(f"\nMax delta  score: {max_ds:.2e} | flatness: {max_df:.2e}")
if not np.isfinite([max_ds, max_df]).all() or max_ds > 1e-6 or max_df > 1e-7:
    raise SystemExit("FFT feature parity failed")
print("Measured Python port parity passed. Kotlin JVM tests remain a separate gate.")
