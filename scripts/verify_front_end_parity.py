"""Verify the audio front end across all three implementations.

    python scripts/verify_front_end_parity.py

Runs with NumPy alone (CI-friendly); torch sections are skipped gracefully
when torch is absent. With torch installed, the full cross-implementation
contract is checked.

Checks, in order:
  1. NumPy reference  ==  pinned PyTorch front end   (the DSP contract)
  2. CMVN invariants  -  per-COEFFICIENT over TIME     (the axis that was wrong)
  3. Pinned front end vs torchaudio T.LFCC + top_db=80 (quantified drift)
  4. Anti-alias resampler vs the old linear one (why it had to go)
  5. Robustness (short signals, silence)

Exit code is non-zero if any HARD check fails, so this gates CI.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ml", "src"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--require-torch", action="store_true", help="Fail instead of skipping the training front end (required in CI)")
args = parser.parse_args()

from data.lfcc_reference import (
    lfcc_raw, normalize_cmn, resample_linear, resample_sinc,
    two_tone_signal, am_speech_like_probe, HOP_LENGTH, SAMPLE_RATE,
)

try:
    import torch
    from data.lfcc_frontend import (
        FRONTEND_ID, N_FFT, N_FILTER, N_LFCC, LFCCFrontend,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    FRONTEND_ID = "lfcc-v3 (torch sections skipped - torch not installed)"
    if args.require_torch:
        raise SystemExit("PyTorch is required for this parity gate")

# The same tolerance LFCCExtractorTest.kt applies to the hand-transcribed
# goldens. One number governs both platforms; loosening it here hides a real
# Python/Kotlin disagreement.
FEATURE_TOLERANCE = 2e-3

HARD_FAILURES = []
SOFT = []
SKIPPED = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        HARD_FAILURES.append(name)


def soft(name, detail=""):
    print(f"  [INFO] {name}{('  ' + detail) if detail else ''}")
    SOFT.append(name)


def skip(name, reason):
    print(f"  [SKIP] {name} ({reason})")
    SKIPPED.append(name)


print("=" * 72)
print("ONN-DETECT front-end parity")
print(f"frontend_id = {FRONTEND_ID}")
print(f"torch available: {HAS_TORCH}")
print("=" * 72)

sig = two_tone_signal()
npfeat = normalize_cmn(lfcc_raw(sig)).T      # (T, n_lfcc), NumPy only

# ---- 1. NumPy reference == pinned PyTorch --------------------------------
print("\n1) NumPy reference vs pinned PyTorch front end (the contract)")
if HAS_TORCH:
    wav = torch.tensor(sig, dtype=torch.float64).unsqueeze(0)
    fe = LFCCFrontend()
    pt = fe(wav)[0].numpy()                 # (T, n_lfcc)
    d1 = float(np.abs(pt - npfeat).max())
    check("numpy == torch on stationary probe", d1 < FEATURE_TOLERANCE,
          f"max|diff| = {d1:.3e} (tolerance {FEATURE_TOLERANCE:.0e})")
    check("frame count", pt.shape == npfeat.shape,
          f"torch {pt.shape} numpy {npfeat.shape}")
    check("training features are float32", pt.dtype == np.float32, str(pt.dtype))
    check("training features are finite", bool(np.isfinite(pt).all()), "")

    # The stationary two-tone probe keeps the log-LF spectrum nearly constant,
    # so the per-coefficient CMVN std stays away from its floor. A float32 STFT
    # therefore looks acceptable on it and hides a real Kotlin disagreement.
    # This dynamic probe drives the high-frequency filters onto the 1e-10 log
    # floor, where float32 round-off dominates, and is the gate that catches it.
    dyn = am_speech_like_probe()
    npfeat_dyn = normalize_cmn(lfcc_raw(dyn)).T
    wav_dyn = torch.tensor(dyn, dtype=torch.float64).unsqueeze(0)
    pt_dyn = fe(wav_dyn)[0].numpy()
    d2 = float(np.abs(pt_dyn - npfeat_dyn).max())
    over = float((np.abs(pt_dyn - npfeat_dyn) > FEATURE_TOLERANCE).mean())
    check("numpy == torch on dynamic probe", d2 < FEATURE_TOLERANCE,
          f"max|diff| = {d2:.3e}, {over * 100:.1f}% of cells over tolerance")
    check("dynamic probe frame count", pt_dyn.shape == npfeat_dyn.shape,
          f"torch {pt_dyn.shape} numpy {npfeat_dyn.shape}")
    check("dynamic probe output is float32", pt_dyn.dtype == np.float32, str(pt_dyn.dtype))
    check("dynamic probe is finite", bool(np.isfinite(pt_dyn).all()), "")
else:
    skip("numpy == torch", "torch not installed")
    check("frame count == len//hop + 1",
          npfeat.shape[0] == len(sig) // HOP_LENGTH + 1,
          f"expected {len(sig) // HOP_LENGTH + 1}, got {npfeat.shape[0]}")

# ---- 2. CMVN invariants (the axis that was wrong) --------------------------
print("\n2) CMVN invariants - per-COEFFICIENT over TIME (NumPy only)")
per_coef_mean = np.abs(npfeat.mean(axis=0)).max()
per_coef_std = np.abs(npfeat.std(axis=0, ddof=1) - 1.0).max()
check("per-coefficient mean ~ 0 over time", per_coef_mean < 1e-4,
      f"max|mean| = {per_coef_mean:.3e}")
check("per-coefficient std ~ 1 over time", per_coef_std < 1e-4,
      f"max|std-1| = {per_coef_std:.3e}")
per_frame_mean = np.abs(npfeat.mean(axis=1)).max()
soft("per-frame mean is NOT zero (proves old bug is gone)",
     f"max|per-frame mean| = {per_frame_mean:.3e} (old code forced this to 0)")

c0_spread = float(npfeat[:, 0].std())
soft("c0 (log energy) spread", f"std = {c0_spread:.4f}")

# ---- 3. Drift vs torchaudio as shipped ------------------------------------
print("\n3) Pinned front end vs torchaudio T.LFCC (top_db=80)")
try:
    import torchaudio
    import torchaudio.transforms as T

    ta = T.LFCC(
        sample_rate=SAMPLE_RATE, n_filter=N_FILTER, n_lfcc=N_LFCC,
        speckwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH, "center": True},
    )(torch.tensor(sig, dtype=torch.float32).unsqueeze(0))[0].numpy()

    mu = ta.mean(axis=-1, keepdims=True)
    # ddof=1 to match torch.std's default correction=1, which is what the
    # pinned front end and the Kotlin runtime assume.
    sd = ta.std(axis=-1, ddof=1, keepdims=True)
    ta_norm = ((ta - mu) / (sd + 1e-6)).T
    d3 = float(np.abs(npfeat - ta_norm).max())
    m3 = float(np.abs(npfeat - ta_norm).mean())
    soft(f"torchaudio {torchaudio.__version__} max|diff| = {d3:.3f}",
         f"mean|diff| = {m3:.3f}")
    if d3 > 0.5:
        soft("ACTION: torchaudio T.LFCC features are NOT interchangeable with "
             "the pinned front end. Never train with T.LFCC; a model trained "
             "that way will not match the Kotlin runtime.")
except ImportError:
    skip("torchaudio drift measurement", "torchaudio not installed")

# ---- 4. Resampler (NumPy only) --------------------------------------------
print("\n4) Resampler - anti-aliasing (linear interpolation is NOT acceptable)")
sr = 48000
tsig = np.arange(sr) / sr
for f in (10000, 15000, 20000):
    # Above the 16 kHz Nyquist. A correct resampler must suppress these;
    # linear interpolation folds them straight into the analysis band.
    x = np.sin(2 * np.pi * f * tsig)
    lin_rms = float(np.sqrt((resample_linear(x, sr, SAMPLE_RATE) ** 2).mean()))
    sinc_rms = float(np.sqrt((resample_sinc(x, sr, SAMPLE_RATE) ** 2).mean()))
    check(f"{f} Hz out-of-band suppressed by sinc",
          sinc_rms < 0.1 * lin_rms,
          f"linear rms {lin_rms:.5f} vs sinc rms {sinc_rms:.5f} "
          f"({lin_rms / max(sinc_rms, 1e-12):.0f}x rejection)")

x = np.sin(2 * np.pi * 1000 * tsig)
sinc_rms = float(np.sqrt((resample_sinc(x, sr, SAMPLE_RATE) ** 2).mean()))
check("in-band 1 kHz tone keeps its amplitude",
      0.8 < sinc_rms / 0.7071 < 1.2,
      f"rms {sinc_rms:.5f} vs ideal 0.70711 (must not be over-filtered)")

# Non-integer ratios and upsampling: integer-decimate implementations get
# these wrong in BOTH length and rate (e.g. 44100->16000 needs factor 2.756,
# not 3). The fractional-position kernel must be exact here.
print("\n4b) Non-integer ratios and upsampling")
for src_rate, dst_rate, freq in ((44100, SAMPLE_RATE, 1000),
                                 (22050, SAMPLE_RATE, 3000),
                                 (8000, SAMPLE_RATE, 1000)):
    n_in = src_rate  # 1 s
    t_in = np.arange(n_in) / src_rate
    tone_in = np.sin(2 * np.pi * freq * t_in)
    y = resample_sinc(tone_in, src_rate, dst_rate)
    want_len = int(np.ceil(n_in * dst_rate / src_rate))
    check(f"{src_rate}->{dst_rate} length == ceil(n*ratio)",
          len(y) == want_len, f"got {len(y)}, want {want_len}")
    rms = float(np.sqrt((y ** 2).mean()))
    check(f"{src_rate}->{dst_rate} {freq} Hz tone preserved",
          0.55 < rms / 0.7071 < 1.25,
          f"rms {rms:.5f} vs ideal 0.70711")

soft("Resampler identity", "quadratic_window_sinc_v1; different window from torchaudio. "
     "The Kotlin JVM golden tests separately check deployed arithmetic. "
     "Correlation with a different filter is not a parity certificate.")

# ---- 5. Robustness ---------------------------------------------------------
print("\n5) Robustness")
if HAS_TORCH:
    for n in (16000, 64000, 1024, 600):
        s = two_tone_signal(n)
        try:
            f = LFCCFrontend()
            t = f(torch.tensor(s, dtype=torch.float64).unsqueeze(0))[0].numpy()
            ok = np.all(np.isfinite(t))
            check(f"{n} samples -> finite output", ok,
                  f"shape {t.shape}, max|v| = {np.abs(t).max():.2f}")
        except Exception as e:
            check(f"{n} samples", False, f"{type(e).__name__}: {e}")

    try:
        z = LFCCFrontend()(torch.zeros(1, 64000, dtype=torch.float64))[0].numpy()
        check("digital silence stays finite", bool(np.all(np.isfinite(z))),
              f"max|v| = {np.abs(z).max():.4f}")
    except Exception as e:
        check("digital silence", False, f"{type(e).__name__}: {e}")
else:
    skip("torch robustness checks", "torch not installed")
    # NumPy-only silence check: CMVN of zeros must stay finite.
    z = normalize_cmn(np.zeros((40, 101)))
    check("digital silence stays finite (NumPy)",
          bool(np.all(np.isfinite(z))), f"max|v| = {np.abs(z).max():.4f}")

# ---- summary -------------------------------------------------------------
print()
print("=" * 72)
if SKIPPED:
    print(f"Skipped ({len(SKIPPED)}): " + ", ".join(SKIPPED))
if HARD_FAILURES:
    print(f"FAILED ({len(HARD_FAILURES)}): " + ", ".join(HARD_FAILURES))
    sys.exit(1)
print(f"All hard checks passed. {len(SOFT)} informational notes above.")
print("=" * 72)
