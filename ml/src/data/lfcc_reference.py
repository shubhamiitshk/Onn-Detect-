"""NumPy reference implementation of the ONN-DETECT audio front end.

This is the *contract*: it reproduces, in float64 and with only NumPy, the
exact LFCC tensor that the PyTorch training pipeline
(``ml/src/data/lfcc_extractor.py``) produces. It exists for three reasons:

1. Golden vectors for the Kotlin unit test can be regenerated without a
   PyTorch/torchaudio install (``scripts/gen_lfcc_golden.py``).
2. ``scripts/check_android_consistency.py`` can numerically compare the
   Python and Kotlin front ends element-wise instead of trusting that the
   Kotlin unit test still means anything.
3. It documents the ONE true normalization semantics in plain arithmetic,
   after this axis bug bit the repo three separate times.

Normalization semantics (the thing that was wrong)
--------------------------------------------------
Training (``lfcc_extractor.py``) does, on a ``(n_lfcc, T)`` tensor::

    mean = lfcc.mean(dim=-1, keepdim=True)     # dim=-1 is TIME
    std  = lfcc.std(dim=-1, keepdim=True)      # -> per-COEFFICIENT statistics
    out  = (lfcc - mean) / (std + 1e-6)        # CMVN over time

That is utterance-level cepstral mean/variance normalisation, the ASVspoof
convention. Per-frame z-scoring across coefficients (what the Kotlin code used
to do) is a *different* matrix, destroys coefficient 0 (log energy) and forces
every frame to zero mean so the network can no longer see silence vs speech.

Note the ordering trap: ``T.LFCC`` emits ``(channels, n_lfcc, time)``, so
``dim=-1`` is time *before* any transpose and *coefficients* after one. Every
earlier bug in this repo came from getting that backwards.
"""

from __future__ import annotations

import numpy as np

# Canonical front-end parameters. Must match LFCCExtractor.kt, the yaml
# configs, and the exported .tflite input signature.
SAMPLE_RATE = 16000
N_FILTER = 128
N_LFCC = 40
N_FFT = 512
HOP_LENGTH = 160
EPSILON = 1e-6


def _window(n_fft: int = N_FFT) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft))


def _linear_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_filter: int = N_FILTER,
    n_freqs: int | None = None,
) -> np.ndarray:
    """Un-normalized linear (Hz-spaced) triangles -> (n_filter, n_freqs)."""
    n_freqs = n_freqs or N_FFT // 2 + 1
    f_max = sample_rate / 2.0
    all_freqs = np.linspace(0.0, f_max, n_freqs)
    pts = np.linspace(0.0, f_max, n_filter + 2)
    fdiff = np.diff(pts)
    down = (all_freqs[:, None] - pts[None, :-2]) / fdiff[:-1]
    up = (pts[None, 2:] - all_freqs[:, None]) / fdiff[1:]
    return np.maximum(0.0, np.minimum(down, up)).T


def _dct_ortho(n_lfcc: int = N_LFCC, n_filter: int = N_FILTER) -> np.ndarray:
    """create_dct(n_lfcc, n_filter, norm='ortho') -> (n_lfcc, n_filter)."""
    k = np.arange(n_lfcc)[:, None]
    n = np.arange(n_filter)[None, :]
    m = np.cos(np.pi / n_filter * (n + 0.5) * k) * np.sqrt(2.0 / n_filter)
    m[0] /= np.sqrt(2.0)
    return m


def _reflect_pad_indices(base: int, n_fft: int, length: int) -> np.ndarray:
    """torch.stft(center=True, pad_mode='reflect') index mapping."""
    idx = base + np.arange(n_fft)
    neg = idx < 0
    idx[neg] = -idx[neg]
    over = idx >= length
    idx[over] = 2 * (length - 1) - idx[over]
    return idx


def lfcc_raw(
    pcm: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_filter: int = N_FILTER,
    n_lfcc: int = N_LFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """LFCC before normalization -> (n_lfcc, n_frames)."""
    pcm = np.asarray(pcm, dtype=np.float64)
    n_frames = len(pcm) // hop_length + 1
    n_freqs = n_fft // 2 + 1
    win = _window(n_fft)
    fb = _linear_filterbank(sample_rate, n_filter, n_freqs)
    dct = _dct_ortho(n_lfcc, n_filter)

    filtered = np.empty((n_filter, n_frames))
    for t in range(n_frames):
        base = t * hop_length - n_fft // 2
        frame = pcm[_reflect_pad_indices(base, n_fft, len(pcm))] * win
        spec = np.fft.rfft(frame)
        power = (spec.real ** 2 + spec.imag ** 2)[:n_freqs]
        filtered[:, t] = fb @ power

    # torchaudio AmplitudeToDB(stype='power', top_db=None)
    log_lf = 10.0 * np.log10(np.maximum(filtered, 1e-10))
    return dct @ log_lf


def normalize_cmn(lfcc: np.ndarray) -> np.ndarray:
    """Utterance CMVN: per-COEFFICIENT over TIME. This is the contract.

    ``lfcc`` must be (n_lfcc, n_frames); statistics are taken along axis=1.
    """
    mean = lfcc.mean(axis=1, keepdims=True)
    std = lfcc.std(axis=1, ddof=1, keepdims=True)  # torch.std -> correction=1
    return (lfcc - mean) / (std + EPSILON)


def lfcc_features(
    pcm: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_filter: int = N_FILTER,
    n_lfcc: int = N_LFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    target_samples: int | None = None,
) -> np.ndarray:
    """End-to-end front end -> (1, n_frames, n_lfcc), the LCNN input layout.

    ``target_samples`` centre-crops or zero-pads the waveform first, matching
    the 4 s / 64 000-sample window the Android decoder produces.
    """
    pcm = np.asarray(pcm, dtype=np.float64)
    if target_samples is not None:
        pcm = center_crop_or_pad(pcm, target_samples)
    return normalize_cmn(
        lfcc_raw(pcm, sample_rate, n_filter, n_lfcc, n_fft, hop_length)
    ).T[None, :]


def center_crop_or_pad(x: np.ndarray, target: int) -> np.ndarray:
    if len(x) == target:
        return x
    if len(x) > target:
        start = (len(x) - target) // 2
        return x[start:start + target]
    out = np.zeros(target)
    out[:len(x)] = x
    return out


def resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampler, mirroring PcmUtil.resampleLinear.

    Kept ONLY so the consistency checker can demonstrate the aliasing
    pathology that motivates replacing it. Do not use for training data.
    """
    if src_rate == dst_rate or len(x) == 0:
        return x
    out_len = len(x) * dst_rate // src_rate
    pos = np.arange(out_len) * (src_rate / dst_rate)
    i0 = np.floor(pos).astype(np.int64)
    i0 = np.clip(i0, 0, len(x) - 1)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    frac = pos - np.floor(pos)
    return x[i0] * (1.0 - frac) + x[i1] * frac


def resample_sinc(
    x: np.ndarray,
    src_rate: int,
    dst_rate: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
) -> np.ndarray:
    """quadratic_window_sinc_v1: deployed Kotlin per-sample sinc contract.

    This quadratic window is intentionally NOT torchaudio's Hann window.
    Reflect edges for long inputs, zero-pad short inputs, renormalize DC gain.
    """
    x = np.asarray(x, dtype=np.float64)
    if src_rate <= 0 or dst_rate <= 0 or x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Invalid resampler input/rate")
    if type(lowpass_filter_width) is not int or not 1 <= lowpass_filter_width <= 128:
        raise ValueError("Invalid low-pass filter width")
    if not np.isfinite(rolloff) or not 0 < rolloff <= 1:
        raise ValueError("Invalid low-pass rolloff")
    if not 1 <= src_rate <= 384000 or not 1 <= dst_rate <= 384000:
        raise ValueError("Unsupported sample rate")
    if src_rate == dst_rate or len(x) == 0:
        return x
    ratio = dst_rate / src_rate
    out_len = int(np.ceil(len(x) * ratio))
    cutoff = rolloff * min(1.0, ratio)
    half = int(np.ceil(lowpass_filter_width * max(1.0, 1.0 / ratio)))
    if half > 65536:
        raise ValueError("Resampling ratio would exceed bounded kernel memory")
    padded = np.pad(x, (half, half), mode="reflect" if len(x) > half else "constant")
    offsets = np.arange(-half, half + 1, dtype=np.int64)
    window = np.maximum(0.0, 1.0 - (offsets / (half + 1.0)) ** 2)
    out = np.empty(out_len, dtype=np.float64)
    # Bound temporary matrices independently of recording length/ratio.
    chunk_size = max(1, min(2048, 1_000_000 // len(offsets)))
    for begin in range(0, out_len, chunk_size):
        end = min(out_len, begin + chunk_size)
        positions = np.arange(begin, end, dtype=np.float64) / ratio
        base = np.floor(positions).astype(np.int64)
        distance = base[:, None] + offsets[None, :] - positions[:, None]
        weights = cutoff * np.sinc(cutoff * distance) * window[None, :]
        values = padded[base[:, None] + offsets[None, :] + half]
        sums = weights.sum(axis=1)
        out[begin:end] = np.divide((values * weights).sum(axis=1), sums, out=np.zeros(end - begin), where=sums != 0)
    return out



def two_tone_signal(
    n: int = 16000, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """The deterministic probe signal used by LFCCExtractorTest."""
    t = np.arange(n) / sample_rate
    return 0.5 * np.sin(2 * np.pi * 1000.0 * t) + 0.25 * np.sin(
        2 * np.pi * 2500.0 * t + 0.7
    )


def am_speech_like_probe(
    n: int = 64000, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Dynamic amplitude-modulated probe; the worst case for LFCC precision.

    ``two_tone_signal`` is two stationary sinusoids, which produces an almost
    constant log-LF spectrum. That keeps the per-coefficient CMVN standard
    deviation away from its floor and exercises the front end in its
    best-conditioned regime, so it cannot detect a precision regression.

    This probe adds a syllabic-rate (3.1 Hz) envelope over four harmonics
    spanning 220 Hz to 6.5 kHz, plus a slower tremolo on the top harmonic.
    The result is a spectrum whose high-frequency linear filters legitimately
    reach the 1e-10 log floor, which is exactly where float32 FFT round-off
    becomes a large *fraction* of the value and where the float64 Kotlin
    runtime and a float32 training front end diverge.

    Deterministic and closed-form: no RNG, so both platforms see identical
    samples and the fixture is reproducible without a stored blob.
    """
    t = np.arange(n) / sample_rate
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3.1 * t))
    voiced = (
        0.4 * np.sin(2 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2 * np.pi * 900.0 * t)
        + 0.1 * np.sin(2 * np.pi * 3200.0 * t)
    )
    fricative = 0.02 * np.sin(2 * np.pi * 6500.0 * t) * (
        1.0 + 0.5 * np.sin(2 * np.pi * 7.0 * t)
    )
    return envelope * voiced + fricative