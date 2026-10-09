"""Explicit, version-pinned LFCC front end (PyTorch).

WHY THIS FILE EXISTS
--------------------
The training pipeline used ``torchaudio.transforms.LFCC``. That is a moving
target: as of torchaudio 2.11 its constructor hardcodes::

    self.top_db = 80.0
    self.amplitude_to_DB = AmplitudeToDB("power", self.top_db)

i.e. the log-spectrogram is floored at ``max - 80 dB``. Measured on a 1 s
two-tone probe signal that clips **91.5 % of all (filter, frame) cells**
(log-LF spans [-100.0, +36.6], floor -43.4), which shifts the normalized
features by up to 10.3 (mean 0.076) versus an unclipped computation. The
Kotlin runtime and the golden vectors in this repo never had that clip, so
whether the app agreed with training silently depended on which torchaudio
release happened to be installed when the checkpoint was made.

Nothing recorded that. This module removes the dependency: the front end is
now spelled out here, in full, with every parameter a named constant, and it
mirrors ``ml/src/data/lfcc_reference.py`` (NumPy) and
``LFCCExtractor.kt`` (Kotlin) exactly.

Decisions, stated deliberately rather than inherited by accident:

* ``top_db = None`` - no flooring. The 80 dB clip destroys the silence floor
  and therefore the energy cues that matter for vocoder/tamper detection. It is
  also utterance-dependent, which makes features non-stationary across clips.
  Changing this requires retraining; the value is part of the contract.
* CMVN is per-COEFFICIENT over TIME. ``T.LFCC`` emits
  ``(channels, n_lfcc, time)``, so ``dim=-1`` is time BEFORE a transpose and
  the coefficient axis AFTER one. Getting this backwards was a real bug here.
* ``std`` uses ``ddof=1`` to match ``torch.std``'s default ``correction=1``.
* Linear (Hz-spaced) filters, un-normalized triangles, matching librosa /
  torchaudio's ``linear_fbanks``.
* reflect padding with ``center=True``; ``n_frames = len//hop + 1``.
* The STFT runs in **float64**. The Kotlin runtime uses a float64 transform
  (``FFTUtil.transform`` over ``DoubleArray``), and the log-LF floor is
  ``1e-10`` (-100 dB). A float32 transform's round-off is small in absolute
  terms but a large *fraction* of values down at that floor, and
  ``10*log10`` multiplies the relative error by 4.34. Measured on
  ``am_speech_like_probe()``, a float32 STFT disagreed with the Kotlin
  runtime by 1.1e-01 with 63 % of cells outside the 2e-3 JVM tolerance;
  float64 brings it to ~7e-06. ``scripts/verify_front_end_parity.py`` gates
  both directions. The float32 **output** tensor is unchanged: the deployed
  TFLite model declares a float32 input, and the ``.float()`` cast below is
  that boundary.

``FRONTEND_ID`` is recorded with every checkpoint so a stale model can be
detected instead of silently mis-scored.
"""

from __future__ import annotations

import math

import torch

# --- the contract ---------------------------------------------------------
SAMPLE_RATE = 16000
N_FILTER = 128
N_LFCC = 40
N_FFT = 512
WIN_LENGTH = 512      # deliberately == N_FFT; not 320
HOP_LENGTH = 160
CENTER = True
EPSILON = 1e-6
TOP_DB = None         # see module docstring
NORM = "ortho"

FRONTEND_ID = (
    f"lfcc-v3-sr{SAMPLE_RATE}-f{N_FILTER}-c{N_LFCC}"
    f"-nfft{N_FFT}-hop{HOP_LENGTH}-topdb{'none' if TOP_DB is None else TOP_DB}"
    "-resampler-quadratic_window_sinc_v1-float64stft-float32"
)


def linear_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_filter: int = N_FILTER,
    n_freqs: int | None = None,
) -> torch.Tensor:
    """Un-normalized linear (Hz-spaced) triangles -> (n_filter, n_freqs).

    Bin edges and filter mid-points follow torchaudio's ``linear_fbanks``:
    ``linspace(0, sample_rate//2, n_freqs)`` and
    ``linspace(f_min, f_max, n_filter + 2)``.
    """
    n_freqs = n_freqs or N_FFT // 2 + 1
    f_max = float(sample_rate // 2)
    # dtype must be explicit: torch.linspace defaults to the global default
    # dtype (float32), which would round the filter edges before the wider
    # module-level self.to(torch.float64) ever sees them.
    all_freqs = torch.linspace(0.0, f_max, n_freqs, dtype=torch.float64)
    f_pts = torch.linspace(0.0, f_max, n_filter + 2, dtype=torch.float64)
    f_diff = torch.diff(f_pts)

    down = (all_freqs[:, None] - f_pts[None, :-2]) / f_diff[:-1]
    up = (f_pts[None, 2:] - all_freqs[:, None]) / f_diff[1:]
    return torch.clamp(torch.minimum(down, up), min=0.0).T.contiguous()


def dct_matrix(n_lfcc: int = N_LFCC, n_filter: int = N_FILTER) -> torch.Tensor:
    """DCT-II ortho -> (n_filter, n_lfcc).

    Orientation matches ``torchaudio.functional.create_dct`` so it can be
    applied as ``specgram.transpose(-1, -2) @ dct`` exactly like upstream.
    """
    k = torch.arange(n_lfcc, dtype=torch.float64).unsqueeze(1)   # (n_lfcc, 1)
    n = torch.arange(n_filter, dtype=torch.float64).unsqueeze(0)  # (1, n_filter)
    m = torch.cos(math.pi / n_filter * (n + 0.5) * k) * math.sqrt(2.0 / n_filter)
    m[0] /= math.sqrt(2.0)          # k == 0 half-scale
    return m.T.contiguous()         # (n_lfcc, n_filter) -> (n_filter, n_lfcc)


class LFCCFrontend(torch.nn.Module):
    """Stateless, buffer-only LFCC front end. Batch, time -> (B, T, n_lfcc)."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_filter: int = N_FILTER,
        n_lfcc: int = N_LFCC,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        center: bool = CENTER,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_filter = n_filter
        self.n_lfcc = n_lfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.register_buffer("window", torch.hann_window(n_fft, dtype=torch.float64), persistent=False)
        self.register_buffer(
            "filterbank",
            linear_filterbank(sample_rate, n_filter, n_fft // 2 + 1),
            persistent=False,
        )
        self.register_buffer("dct", dct_matrix(n_lfcc, n_filter), persistent=False)

        # Filterbank, logarithm, DCT and normalization use float64 for DSP
        # stability, and the STFT above follows so the whole chain matches
        # the Kotlin runtime. The buffers are widened here; the filterbank and
        # dct matrices were built by torch with its default dtype, so build
        # them in float64 explicitly rather than relying on this cast.
        self.to(torch.float64)

    def n_frames(self, n_samples: int) -> int:
        if self.center:
            return n_samples // self.hop_length + 1
        return (n_samples - self.n_fft) // self.hop_length + 1

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """@param waveform (B, T) float -> (B, frames, n_lfcc), CMVN applied."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[-1] <= self.n_fft // 2:
            # torch.stft(center=True, pad_mode='reflect') requires the signal to
            # be longer than the half-window, otherwise it indexes out of range.
            # Callers must implement the declared source-window padding policy.
            raise ValueError(
                f"signal too short for LFCC: need > {self.n_fft // 2} samples, "
                f"got {waveform.shape[-1]}. Decode/pad according to the model contract."
            )

        # float64 STFT: matches the Kotlin runtime's FFTUtil.transform and the
        # NumPy oracle. A float32 transform disagrees by >1e-1 once the linear
        # filters reach the 1e-10 log floor; see the module docstring. The
        # result is cast back to float32 at the end of this method - that cast
        # is the TFLite input boundary and must not be moved or removed.
        spec = torch.stft(
            waveform.to(torch.float64),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(waveform.device, torch.float64),
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )                                                        # (B, F, T)
        power = spec.real.pow(2) + spec.imag.pow(2)                # (B, F, T)
        power = power.to(self.filterbank.dtype)

        # (B, n_filter, T); filterbank is (n_filter, n_freqs)
        filtered = torch.einsum("kf,bft->bkt", self.filterbank, power)
        filtered = filtered.to(self.dct.dtype)

        # 10*log10(max(x, 1e-10)); no top_db flooring (see module docstring).
        log_lf = 10.0 * torch.log10(torch.clamp(filtered, min=1e-10))

        # DCT-II ortho contracts n_filter -> n_lfcc:
        #   (B, T, n_filter) @ (n_filter, n_lfcc) -> (B, T, n_lfcc)
        lfcc_t = torch.matmul(log_lf.transpose(-1, -2), self.dct)

        # Normalize on (B, n_lfcc, T) so dim=-1 is TIME, then put T back at -2.
        # Normalizing the (B, T, n_lfcc) tensor instead would make dim=-1 the
        # COEFFICIENT axis and silently reintroduce per-frame z-scoring â€” the
        # original bug this module exists to prevent.
        normalized = normalize_cmn(lfcc_t.transpose(-1, -2))     # (B, n_lfcc, T)
        # float32 OUTPUT: the deployed TFLite model declares a float32 input
        # tensor, so the cast here is the DSP/model boundary, not a shortcut.
        return normalized.transpose(-1, -2).contiguous().float()  # (B, T, n_lfcc)


def normalize_cmn(lfcc: torch.Tensor, epsilon: float = EPSILON) -> torch.Tensor:
    """Utterance CMVN: per-COEFFICIENT over TIME.

    ``lfcc`` must be (..., n_lfcc, T). Statistics are taken over the LAST axis
    (time) so every cepstral coefficient is centred and scaled across the
    utterance. Callers MUST transpose to this layout first; ``forward`` does.

    ``std`` uses ``correction=1`` to match ``torch.std``'s default, which is
    what the Kotlin runtime and the NumPy reference assume (n-1, not n).
    """
    mean = lfcc.mean(dim=-1, keepdim=True)
    std = lfcc.std(dim=-1, keepdim=True, correction=1)
    return (lfcc - mean) / (std + epsilon)


def extract(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    n_filter: int = N_FILTER,
    n_lfcc: int = N_LFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    target_samples: int | None = None,
) -> torch.Tensor:
    """Functional convenience wrapper -> (1, frames, n_lfcc)."""
    if target_samples is not None:
        waveform = center_crop_or_pad(waveform, target_samples)
    fe = LFCCFrontend(sample_rate, n_filter, n_lfcc, n_fft, hop_length)
    return fe(waveform)


def center_crop_or_pad(x: torch.Tensor, target: int) -> torch.Tensor:
    """Centre-crop or right-zero-pad the TIME axis (last dim)."""
    if x.shape[-1] == target:
        return x
    if x.shape[-1] > target:
        start = (x.shape[-1] - target) // 2
        return x[..., start:start + target]
    return torch.nn.functional.pad(x, (0, target - x.shape[-1]))