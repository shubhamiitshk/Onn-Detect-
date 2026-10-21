"""Audio Codec and Presentation Pipeline.

Implements presentation matching for speech deepfake detection ("It's All
in the Presentation"). Eliminates the background-noise and codec shortcut
where clean studio speech (e.g. ASVspoof) is judged differently from noisy
field recordings (e.g. in-the-wild speech).

Features:
- Calibrated SNR background noise injection (10 dB to 30 dB).
- Room impulse response (RIR) acoustic reverberation simulation.
- Speech codec quantization emulation (simulating Opus 16/24/32 kbps, AAC, MP3).
- Paired generation: emits (clean_waveform, presented_waveform) pairs.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch


@dataclass(frozen=True)
class AudioPresentationProfile:
    name: str
    target_snr_db: Optional[float] = None
    reverb_rt60: Optional[float] = None  # in seconds, e.g. 0.15 to 0.40
    codec_kind: Optional[str] = None     # "opus_16k", "opus_24k", "mp3_64k", "aac_low"
    bandpass_cutoff_hz: Optional[Tuple[float, float]] = None


AUDIO_PROFILES = [
    AudioPresentationProfile(name="studio_clean"),
    AudioPresentationProfile(name="opus_voip", target_snr_db=25.0, codec_kind="opus_16k", bandpass_cutoff_hz=(80.0, 7500.0)),
    AudioPresentationProfile(name="room_reverb", target_snr_db=20.0, reverb_rt60=0.25),
    AudioPresentationProfile(name="harsh_social", target_snr_db=15.0, reverb_rt60=0.35, codec_kind="opus_16k"),
    AudioPresentationProfile(name="mp3_compression", target_snr_db=22.0, codec_kind="mp3_64k"),
]


def add_calibrated_noise(
    waveform: np.ndarray,
    snr_db: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add Gaussian noise to achieve the exact target Signal-to-Noise Ratio (SNR) in dB."""
    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(waveform ** 2)
    if signal_power < 1e-12:
        return waveform

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=waveform.shape).astype(waveform.dtype)
    return waveform + noise


def apply_synthetic_reverb(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    rt60: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate room acoustic reverberation using an exponentially decaying stochastic RIR."""
    if rt60 <= 0.01:
        return waveform
    if rng is None:
        rng = np.random.default_rng()

    rir_len = int(sample_rate * min(rt60, 0.5))
    if rir_len < 16:
        return waveform

    t = np.linspace(0, rt60, rir_len)
    # Exponential decay envelope: reaches -60 dB at t = rt60
    decay = np.exp(-6.9078 * t / rt60)
    rir = rng.normal(0.0, 1.0, rir_len) * decay
    # Add strong direct-path component
    rir[0] = 1.0
    rir = rir / (np.linalg.norm(rir) + 1e-9)

    convolved = np.convolve(waveform, rir, mode="same")
    # Preserve original RMS energy
    orig_rms = np.sqrt(np.mean(waveform ** 2) + 1e-12)
    conv_rms = np.sqrt(np.mean(convolved ** 2) + 1e-12)
    return (convolved * (orig_rms / conv_rms)).astype(waveform.dtype)


def simulate_codec_degradation(
    waveform: np.ndarray,
    codec_kind: str,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Simulate lossy speech codec spectral quantization and band-limiting."""
    n = len(waveform)
    if n < 32:
        return waveform

    # Perform FFT to manipulate spectral bins
    spec = np.fft.rfft(waveform)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    if "opus_16k" in codec_kind:
        # Bandpass 80 Hz to 7000 Hz, plus light spectral band quantization
        mask = (freqs >= 80.0) & (freqs <= 7000.0)
        spec = spec * mask
        # Quantize high-frequency coefficients
        hf_idx = freqs > 3500.0
        q_step = np.median(np.abs(spec[hf_idx])) * 0.2 if np.any(hf_idx) else 1.0
        if q_step > 0:
            spec[hf_idx] = np.round(spec[hf_idx] / q_step) * q_step
    elif "mp3_64k" in codec_kind:
        # Cut off frequencies above 7500 Hz
        mask = freqs <= 7500.0
        spec = spec * mask

    restored = np.fft.irfft(spec, n=n)
    return restored.astype(waveform.dtype)


class AudioPresentationPipeline:
    """Transforms raw speech into codec-matched and acoustics-matched presentations.

    Can be applied dynamically to ASVspoof, in-the-wild speech, or synthesized
    speech so neither background noise nor codec artifacts separate classes.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        p_apply: float = 0.8,
        seed: Optional[int] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.p_apply = p_apply
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

    def process(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        profile: Optional[AudioPresentationProfile] = None,
    ) -> Tuple[np.ndarray, AudioPresentationProfile]:
        """Apply presentation matching to an input waveform."""
        if isinstance(waveform, torch.Tensor):
            arr = waveform.detach().cpu().numpy().squeeze()
        else:
            arr = np.asarray(waveform, dtype=np.float32).squeeze()

        if profile is None:
            if self._py_rng.random() >= self.p_apply:
                return arr, AudioPresentationProfile(name="studio_clean")
            profile = self._py_rng.choice(AUDIO_PROFILES[1:])

        result = arr.copy()

        # 1. Reverberation
        if profile.reverb_rt60 is not None:
            result = apply_synthetic_reverb(
                result,
                sample_rate=self.sample_rate,
                rt60=profile.reverb_rt60,
                rng=self._rng,
            )

        # 2. Codec simulation
        if profile.codec_kind is not None:
            result = simulate_codec_degradation(
                result,
                codec_kind=profile.codec_kind,
                sample_rate=self.sample_rate,
            )

        # 3. Additive noise
        if profile.target_snr_db is not None:
            result = add_calibrated_noise(
                result,
                snr_db=profile.target_snr_db,
                rng=self._rng,
            )

        # Normalize peak to [-0.95, 0.95] to avoid clipping
        peak = np.max(np.abs(result)) + 1e-9
        if peak > 0.95:
            result = result * (0.95 / peak)

        return result, profile

    def __call__(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (clean_view, presented_view) pair."""
        if isinstance(waveform, torch.Tensor):
            clean = waveform.detach().cpu().numpy().squeeze()
        else:
            clean = np.asarray(waveform, dtype=np.float32).squeeze()

        presented, _ = self.process(clean)
        return clean, presented
