"""LFCC feature extraction for audio deepfake detection.

Linear Frequency Cepstral Coefficients (LFCC) keep the filterbank linear in
absolute frequency rather than perceptual (Mel) spacing, which preserves the
high-frequency linearity that neural vocoders and diffusion TTS disturb.

The DSP and the normalization now live in ``lfcc_frontend`` (PyTorch, explicit
and version-pinned) and ``lfcc_reference`` (NumPy, the parity oracle). This
module keeps only the API the datasets and export scripts already call.

Do NOT reintroduce ``torchaudio.transforms.LFCC`` here. As of torchaudio 2.11
it hardcodes ``top_db=80.0``, which floors 91.5 % of spectrogram cells and
silently changed the training features relative to the Kotlin runtime. See the
``lfcc_frontend`` module docstring for the measurements.
"""

import torch

from .lfcc_frontend import (
    CENTER,
    EPSILON,
    FRONTEND_ID,
    HOP_LENGTH,
    N_FFT,
    N_FILTER,
    N_LFCC,
    SAMPLE_RATE,
    TOP_DB,
    LFCCFrontend,
    center_crop_or_pad,
    normalize_cmn,
)


class LFCCExtractor:
    """LFCC extractor with the ONN-DETECT front-end contract.

    Args:
        sample_rate: analysis rate (the waveform must already be resampled).
        n_filter: number of linear (Hz-spaced) triangular filters.
        n_lfcc: number of cepstral coefficients (DCT-II output length).
        n_fft: FFT size, also the window length.
        hop_length: STFT hop in samples (10 ms at 16 kHz).
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_filter: int = N_FILTER,
        n_lfcc: int = N_LFCC,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_filter = n_filter
        self.n_lfcc = n_lfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self._frontend = LFCCFrontend(
            sample_rate=sample_rate,
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            center=CENTER,
        )
        if (sample_rate, n_filter, n_lfcc, n_fft, hop_length) != (SAMPLE_RATE, N_FILTER, N_LFCC, N_FFT, HOP_LENGTH):
            raise ValueError("Non-default frontend requires a separately versioned mobile contract")
        self.frontend_id = FRONTEND_ID

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract LFCC for one utterance.

        Args:
            waveform: (channels, time) or (time,) float tensor.

        Returns:
            (1, 1, n_frames, n_lfcc) â€” the layout LCNNMFMAudio expects.
            Frames are CMVN-normalized per cepstral coefficient over time.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # (channels, time) -> (1, time) mono mean when multi-channel.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        features = self._frontend(waveform)      # (1, n_frames, n_lfcc)
        return features.unsqueeze(1).float()             # (1, 1, n_frames, n_lfcc)

    def n_frames(self, n_samples: int) -> int:
        """Frame count for a given sample count (matches extract())."""
        return self._frontend.n_frames(n_samples)


__all__ = [
    "LFCCExtractor",
    "LFCCFrontend",
    "normalize_cmn",
    "center_crop_or_pad",
    "FRONTEND_ID",
    "SAMPLE_RATE",
    "N_FILTER",
    "N_LFCC",
    "N_FFT",
    "HOP_LENGTH",
    "CENTER",
    "EPSILON",
    "TOP_DB",
]