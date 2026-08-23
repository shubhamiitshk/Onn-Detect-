import torch
import torchaudio
import torchaudio.transforms as T

class LFCCExtractor:
    """
    Linear Frequency Cepstral Coefficients (LFCC) Extractor.
    Designed for audio deepfake detection, as high-frequency linearity preserves
    synthetic artifacts better than Mel-scaled filters.
    """
    def __init__(self, sample_rate: int = 16000, n_filter: int = 128, n_lfcc: int = 40, n_fft: int = 512, hop_length: int = 160):
        """
        Initializes the LFCC transform.
        Args:
            sample_rate (int): Audio sample rate (default 16kHz).
            n_filter (int): Number of linear filters to apply.
            n_lfcc (int): Number of cepstral coefficients to retain.
            n_fft (int): Size of FFT.
            hop_length (int): Length of hop between STFT windows.
        """
        self.sample_rate = sample_rate
        
        self.lfcc_transform = T.LFCC(
            sample_rate=sample_rate,
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            speckwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'center': True
            }
        )

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extracts and normalizes LFCC features from a given waveform.
        
        Args:
            waveform (torch.Tensor): Audio waveform of shape (..., time).
        
        Returns:
            torch.Tensor: Normalized LFCC features of shape (1, 1, time_frames, n_lfcc).
        """
        # Ensure waveform is 2D: (channels, time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # Compute LFCC: output shape (channels, n_lfcc, time_frames)
        lfcc = self.lfcc_transform(waveform)
        
        # Normalize: zero mean, unit variance per utterance
        mean = lfcc.mean(dim=-1, keepdim=True)
        std = lfcc.std(dim=-1, keepdim=True)
        # Add epsilon to prevent division by zero
        lfcc_normalized = (lfcc - mean) / (std + 1e-6)
        
        # Rearrange dimensions to (batch, channels, time_frames, n_lfcc_coeffs)
        # Expected output shape: [1, 1, time_frames, n_lfcc]
        lfcc_transposed = lfcc_normalized.transpose(1, 2)
        
        # If input didn't have a batch dimension natively, ensure the final shape is exactly [1, 1, T, F]
        if lfcc_transposed.dim() == 3 and lfcc_transposed.size(0) == 1:
            lfcc_final = lfcc_transposed.unsqueeze(0)
        else:
            # Assumes input was (batch, time) or (batch, channels, time)
            # Ensure it fits the [B, 1, T, F] structure
            if lfcc_transposed.dim() == 3:
                lfcc_final = lfcc_transposed.unsqueeze(1)
            else:
                lfcc_final = lfcc_transposed
                
        return lfcc_final
