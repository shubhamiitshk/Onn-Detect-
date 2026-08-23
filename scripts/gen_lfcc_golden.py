"""Generate golden LFCC vectors (torchaudio reference) for the Kotlin
LFCCExtractor unit test."""
import numpy as np
import torch
import torchaudio.transforms as T

SR = 16000
N = 16000  # 1 s -> T = N/hop + 1 = 101 frames

t = np.arange(N) / SR
sig = (
    0.5 * np.sin(2 * np.pi * 1000 * t)
    + 0.25 * np.sin(2 * np.pi * 2500 * t + 0.7)
   
)
wav = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)  # (1, N)

lfcc_t = T.LFCC(sample_rate=SR, n_filter=128, n_lfcc=40,
                speckwargs={"n_fft": 512, "hop_length": 160, "center": True})
lfcc = lfcc_t(wav).transpose(1, 2)[0]     # (101, 40)

mean = lfcc.mean(dim=-1, keepdim=True)
std = lfcc.std(dim=-1, keepdim=True)      # unbiased like torch default
norm = ((lfcc - mean) / (std + 1e-6))

print("shape:", tuple(norm.shape))
print("FRAME50 =", [round(float(v), 6) for v in norm[50, :8]])
print("FRAME0  =", [round(float(v), 6) for v in norm[0, :4]])
print("ENERGY  =", round(float(norm.pow(2).mean()), 6))
