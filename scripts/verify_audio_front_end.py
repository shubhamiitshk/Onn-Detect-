"""Does the LFCC implementation gap matter for VERDICTS?

Scores N real ASVspoof test files twice through the SAME trained LCNN:
  A) torchaudio LFCC front end (what training used)
  B) our numpy replica of the Kotlin extractor (float64 FFT)
Reports per-file score deltas and threshold agreement.
"""
import os
import sys
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "ml", "src"))

from models.lcnn_mfm_audio import LCNNMFMAudio

SR, NFILT, NLFCC, NFFT, HOP = 16000, 128, 40, 512, 160
TARGET = SR * 4

lfcc_t = T.LFCC(sample_rate=SR, n_filter=NFILT, n_lfcc=NLFCC,
                speckwargs={"n_fft": NFFT, "hop_length": HOP, "center": True})
model = LCNNMFMAudio.from_pretrained(os.path.join(ROOT, "ml", "checkpoints",
                                                  "lcnn_audio_asvspoof_best.pth")).eval()

win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(NFFT) / NFFT))
afreqs = np.linspace(0, SR / 2, 257, dtype=np.float32)
fpts = np.linspace(0, SR / 2, NFILT + 2, dtype=np.float32)
fdiff = np.diff(fpts)
down = (afreqs[:, None] - fpts[None, :-2]) / fdiff[:-1]
up = (fpts[None, 2:] - afreqs[:, None]) / fdiff[1:]
fb = np.maximum(0, np.minimum(down, up)).astype(np.float64).T
k = np.arange(NLFCC)[:, None]
nn = np.arange(NFILT)[None, :]
dctm = np.cos(np.pi / NFILT * (nn + 0.5) * k) * np.sqrt(2 / NFILT)
dctm[0] /= np.sqrt(2)


def prep(path):
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav.mean(0)
    n = wav.shape[-1]
    if n > TARGET:
        s = (n - TARGET) // 2
        wav = wav[s:s + TARGET]
    elif n < TARGET:
        wav = torch.nn.functional.pad(wav, (0, TARGET - n))
    return wav.numpy()


def score_a(pcm):  # torchaudio front end
    x = lfcc_t(torch.tensor(pcm)[None]).transpose(1, 2)
    mu, sd = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True)
    x = ((x - mu) / (sd + 1e-6))[:, None]
    with torch.no_grad():
        return float(model(x))


def reflect(idx, n):
    src = idx.copy()
    m = src < 0
    src[m] = -src[m]
    o = src >= n
    src[o] = 2 * (n - 1) - src[o]
    return src


def score_b(pcm):  # Kotlin-equivalent replica
    nf = len(pcm) // HOP + 1
    filt = np.zeros((NFILT, nf))
    for ti in range(nf):
        base = ti * HOP - NFFT // 2
        frame = pcm[reflect(base + np.arange(NFFT), len(pcm))].astype(np.float64) * win
        sp = np.fft.rfft(frame)
        pw = (sp.real**2 + sp.imag**2)[:257]
        filt[:, ti] = fb @ pw
    lf = dctm @ (10 * np.log10(np.maximum(filt, 1e-10)))
    mu = lf.mean(axis=0, keepdims=True)
    sd = lf.std(axis=0, ddof=1, keepdims=True)
    x = ((lf - mu) / (sd + 1e-6)).T[None, None]   # (1,1,T,F)
    with torch.no_grad():
        return float(model(torch.tensor(x, dtype=torch.float32)))


test_dir = os.path.join(ROOT, "datasets", "asvspoof2019", "test")
files = []
for cls in ("real", "fake"):
    d = os.path.join(test_dir, cls)
    fs = sorted(os.listdir(d))[:6]
    files += [(os.path.join(d, f), cls) for f in fs]

thr = 0.4424
agree = 0
print(f"{'file':<28} {'label':<5} {'A(torchaudio)':>14} {'B(kotlin-eq)':>13} {'|dA-B|':>8}  verdicts")
for path, label in files:
    pcm = prep(path)
    sa, sb = score_a(pcm), score_b(pcm)
    va, vb = sa >= thr, sb >= thr
    agree += va == vb
    print(f"{os.path.basename(path):<28} {label:<5} {sa:>14.4f} {sb:>13.4f} "
          f"{abs(sa - sb):>8.4f}  {int(va)}|{int(vb)}")
print(f"\nverdict agreement: {agree}/{len(files)}")
