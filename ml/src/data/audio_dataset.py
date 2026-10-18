"""Strict lazy speech loader using the deployed resampler and LFCC contract."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .contracts import AUDIO_EXTS, balanced_subset, discover_samples
from .lfcc_extractor import LFCCExtractor
from .lfcc_reference import resample_sinc

def load_analysis_waveform(path, split='calibration', sample_rate=16000):
    """Read a bounded native-rate source window, then resample/crop like Android."""
    import soundfile as sf
    try:
        info = sf.info(path)
        rate, frames = int(info.samplerate), int(info.frames)
        if not 1 <= rate <= 384000 or frames <= 0:
            raise ValueError('Empty audio or unsupported sample rate')
        source_count = int(np.ceil(64000 * rate / sample_rate))
        extra = max(0, frames - source_count)
        start = random.randint(0, extra) if split == 'train' else extra // 2
        waveform, decoded_rate = sf.read(path, start=start, stop=min(frames, start + source_count), dtype='float32', always_2d=True)
        if decoded_rate != rate:
            raise ValueError('Audio metadata/decoded sample rate mismatch')
    except Exception as exc:
        raise RuntimeError(f'Unable to decode audio {path}; quarantine it before training') from exc
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f'Empty/non-finite audio: {path}')
    mono = resample_sinc(waveform.mean(axis=1, dtype=np.float64), rate, sample_rate).astype(np.float32)
    if len(mono) > 64000:
        offset = (len(mono) - 64000) // 2
        mono = mono[offset:offset + 64000]
    elif len(mono) < 64000:
        mono = np.pad(mono, (0, 64000 - len(mono)))
    return torch.from_numpy(mono)

def augment_train_waveform(mono: torch.Tensor) -> torch.Tensor:
    """Realistic channel/gain/noise perturbations for training speech windows only."""
    x = mono.clone().float()
    # 1. Random gain variation
    gain = random.uniform(0.65, 1.35)
    x = x * gain
    # 2. Random 1st-order spectral tilt / pre-emphasis (channel response variation)
    if random.random() < 0.35 and x.numel() > 1:
        alpha = random.uniform(-0.20, 0.20)
        tilted = x.clone()
        tilted[1:] = x[1:] + alpha * x[:-1]
        x = tilted
    # 3. Random SNR-bounded additive Gaussian noise
    if random.random() < 0.35:
        rms = float(torch.sqrt(torch.mean(x * x) + 1e-12).item())
        if rms > 1e-4:
            snr_db = random.uniform(20.0, 38.0)
            noise_std = rms / (10.0 ** (snr_db / 20.0))
            x = x + torch.randn_like(x) * noise_std
    return torch.clamp(x, -1.0, 1.0)

class StrictAudioDataset(Dataset):
    def __init__(self, root_dir, split='train', max_duration=4.0, sample_rate=16000,
                 cache_in_memory=False, subsample_limit=None):
        if sample_rate != 16000 or max_duration != 4.0:
            raise ValueError('Current mobile LFCC contract requires 16000 Hz and four seconds')
        if cache_in_memory:
            raise ValueError('Audio feature caching freezes random crops; use a lazy loader')
        self.root_dir, self.split = root_dir, split
        self.sample_rate, self.target_samples = sample_rate, int(max_duration * sample_rate)
        self.samples = discover_samples(root_dir, split, AUDIO_EXTS)
        if subsample_limit is not None:
            self.samples = balanced_subset(self.samples, subsample_limit)
        self.labels = [label for _, label in self.samples]
        self.lfcc_extractor = LFCCExtractor(sample_rate=sample_rate)

    def __len__(self):
        return len(self.samples)

    def get_class_distribution(self):
        return {'real': self.labels.count(0), 'fake': self.labels.count(1)}

    def __getitem__(self, index):
        path, label = self.samples[index]
        mono = load_analysis_waveform(path, self.split, self.sample_rate)
        if self.split == 'train':
            mono = augment_train_waveform(mono)
        feature = self.lfcc_extractor.extract(mono).squeeze(0).float()
        if feature.shape != (1, 401, 40) or not torch.isfinite(feature).all():
            raise ValueError(f'Invalid LFCC feature for {path}: {feature.shape}')
        return feature, label

