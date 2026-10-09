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
        feature = self.lfcc_extractor.extract(mono).squeeze(0).float()
        if feature.shape != (1, 401, 40) or not torch.isfinite(feature).all():
            raise ValueError(f'Invalid LFCC feature for {path}: {feature.shape}')
        return feature, label
