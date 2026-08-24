import os
import torch
import torchaudio
import random
from torch.utils.data import Dataset
from typing import Tuple, Dict, List
import math
from .lfcc_extractor import LFCCExtractor

class ASVspoofDataset(Dataset):
    """
    Dataset for ASVspoof audio deepfake detection.
    Directory structure expected: root_dir/split/{real,fake}/
    """
    def __init__(self, root_dir: str, split: str = 'train', max_duration: float = 4.0, sample_rate: int = 16000):
        self.root_dir = root_dir
        self.split = split
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.target_samples = int(max_duration * sample_rate)
        
        self.lfcc_extractor = LFCCExtractor(sample_rate=sample_rate)
        self.samples = []
        self.durations = []
        
        split_path = os.path.join(root_dir, split)
        real_dir = os.path.join(split_path, "real")
        fake_dir = os.path.join(split_path, "fake")
        
        # Helper to load and record stats
        def scan_dir(directory: str, label: int):
            if os.path.exists(directory):
                for f in os.listdir(directory):
                    if f.lower().endswith(('.wav', '.flac')):
                        path = os.path.join(directory, f)
                        self.samples.append((path, label))
                        # Record duration stats (using metadata to avoid loading full audio)
                        try:
                            info = torchaudio.info(path)
                            self.durations.append(info.num_frames / info.sample_rate)
                        except Exception:
                            pass
        
        scan_dir(real_dir, 0) # 0 = real
        scan_dir(fake_dir, 1) # 1 = fake
        
        # Print statistics
        dist = self.get_class_distribution()
        print(f"[OK] Init ASVspoofDataset ('{split}') - Total: {len(self.samples)} | Real (0): {dist['real']} | Fake (1): {dist['fake']}")
        if self.durations:
            stats = self.get_duration_stats()
            print(f"   Duration Stats -> Min: {stats['min']:.2f}s | Max: {stats['max']:.2f}s | Avg: {stats['avg']:.2f}s")

    def get_class_distribution(self) -> Dict[str, int]:
        """Returns the counts per class."""
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        return {"real": real_count, "fake": fake_count}

    def get_duration_stats(self) -> Dict[str, float]:
        """Returns min, max, and avg duration of the loaded audio files."""
        if not self.durations:
            return {"min": 0.0, "max": 0.0, "avg": 0.0}
        return {
            "min": min(self.durations),
            "max": max(self.durations),
            "avg": sum(self.durations) / len(self.durations)
        }

    def _pad_or_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pads with zeros or randomly crops the waveform to the target length."""
        num_channels, num_frames = waveform.shape
        
        if num_frames > self.target_samples:
            # Crop
            if self.split == 'train':
                max_offset = num_frames - self.target_samples
                offset = random.randint(0, max_offset)
                waveform = waveform[:, offset:offset + self.target_samples]
            else:
                # Center crop for evaluation
                offset = (num_frames - self.target_samples) // 2
                waveform = waveform[:, offset:offset + self.target_samples]
        elif num_frames < self.target_samples:
            # Zero-pad
            pad_amount = self.target_samples - num_frames
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        return waveform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        
        try:
            waveform, sr = torchaudio.load(path)
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        except Exception:
            # Fallback on empty silence
            waveform = torch.zeros(1, self.target_samples)
            
        waveform = self._pad_or_crop(waveform)
        
        # Extract LFCC -> expected output [1, 1, time_frames, n_lfcc_coeffs]
        # Squeeze the first dimension (batch) returned by LFCCExtractor since DataLoader will batch it
        lfcc = self.lfcc_extractor.extract(waveform)
        lfcc = lfcc.squeeze(0) # Returns shape (1, time_frames, 40)
        
        return lfcc, label
