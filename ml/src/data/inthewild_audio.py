import os
import torch
import torchaudio
import random
from torch.utils.data import Dataset
from typing import Tuple, Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
from .lfcc_extractor import LFCCExtractor

class InTheWildAudioDataset(Dataset):
    """
    Dataset for In-The-Wild Audio and WaveFake deepfake audio detection.
    Supports directory structures:
      - root_dir/split/{real,fake}/*.{wav,flac,mp3,m4a}
      - root_dir/{real,fake}/*.{wav,flac,mp3,m4a} (with automatic split hashing)
      - Nested generator folders (e.g. root_dir/split/fake/hifigan/*.wav)
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        max_duration: float = 4.0,
        sample_rate: int = 16000,
        cache_in_memory: bool = False,
        subsample_limit: Optional[int] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.target_samples = int(max_duration * sample_rate)
        self.cache_in_memory = cache_in_memory and split == 'train'
        
        self.lfcc_extractor = LFCCExtractor(sample_rate=sample_rate)
        self.samples: List[Tuple[Path, int]] = []
        self.durations: List[float] = []
        self.feature_cache: Dict[int, torch.Tensor] = {}
        
        self._scan_dataset()
        
        if subsample_limit and len(self.samples) > subsample_limit:
            random.seed(42)
            self.samples = random.sample(self.samples, subsample_limit)
            
        if self.cache_in_memory and self.samples:
            print(f"[CACHE] Caching {len(self.samples)} LFCC audio features in RAM...")
            for idx in tqdm(range(len(self.samples)), desc=f"Caching {split}"):
                self.feature_cache[idx] = self._extract_feature(idx)
                
        dist = self.get_class_distribution()
        print(f"[OK] Init InTheWildAudioDataset ('{split}') - Total: {len(self.samples)} | Real (0): {dist['real']} | Fake (1): {dist['fake']}")

    def _scan_dataset(self):
        split_path = self.root_dir / self.split
        audio_exts = {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}
        
        all_samples = []
        # Check if the specific split directory exists and has audio files
        has_split_files = False
        if split_path.exists():
            for p in split_path.rglob('*'):
                if p.is_file() and p.suffix.lower() in audio_exts:
                    has_split_files = True
                    parts = [part.lower() for part in p.parts]
                    if 'real' in parts or 'authentic' in parts:
                        all_samples.append((p, 0))
                    elif 'fake' in parts or 'synthetic' in parts:
                        all_samples.append((p, 1))
                        
        if has_split_files:
            self.samples = all_samples
        else:
            # Fallback: scan whole root_dir and partition 80/10/10 deterministically
            real_samples = []
            fake_samples = []
            for p in self.root_dir.rglob('*'):
                if p.is_file() and p.suffix.lower() in audio_exts:
                    parts = [part.lower() for part in p.parts]
                    if 'real' in parts or 'authentic' in parts:
                        real_samples.append((p, 0))
                    elif 'fake' in parts or 'synthetic' in parts:
                        fake_samples.append((p, 1))
                        
            random.seed(42)
            random.shuffle(real_samples)
            random.shuffle(fake_samples)
            
            def partition(samples):
                n = len(samples)
                if n == 0:
                    return []
                if self.split == 'train':
                    return samples[:max(1, int(n * 0.8))]
                elif self.split in ['val', 'validation']:
                    return samples[int(n * 0.8):max(int(n * 0.8) + 1, int(n * 0.9))]
                else: # test
                    return samples[int(n * 0.9):] if int(n * 0.9) < n else samples[-1:]
                    
            self.samples = partition(real_samples) + partition(fake_samples)
            
        random.seed(42)
        random.shuffle(self.samples)

    @property
    def labels(self) -> List[int]:
        return [label for _, label in self.samples]

    def get_class_distribution(self) -> Dict[str, int]:
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        return {"real": real_count, "fake": fake_count}

    def _pad_or_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        num_channels, num_frames = waveform.shape
        if num_frames > self.target_samples:
            if self.split == 'train':
                max_offset = num_frames - self.target_samples
                offset = random.randint(0, max_offset)
                waveform = waveform[:, offset:offset + self.target_samples]
            else:
                offset = (num_frames - self.target_samples) // 2
                waveform = waveform[:, offset:offset + self.target_samples]
        elif num_frames < self.target_samples:
            pad_amount = self.target_samples - num_frames
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        return waveform

    def _extract_feature(self, idx: int) -> torch.Tensor:
        path, _ = self.samples[idx]
        try:
            waveform, sr = torchaudio.load(str(path))
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        except Exception:
            waveform = torch.zeros(1, self.target_samples)
            
        waveform = self._pad_or_crop(waveform)
        lfcc = self.lfcc_extractor.extract(waveform)
        return lfcc.squeeze(0) # [1, time_frames, n_lfcc]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.cache_in_memory and idx in self.feature_cache:
            return self.feature_cache[idx], self.samples[idx][1]
        
        feature = self._extract_feature(idx)
        return feature, self.samples[idx][1]
