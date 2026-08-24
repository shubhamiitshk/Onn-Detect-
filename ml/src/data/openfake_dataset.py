import os
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Tuple, Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", "(?s).*Image size.*")
Image.MAX_IMAGE_PIXELS = None

class OpenFakeDataset(Dataset):
    """
    Unified Multi-Generator Visual Deepfake Dataset.
    Supports GenImage, Artifact, DiffusionDB, and OpenFake structures.
    Recursively discovers real vs. fake images across multi-generator subdirectories.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        img_size: int = 256,
        transform=None,
        cache_in_memory: bool = False,
        max_samples_per_class: Optional[int] = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.cache_in_memory = cache_in_memory and split == 'train'
        
        self.samples: List[Tuple[Path, int]] = []
        self.image_cache: Dict[int, torch.Tensor] = {}
        
        # Forensics-preserving transforms by default
        if transform is not None:
            self.transform = transform
        else:
            if self.split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                ])
                
        self._scan_dataset(max_samples_per_class)
        
        if self.cache_in_memory and self.samples:
            print(f"[CACHE] Caching {len(self.samples)} images into RAM...")
            for idx in tqdm(range(len(self.samples)), desc=f"Caching {split}"):
                self.image_cache[idx] = self._load_image_tensor(idx)
                
        dist = self.get_class_distribution()
        print(f"[OK] Init OpenFakeDataset ('{split}') - Total: {len(self.samples)} | Real (0): {dist['real']} | Fake (1): {dist['fake']}")

    def _scan_dataset(self, max_samples_per_class: Optional[int]):
        split_path = self.root_dir / self.split
        search_root = split_path if split_path.exists() else self.root_dir
        
        image_exts = {'.png', '.jpg', '.jpeg', '.webp'}
        
        real_samples = []
        fake_samples = []
        
        # Standard folder matching
        for p in search_root.rglob('*'):
            if p.is_file() and p.suffix.lower() in image_exts:
                parts = [part.lower() for part in p.parts]
                if 'real' in parts or 'authentic' in parts or 'nature' in parts or '0_real' in parts:
                    real_samples.append((p, 0))
                elif 'fake' in parts or 'synthetic' in parts or 'ai' in parts or '1_fake' in parts:
                    fake_samples.append((p, 1))
                    
        # Apply per-class balancing if requested
        if max_samples_per_class:
            random.seed(42)
            if len(real_samples) > max_samples_per_class:
                real_samples = random.sample(real_samples, max_samples_per_class)
            if len(fake_samples) > max_samples_per_class:
                fake_samples = random.sample(fake_samples, max_samples_per_class)
                
        self.samples = real_samples + fake_samples
        random.seed(42)
        random.shuffle(self.samples)

    def get_class_distribution(self) -> Dict[str, int]:
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        return {"real": real_count, "fake": fake_count}

    def _load_image_tensor(self, idx: int) -> torch.Tensor:
        path, _ = self.samples[idx]
        try:
            img = Image.open(str(path)).convert('RGB')
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, self.img_size, self.img_size)
        return tensor

    @property
    def labels(self) -> List[int]:
        return [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.cache_in_memory and idx in self.image_cache:
            return self.image_cache[idx], self.samples[idx][1]
        
        tensor = self._load_image_tensor(idx)
        return tensor, self.samples[idx][1]
