"""CIFAKE Dataset Loader - Optimized for deepfake detection"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random

class CIFAKEDataset(Dataset):
    """
    CIFAKE: Real and AI-Generated Synthetic Images
    - Real: CIFAR-10 photographs
    - Fake: Stable Diffusion generated images
    """
    
    def __init__(self, root_dir, split='train', img_size=224, transform=None, cache_in_memory=False):
        """
        Args:
            root_dir: Path to genimage directory
            split: 'train', 'val', or 'test'
            img_size: Target image size (default 224)
            transform: torchvision transforms
            cache_in_memory: If True, load all images into RAM (requires ~10GB RAM)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.cache_in_memory = cache_in_memory
        
        # Collect all image paths
        self.samples = []
        self.labels = []
        
        real_dir = self.root_dir / split / 'real'
        fake_dir = self.root_dir / split / 'fake'
        
        # Real images (label 0)
        if real_dir.exists():
            for img_path in sorted(real_dir.glob('*.jpg')):
                self.samples.append(img_path)
                self.labels.append(0)
        
        # Fake images (label 1)
        if fake_dir.exists():
            for img_path in sorted(fake_dir.glob('*.jpg')):
                self.samples.append(img_path)
                self.labels.append(1)
        
        # Shuffle
        combined = list(zip(self.samples, self.labels))
        random.shuffle(combined)
        self.samples, self.labels = zip(*combined) if combined else ([], [])
        
        # Cache in memory if requested
        self.image_cache = {}
        if self.cache_in_memory and self.samples:
            print(f"[CACHE] Caching {len(self.samples)} images in memory...")
            for idx, img_path in enumerate(tqdm(self.samples, desc="Loading")):
                try:
                    img = Image.open(img_path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    self.image_cache[idx] = img
                except Exception as e:
                    print(f"[ERROR] Error loading {img_path}: {e}")
        
        print(f"[OK] Init CIFAKEDataset ('{split}') - Total: {len(self.samples)} | "
              f"Real (0): {self.labels.count(0)} | Fake (1): {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Return cached image if available
        if self.cache_in_memory and idx in self.image_cache:
            return self.image_cache[idx], self.labels[idx]
        
        # Load from disk
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Return black image on error
            print(f"⚠️  Error loading {img_path}: {e}")
            black_image = torch.zeros(3, self.img_size, self.img_size)
            return black_image, label
