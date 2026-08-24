import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Tuple, Dict
from tqdm import tqdm
import warnings

# Suppress PIL max image pixels warnings if dealing with massive images
warnings.filterwarnings("ignore", "(?s).*Image size.*")
Image.MAX_IMAGE_PIXELS = None

class GenImageDataset(Dataset):
    """
    Dataset for GenImage or similar image-based visual deepfake datasets.
    Directory structure expected: root_dir/split/{real,fake}/
    """
    def __init__(self, root_dir: str, split: str = 'train', img_size: int = 224, cache_in_memory: bool = False):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.cache_in_memory = cache_in_memory and split == 'train'
        
        self.samples = []
        self.ram_cache = {}
        
        # Determine paths
        split_path = os.path.join(root_dir, split)
        real_dir = os.path.join(split_path, "real")
        fake_dir = os.path.join(split_path, "fake")
        
        # Load file paths
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.samples.append((os.path.join(real_dir, f), 0)) # 0 = real
                    
        if os.path.exists(fake_dir):
            for f in os.listdir(fake_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.samples.append((os.path.join(fake_dir, f), 1)) # 1 = fake
                    
        # Define Transforms
        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Pre-load to RAM if requested (Requires massive RAM for full datasets!)
        if self.cache_in_memory:
            print(f"Loading {len(self.samples)} images into RAM for split '{split}'...")
            for path, label in tqdm(self.samples, desc="Caching in memory"):
                try:
                    img = Image.open(path).convert('RGB')
                    # Resize prior to caching to save RAM
                    img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
                    self.ram_cache[path] = img
                except Exception as e:
                    print(f"Error loading {path}: {e}")

        # Print statistics
        dist = self.get_class_distribution()
        print(f"[OK] Init GenImageDataset ('{split}') - Total: {len(self.samples)} | Real (0): {dist['real']} | Fake (1): {dist['fake']}")

    def get_class_distribution(self) -> Dict[str, int]:
        """Returns the counts per class."""
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        return {"real": real_count, "fake": fake_count}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        
        if self.cache_in_memory and path in self.ram_cache:
            img = self.ram_cache[path].copy()
        else:
            try:
                img = Image.open(path).convert('RGB')
            except Exception:
                # Fallback on black image if corrupted
                img = Image.new('RGB', (self.img_size, self.img_size))
                
        tensor = self.transform(img)
        return tensor, label
