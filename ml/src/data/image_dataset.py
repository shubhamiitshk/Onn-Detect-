"""Lazy image loader with a bounded per-worker LRU decode cache."""
from collections import OrderedDict
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .contracts import IMAGE_EXTS, balanced_subset, discover_samples
from .image_frontend import decode_image

def image_tensor(image, size):
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)

class StrictImageDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=32, transform=None,
                 cache_in_memory=False, max_samples_per_class=None, cache_bytes=32 * 1024 * 1024):
        self.root_dir, self.split, self.img_size = root_dir, split, img_size
        self.transform = transform
        self.samples = discover_samples(root_dir, split, IMAGE_EXTS)
        if max_samples_per_class is not None:
            self.samples = balanced_subset(self.samples, 2 * max_samples_per_class)
        self.labels = [label for _, label in self.samples]
        self.cache_limit = int(cache_bytes) if cache_in_memory else 0
        self.cache_size = 0
        self.image_cache = OrderedDict()

    def __len__(self):
        return len(self.samples)

    def get_class_distribution(self):
        return {'real': self.labels.count(0), 'fake': self.labels.count(1)}

    def __getitem__(self, index):
        path, label = self.samples[index]
        if index in self.image_cache:
            image = self.image_cache.pop(index)
            self.image_cache[index] = image
            image = image.copy()
        else:
            image = decode_image(path).image
            size = image.width * image.height * 3
            if size <= self.cache_limit:
                while self.image_cache and self.cache_size + size > self.cache_limit:
                    _, old = self.image_cache.popitem(last=False)
                    self.cache_size -= old.width * old.height * 3
                self.image_cache[index] = image.copy()
                self.cache_size += size
        tensor = self.transform(image) if self.transform else image_tensor(image, self.img_size)
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid image transform output for {path}')
        return tensor.float(), label
