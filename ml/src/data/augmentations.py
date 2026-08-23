import torch
import random
import io
import librosa
from PIL import Image
import torchvision.transforms.functional as TF

class JPEGCompressionAugmentation:
    """
    Randomly applies JPEG compression to an image to improve model robustness
    against social media compression artifacts.
    """
    def __init__(self, min_quality: int = 30, max_quality: int = 95, p: float = 0.5):
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
            
        quality = random.randint(self.min_quality, self.max_quality)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)


class ResizeAugmentation:
    """
    Randomly resizes the image down/up to a random size between min_size and max_size,
    then resizes back to the target size. This simulates scaling artifacts.
    """
    def __init__(self, min_size: int = 128, max_size: int = 384, target_size: int = 224, p: float = 0.5):
        self.min_size = min_size
        self.max_size = max_size
        self.target_size = target_size
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            # Still ensure it hits target size if no augmentation applied
            return TF.resize(img, [self.target_size, self.target_size])
            
        temp_size = random.randint(self.min_size, self.max_size)
        img = TF.resize(img, [temp_size, temp_size])
        img = TF.resize(img, [self.target_size, self.target_size])
        return img


class AdditiveNoiseAugmentation:
    """
    Adds small Gaussian noise to the image tensor.
    Expects input to be a PyTorch Tensor in [0, 1] or normalized.
    """
    def __init__(self, min_sigma: float = 0.01, max_sigma: float = 0.05, p: float = 0.5):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
            
        sigma = random.uniform(self.min_sigma, self.max_sigma)
        noise = torch.randn_like(tensor) * sigma
        return tensor + noise


class AudioTimeStretchAugmentation:
    """
    Slight time stretching for audio waveforms (0.95x to 1.05x).
    Uses librosa for time-domain stretching.
    """
    def __init__(self, min_rate: float = 0.95, max_rate: float = 1.05, p: float = 0.5):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.p = p

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform (torch.Tensor): shape (channels, time)
        """
        if random.random() > self.p:
            return waveform
            
        rate = random.uniform(self.min_rate, self.max_rate)
        # librosa expects numpy array of shape (time,) or (channels, time)
        np_wave = waveform.numpy()
        
        try:
            stretched = librosa.effects.time_stretch(y=np_wave, rate=rate)
            return torch.from_numpy(stretched)
        except Exception:
            # Fallback in case of librosa errors on very short clips
            return waveform
