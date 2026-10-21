import torch
import random
import io
import numpy as np
from PIL import Image


def _add_gaussian_noise_pil(img: Image.Image, sigma: float) -> Image.Image:
    """Add zero-mean Gaussian noise to a PIL image.

    `AdditiveNoiseAugmentation` works on tensors, but the paired image
    branch of DCPT operates on PIL images, so the same transform is needed
    in image space. Noise is applied in [0, 1] and clipped back to the
    original range.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((out * 255.0).round().astype(np.uint8), "RGB")

# torchvision is only needed for ResizeAugmentation; import lazily so that the
# phone-pipeline and JPEG augmentations are importable without torchvision.

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
        import torchvision.transforms.functional as TF  # lazy: optional dep

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
        return (tensor + noise).clamp(0.0, 1.0)


class PhonePipelineAugmentation:
    """Chains realistic phone-camera transformations so training features
    survive the path the Android app actually exercises.

    Apply in this order: downscale (phones+social recompress), re-encode at a
    random JPEG quality, optionally blur (motion/misfocus), then up-scale back
    toward the analysis size. Each stage is sampled independently.
    """

    def __init__(
        self,
        min_jpeg: int = 30,
        max_jpeg: int = 95,
        p_jpeg: float = 0.7,
        downscale_min: float = 0.5,
        downscale_max: float = 1.0,
        p_downscale: float = 0.5,
        p_blur: float = 0.2,
        target_size: int | None = None,
    ) -> None:
        import random

        self.min_jpeg = min_jpeg
        self.max_jpeg = max_jpeg
        self.p_jpeg = p_jpeg
        self.downscale_min = downscale_min
        self.downscale_max = downscale_max
        self.p_downscale = p_downscale
        self.p_blur = p_blur
        self.target_size = target_size
        self._rng = random

    def __call__(self, img):
        import io
        from PIL import Image, ImageFilter

        if self._rng.random() < self.p_downscale:
            f = self._rng.uniform(self.downscale_min, self.downscale_max)
            w, h = img.size
            small = (max(8, int(w * f)), max(8, int(h * f)))
            img = img.resize(small, Image.BILINEAR)
            img = img.resize((w, h), Image.BILINEAR)  # re-upscale

        if self._rng.random() < self.p_blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=self._rng.uniform(0.3, 1.5)))

        if self._rng.random() < self.p_jpeg:
            q = self._rng.randint(self.min_jpeg, self.max_jpeg)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        if self.target_size is not None:
            img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        return img


class PairedDegradationAugmentation:
    """Yield a (clean, degraded) pair from the SAME source image.

    DCPT (arXiv:2604.10102) shows that treating degradation robustness as a
    data-augmentation by-product is not enough: the objective must
    explicitly constrain a clean view and a degraded view of ONE image to
    agree, both in feature space and in the output distribution.

    This class produces the two views; the losses live in
    ``training/dcpt_losses.py``.

    The degradation chain is the one a real distribution channel applies -
    JPEG re-encoding, downscaling and noise - sampled independently per
    stage, and applied in the order a file actually meets them.
    """

    def __init__(
        self,
        p_degrade: float = 0.8,
        min_jpeg: int = 30,
        max_jpeg: int = 90,
        p_jpeg: float = 0.6,
        min_size: int = 160,
        max_size: int = 320,
        p_resize: float = 0.4,
        min_sigma: float = 0.01,
        max_sigma: float = 0.04,
        p_noise: float = 0.3,
        target_size: int | None = None,
    ) -> None:
        self.p_degrade = p_degrade
        self.p_jpeg = p_jpeg
        self.p_resize = p_resize
        self.p_noise = p_noise
        self.target_size = target_size
        self._jpeg = JPEGCompressionAugmentation(min_quality=min_jpeg, max_quality=max_jpeg, p=1.0)
        self._resize = ResizeAugmentation(min_size=min_size, max_size=max_size, target_size=target_size or 224, p=1.0)
        self._noise = AdditiveNoiseAugmentation(min_sigma=min_sigma, max_sigma=max_sigma, p=1.0)

    def __call__(self, img: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Return ``(clean, degraded)`` views of ``img``."""
        clean = img
        if random.random() >= self.p_degrade:
            return clean, clean

        view = clean
        if random.random() < self.p_resize:
            view = self._resize(view)
        if random.random() < self.p_noise:
            view = _add_gaussian_noise_pil(view, random.uniform(self.min_sigma, self.max_sigma))
        if random.random() < self.p_jpeg:
            view = self._jpeg(view).convert("RGB")
        if self.target_size is not None:
            view = view.resize((self.target_size, self.target_size), Image.Resampling.BILINEAR)
        return clean, view


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
            import librosa  # lazy: audio-only dependency, keep module import light

            stretched = librosa.effects.time_stretch(y=np_wave, rate=rate)
            return torch.from_numpy(stretched)
        except Exception:
            return waveform
