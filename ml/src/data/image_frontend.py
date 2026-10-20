"""Canonical image decode policy shared by training, fusion and export.

EXIF orientation, sRGB conversion, white alpha compositing and a power-of-two
1024-pixel working cap mirror the intended Android policy. Pillow and Android
codec sampling/bilinear implementations still require measured device parity;
this helper does not certify their pixel-level equivalence.
"""
from dataclasses import dataclass
import io
from PIL import Image, ImageCms, ImageOps

WORKING_CAP = 1024
SOURCE_SIZE_KEY = 'onn_source_size'
SOURCE_FORMAT_KEY = 'onn_source_format'

# Engine C ("Lite C") appearance frontend. The semantic probe was distilled
# from a CLIP vision tower, so it consumes CLIP-style normalization rather
# than the [0,1] RGB scale Engine A uses. The Android counterpart lives in
# `detection/visual/SemanticProbeAnalyzer.kt` and must match exactly.
SEMANTIC_SIZE = 224
# The frozen vision tower (MobileCLIP2-S0) takes 256x256 with a shortest-edge
# resize and a centre crop, per its own open_clip preprocessing config.
FROZEN_PROBE_SIZE = 256
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def frontend_id(size):
    return f'rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-size{size}-float32'


def semantic_frontend_id(size=SEMANTIC_SIZE):
    """Versioned identifier for the Engine C appearance frontend."""
    return f'clip-appearance-bicubic{size}-clipnorm-meanstd-float32'


def frozen_probe_frontend_id(size=FROZEN_PROBE_SIZE):
    """Versioned identifier for the frozen vision-tower frontend."""
    return f'clip-frozen-tower-shortest{size}-centercrop-bicubic-clipnorm-meanstd-float32'


def semantic_image_tensor(image, size=SEMANTIC_SIZE):
    """Bicubic square resize + CLIP mean/std normalization -> CHW float32.

    Mirrors the Android `SemanticProbeAnalyzer` byte-for-byte. The resampling
    filter differs from the Engine A frontend on purpose: the two engines
    must not share a resampling artifact, or the fusion would be measuring
    the same low-level cue twice.
    """
    import numpy as np
    import torch

    resized = image.convert('RGB').resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray(CLIP_MEAN, dtype=np.float32)
    std = np.asarray(CLIP_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1))


def frozen_probe_image_tensor(image, size=FROZEN_PROBE_SIZE):
    """Frozen vision-tower (MobileCLIP2-S0) preprocessing -> CHW float32.

    This reproduces the tower's own `open_clip` transform exactly:

      1. the SHORTEST edge is resized to ``size`` and the longer edge
         becomes ``int(size * long / short)`` (torchvision's scalar Resize);
      2. a bicubic resize to those dimensions;
      3. a centre crop of ``size x size`` whose offsets are
         ``int(round((dim - size) / 2.0))``;
      4. per-channel ``(x / 255 - mean) / std`` with the CLIP constants.

    The resample/crop order matters: a plain square squash would feed the
    tower different pixels than it was trained on, and the measured probe AUC
    would not transfer to the device.
    """
    import numpy as np
    import torch

    rgb = image.convert('RGB')
    width, height = rgb.size
    if width <= 0 or height <= 0:
        raise ValueError('Empty image')
    short, long = (width, height) if width <= height else (height, width)
    new_short = size
    new_long = int(size * long / short)
    if width <= height:
        new_size = (new_short, new_long)
    else:
        new_size = (new_long, new_short)

    resized = rgb.resize(new_size, Image.Resampling.BICUBIC)
    resized_width, resized_height = resized.size
    crop_top = int(round((resized_height - size) / 2.0))
    crop_left = int(round((resized_width - size) / 2.0))
    cropped = resized.crop((crop_left, crop_top, crop_left + size, crop_top + size))

    arr = np.asarray(cropped, dtype=np.float32) / 255.0
    mean = np.asarray(CLIP_MEAN, dtype=np.float32)
    std = np.asarray(CLIP_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1))




@dataclass(frozen=True)
class DecodedImage:
    image: Image.Image
    source_size: tuple[int, int]
    sample_size: int
    source_format: str


def decode_image(path):
    """Decode one file, retaining oriented source dimensions for quality features.

    Untagged color is assumed sRGB. A malformed or unsupported embedded ICC
    profile fails explicitly rather than silently using incompatible pixels.
    JPEG draft decoding can bound the first decode; other codecs may decode
    their native image before the bounded working bitmap is produced.
    """
    try:
        with Image.open(path) as source:
            source_format = source.format or ''
            width, height = source.size
            if width <= 0 or height <= 0:
                raise ValueError('Empty image')
            orientation = source.getexif().get(274, 1)
            original_size = (height, width) if orientation in (5, 6, 7, 8) else (width, height)
            sample = 1
            while max(width, height) // sample > WORKING_CAP:
                sample *= 2
            target = (max(1, width // sample), max(1, height // sample))
            # RGB JPEGs can be sampled in the codec before allocating full RGB.
            # CMYK color conversion needs the source channels intact.
            if source.format == 'JPEG' and source.mode in ('RGB', 'L') and sample > 1:
                source.draft(source.mode, target)
            image = ImageOps.exif_transpose(source)
            oriented_target = target[::-1] if orientation in (5, 6, 7, 8) else target
            if image.size != oriented_target:
                image = image.resize(oriented_target, Image.Resampling.BILINEAR)
            alpha = image.convert('RGBA').getchannel('A') if 'A' in image.getbands() or 'transparency' in image.info else None
            icc = source.info.get('icc_profile')
            if icc:
                try:
                    profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                    cms_input = image if image.mode in ('RGB', 'RGBA', 'CMYK', 'L', 'LAB') else image.convert('RGB')
                    rgb = ImageCms.profileToProfile(cms_input, profile, ImageCms.createProfile('sRGB'), outputMode='RGB')
                except Exception as exc:
                    raise ValueError('Embedded ICC profile cannot be converted to sRGB') from exc
            else:
                rgb = image.convert('RGB')
            if alpha is not None:
                white = Image.new('RGB', rgb.size, 'white')
                white.paste(rgb, mask=alpha)
                rgb = white
            rgb.load()
            rgb.info[SOURCE_SIZE_KEY] = original_size
            rgb.info[SOURCE_FORMAT_KEY] = source_format
            return DecodedImage(rgb, original_size, sample, source_format)
    except Exception as exc:
        raise RuntimeError(f'Unable to decode image {path} with the versioned frontend; quarantine it before training') from exc
