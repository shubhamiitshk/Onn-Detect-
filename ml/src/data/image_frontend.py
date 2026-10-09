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


def frontend_id(size):
    return f'rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-size{size}-float32'


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
