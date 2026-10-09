"""Pixel-policy regressions without claiming Android codec equivalence."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageCms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data.image_frontend import decode_image, frontend_id
from meta_classifier.feature_extractor import quality_features, calibrate_baseline


class ImageFrontendTests(unittest.TestCase):
    def test_transparent_pixels_are_white_and_partial_alpha_is_composited(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'alpha.png'
            image = Image.new('RGBA', (2, 1))
            image.putdata([(255, 0, 0, 0), (255, 0, 0, 128)])
            image.save(path)
            decoded = decode_image(path)
            self.assertEqual(decoded.image.mode, 'RGB')
            np.testing.assert_array_equal(np.asarray(decoded.image), [[[255, 255, 255], [255, 127, 127]]])

    def test_exif_orientation_and_source_dimensions_survive_working_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'oriented.jpg'
            image = Image.new('RGB', (2400, 600), 'red')
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            decoded = decode_image(path)
            self.assertEqual(decoded.source_size, (600, 2400))
            self.assertEqual(decoded.sample_size, 4)
            self.assertEqual(decoded.image.size, (150, 600))
            # Quality uses original min edge, not the 150-pixel working edge.
            resolution, compressed = quality_features(decoded.image, str(path), decoded.source_size)
            self.assertAlmostEqual(resolution, 600 / 1024)
            self.assertEqual(compressed, 1)
            self.assertEqual(quality_features(decoded.image.copy(), str(path))[0], resolution)

    def test_icc_srgb_is_converted_and_invalid_profile_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.png'
            profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
            image = Image.new('RGB', (2, 2), (30, 80, 130))
            image.save(path, icc_profile=profile)
            np.testing.assert_array_equal(np.asarray(decode_image(path).image), np.asarray(image))
            image.save(path, icc_profile=b'invalid profile bytes')
            with self.assertRaises(RuntimeError):
                decode_image(path)

    def test_jpeg_feature_uses_encoding_even_with_mismatched_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            jpeg_path, png_path = Path(directory) / 'jpeg.png', Path(directory) / 'png.jpg'
            image = Image.new('RGB', (12, 12), 'blue')
            image.save(jpeg_path, format='JPEG')
            image.save(png_path, format='PNG')
            jpeg, png = decode_image(jpeg_path), decode_image(png_path)
            self.assertEqual(jpeg.source_format, 'JPEG')
            self.assertEqual(quality_features(jpeg.image, str(jpeg_path))[1], 1)
            self.assertEqual(quality_features(png.image, str(png_path))[1], 0)

    def test_baseline_uses_white_composite_and_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'alpha.png'
            image = Image.new('RGBA', (4, 2))
            image.putdata([(255, 0, 0, 0), (0, 0, 255, 0)] * 4)
            image.save(path)
            baseline = calibrate_baseline([str(path)])
            self.assertGreater(baseline['bins'][10], .999)
            self.assertEqual(baseline['num_images'], 1)
            self.assertIn('whitealpha-pow2cap1024', frontend_id(256))


if __name__ == '__main__':
    unittest.main()
