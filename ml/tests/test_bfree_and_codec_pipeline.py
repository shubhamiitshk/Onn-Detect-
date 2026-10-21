"""Unit tests for Pillar 1: B-Free / Codec-Matched Corpus and Audio Presentation Pipeline."""
import unittest
import numpy as np
from PIL import Image

from ml.src.data.bfree_paired_corpus import (
    CodecProfile,
    STANDARD_CODEC_PROFILES,
    apply_codec_profile,
    BFreePairedAugmentation,
    CorpusSample,
    CodecMatchedBatchSampler,
)
from ml.src.data.audio_codec_pipeline import (
    AudioPresentationProfile,
    add_calibrated_noise,
    apply_synthetic_reverb,
    simulate_codec_degradation,
    AudioPresentationPipeline,
)
from ml.scripts.evaluate_degradation_matrix import (
    compute_metrics,
    evaluate_degradations,
)


class BFreeAndCodecPipelineTest(unittest.TestCase):

    def setUp(self):
        self.test_img = Image.new("RGB", (256, 256), color=(120, 80, 200))

    def test_apply_codec_profile_jpeg(self):
        prof = CodecProfile("test_jpeg", "JPEG", quality=50, downscale_ratio=0.75)
        out = apply_codec_profile(self.test_img, prof)
        self.assertEqual(out.size, (256, 256))
        self.assertEqual(out.mode, "RGB")

    def test_apply_codec_profile_webp(self):
        prof = CodecProfile("test_webp", "WEBP", quality=60, blur_radius=0.5, noise_sigma=0.02)
        out = apply_codec_profile(self.test_img, prof)
        self.assertEqual(out.size, (256, 256))
        self.assertEqual(out.mode, "RGB")

    def test_bfree_paired_augmentation_triplets(self):
        aug = BFreePairedAugmentation(target_size=224, p_degrade=1.0, seed=42)
        clean, degraded, prof = aug(self.test_img)
        self.assertEqual(clean.size, (224, 224))
        self.assertEqual(degraded.size, (224, 224))
        self.assertIsNotNone(prof)

    def test_codec_matched_batch_sampler(self):
        samples = [
            CorpusSample(f"real_{i}", "laion", 0, f"/fake/path/real_{i}.png")
            for i in range(10)
        ] + [
            CorpusSample(f"fake_{i}", "flux", 1, f"/fake/path/fake_{i}.png")
            for i in range(10)
        ]
        sampler = CodecMatchedBatchSampler(samples, batch_size=8, seed=123)
        batch = sampler.sample_balanced_batch()
        self.assertEqual(len(batch), 8)
        reals = sum(1 for s in batch if s.label == 0)
        fakes = sum(1 for s in batch if s.label == 1)
        self.assertEqual(reals, 4)
        self.assertEqual(fakes, 4)

    def test_audio_calibrated_noise_snr(self):
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        clean_sine = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        noisy = add_calibrated_noise(clean_sine, snr_db=20.0)
        self.assertEqual(noisy.shape, clean_sine.shape)
        self.assertFalse(np.isnan(noisy).any())

    def test_audio_synthetic_reverb(self):
        waveform = np.random.randn(8000).astype(np.float32) * 0.1
        reverbed = apply_synthetic_reverb(waveform, sample_rate=16000, rt60=0.2)
        self.assertEqual(reverbed.shape, waveform.shape)
        self.assertFalse(np.isnan(reverbed).any())

    def test_audio_codec_simulation(self):
        waveform = np.random.randn(16000).astype(np.float32) * 0.2
        opus_out = simulate_codec_degradation(waveform, "opus_16k", 16000)
        self.assertEqual(opus_out.shape, waveform.shape)
        self.assertFalse(np.isnan(opus_out).any())

    def test_audio_presentation_pipeline(self):
        pipeline = AudioPresentationPipeline(sample_rate=16000, p_apply=1.0, seed=99)
        waveform = np.random.randn(16000).astype(np.float32) * 0.1
        clean, presented = pipeline(waveform)
        self.assertEqual(clean.shape, (16000,))
        self.assertEqual(presented.shape, (16000,))
        self.assertTrue(np.max(np.abs(presented)) <= 1.0)

    def test_compute_metrics_auc(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.85, 0.9])
        m = compute_metrics(labels, scores, threshold=0.5)
        self.assertAlmostEqual(m["auc"], 1.0, places=2)
        self.assertAlmostEqual(m["accuracy"], 1.0, places=2)
        self.assertEqual(m["fpr"], 0.0)
        self.assertEqual(m["tpr"], 1.0)


if __name__ == "__main__":
    unittest.main()
