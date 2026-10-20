"""Unavailable evidence must never become a zero or a neutral value.

These are regression guards for three silent-zero paths in
``meta_classifier/feature_extractor.py``. Each one previously returned a
plausible-looking value that the fusion contract would interpret as measured
evidence:

* ``power_law_slope`` returned the natural slope ``-2.0``, which reaches the
  feature vector through ``abs(slope - (-2.0)) / 2`` and becomes
  ``fft_score = 0.0`` - maximally authentic under ``real=0_fake=1``.
* ``diff_histogram`` returned an unnormalized ``np.full(21, 1e-6)`` summing to
  2.1e-5 and then fed it to ``kl_score`` as a probability distribution.
* ``wiener_entropy`` still returns 0.0 for an all-zero spectrum, which is
  platform-parity-preserving with the Kotlin implementation and therefore
  deliberately left alone; it is pinned here so that decision stays explicit.

The Android counterparts raise instead: ``FFTSpectralAnalyzer.kt`` requires
"Insufficient spectral evidence" / "Degenerate spectrum", and
``KLCooccurrenceAnalyzer.kt`` errors with "Insufficient pixel differences".
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meta_classifier import feature_extractor as fx


class DegenerateSpectrumIsUnavailableTest(unittest.TestCase):
    def test_flat_radial_profile_raises_instead_of_reporting_natural_slope(self):
        with self.assertRaises(ValueError) as caught:
            fx.power_law_slope(np.zeros(180))
        self.assertIn("Insufficient spectral evidence", str(caught.exception))

    def test_degenerate_ols_denominator_raises(self):
        radial = np.zeros(180)
        # Exactly two usable bins at the same radius cannot determine a slope.
        radial[5] = 1.0
        radial[5] = radial[5]
        radial = np.zeros(180)
        radial[5] = 1e-3
        radial[5] = 1e-3
        with self.assertRaises(ValueError):
            fx.power_law_slope(radial)

    def test_fft_features_on_a_constant_colour_image_never_return_score_zero(self):
        """A flat image has no usable spectrum and must not score as authentic."""
        from PIL import Image

        flat = Image.new("RGB", (64, 64), (128, 128, 128))
        with self.assertRaises(ValueError):
            fx.fft_features(flat)

    def test_real_natural_image_still_produces_a_score(self):
        """The raise must not fire on ordinary content."""
        rng = np.random.default_rng(20261008)
        array = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        from PIL import Image

        score, flatness = fx.fft_features(Image.fromarray(array))
        self.assertTrue(np.isfinite(score) and 0.0 <= score <= 1.0)
        self.assertTrue(np.isfinite(flatness) and 0.0 <= flatness <= 1.0)

    def test_wiener_entropy_zero_is_deliberate_platform_parity(self):
        self.assertEqual(fx.wiener_entropy(np.zeros(8)), 0.0)


class DegenerateHistogramIsUnavailableTest(unittest.TestCase):
    def test_single_column_grayscale_raises(self):
        with self.assertRaises(ValueError) as caught:
            fx.diff_histogram(np.zeros((16, 1), dtype=np.uint8))
        self.assertIn("No adjacent pixel pairs", str(caught.exception))

    def test_one_dimensional_input_raises(self):
        with self.assertRaises(ValueError):
            fx.diff_histogram(np.zeros(16, dtype=np.uint8))

    def test_valid_histogram_is_a_normalized_distribution(self):
        gray = np.random.default_rng(7).integers(0, 256, size=(32, 32), dtype=np.uint8)
        hist = fx.diff_histogram(gray)
        self.assertEqual(hist.shape, (fx.KL_BINS,))
        self.assertAlmostEqual(float(hist.sum()), 1.0, places=12)
        self.assertTrue((hist > 0).all())

    def test_kl_score_never_treats_missing_evidence_as_perfectly_authentic(self):
        gray = np.zeros((8, 1), dtype=np.uint8)
        baseline = np.full(fx.KL_BINS, 1.0 / fx.KL_BINS)
        with self.assertRaises(ValueError):
            fx.kl_score(gray, baseline)

    def test_baseline_calibration_fails_loudly_on_unusable_samples(self):
        with self.assertRaises(RuntimeError):
            fx.calibrate_baseline(["definitely-not-an-image.png"])


class FeatureVectorContractTest(unittest.TestCase):
    def test_provenance_measurements_are_rejected(self):
        """Supplying provenance must fail before any model or decode work."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            Image.new("RGB", (32, 32), (90, 120, 200)).save(path)
            with self.assertRaises(ValueError) as caught:
                fx.extract_visual_features(
                    str(path), None, None, np.full(fx.KL_BINS, 1.0 / fx.KL_BINS),
                    provenance={"c2pa": True},
                )
        self.assertIn("provenance", str(caught.exception).lower())

    def test_feature_names_match_the_android_order(self):
        self.assertEqual(fx.FEATURE_NAMES, [
            "srm_score", "fft_score", "kl_score", "flatness_score",
            "resolution_norm", "is_compressed", "semantic_probe_score", "c2pa_present",
            "lip_sync_corr", "landmark_jitter",
        ])

    def test_semantic_probe_slot_is_the_only_optional_slot(self):
        """Only slot 6 may be unavailable; every other slot must be finite."""
        self.assertEqual(fx.OPTIONAL_FEATURE_INDICES, (fx.SEMANTIC_PROBE_INDEX,))
        self.assertEqual(fx.FEATURE_NAMES[fx.SEMANTIC_PROBE_INDEX], "semantic_probe_score")


class SemanticProbeSlotTest(unittest.TestCase):
    """Slot 6 carries a real measurement or is UNAVAILABLE - never a neutral.

    A neutral constant (0.1) is indistinguishable from "low anomaly" to a
    tree that splits on it, so it must never stand in for a probe that did
    not run.
    """

    def test_absent_probe_is_nan_not_a_neutral_constant(self):
        from meta_classifier.dual_engine import assemble_features

        vector = assemble_features(
            srm_score=0.4, fft_score=0.2, kl_score=0.1, flatness_score=0.3,
            resolution_norm=0.5, is_compressed=True, semantic_probe_score=None,
        )
        self.assertTrue(np.isnan(vector[fx.SEMANTIC_PROBE_INDEX]))
        self.assertFalse(np.isnan(vector[0]))
        self.assertTrue(np.isfinite(vector[[i for i in range(10) if i != fx.SEMANTIC_PROBE_INDEX]]).all())

    def test_present_probe_lands_in_slot_six(self):
        from meta_classifier.dual_engine import assemble_features

        vector = assemble_features(
            srm_score=0.4, fft_score=0.2, kl_score=0.1, flatness_score=0.3,
            resolution_norm=0.5, is_compressed=True, semantic_probe_score=0.87,
        )
        self.assertAlmostEqual(float(vector[fx.SEMANTIC_PROBE_INDEX]), 0.87, places=6)
        self.assertTrue(np.isfinite(vector).all())

    def test_out_of_range_probe_score_is_rejected(self):
        from meta_classifier.dual_engine import assemble_features

        with self.assertRaises(ValueError):
            assemble_features(
                srm_score=0.4, fft_score=0.2, kl_score=0.1, flatness_score=0.3,
                resolution_norm=0.5, is_compressed=True, semantic_probe_score=1.5,
            )



if __name__ == "__main__":
    unittest.main()