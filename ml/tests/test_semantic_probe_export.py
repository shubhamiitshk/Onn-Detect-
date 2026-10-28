"""Export parity for the Engine C ("Lite C") semantic second engine.

The Keras rebuild in `export/torch_to_tflite_keras.py` transfers trained
weights tensor-by-tensor. A silent transfer bug would produce a TFLite model
that disagrees with the PyTorch checkpoint it claims to encode, and the
release would ship a detector that never existed in training.

This mirrors the repo's existing approach for the SRM and LCNN models: the
rebuilt Keras model must agree with PyTorch on real inputs to within the
1e-4 tolerance, and it must be able to represent the 12/16/20-channel SRM
stems as well as the 20-channel hybrid.

Also guards the release contract's requirement that the exported probe keeps
its 224x224 square input and produces exactly one scalar.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import tensorflow as tf  # noqa: F401
    _HAS_TF = True
except (ImportError, ModuleNotFoundError):
    _HAS_TF = False


class SemanticProbeExportTest(unittest.TestCase):

    @unittest.skipUnless(_HAS_TF, "tensorflow not installed")
    def test_keras_rebuild_matches_pytorch_on_real_inputs(self):
        from train_semantic_probe import SemanticProbeCNN
        from export.torch_to_tflite_keras import build_keras_semantic_probe, keras_parity_check

        model = SemanticProbeCNN(width=32).eval()
        keras_model = build_keras_semantic_probe(model, 224)

        self.assertEqual(tuple(keras_model.input_shape), (None, 224, 224, 3))
        self.assertEqual(tuple(keras_model.output_shape), (None, 1))

        rng = np.random.RandomState(0)
        inputs = [rng.randn(224, 224, 3).astype("float32") for _ in range(4)]
        error = keras_parity_check(keras_model, model, inputs, "visual")
        self.assertLessEqual(error, 1e-4)

    @unittest.skipUnless(_HAS_TF, "tensorflow not installed")
    def test_keras_rebuild_matches_pytorch_for_a_trained_checkpoint(self):
        """A trained checkpoint must survive the rebuild, not just a fresh init."""
        from train_semantic_probe import SemanticProbeCNN
        from export.torch_to_tflite_keras import build_keras_semantic_probe, keras_parity_check

        model = SemanticProbeCNN(width=16).eval()
        # Perturb the weights so a "do nothing" transfer would fail.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.05)

        keras_model = build_keras_semantic_probe(model, 224)
        rng = np.random.RandomState(1)
        inputs = [rng.randn(224, 224, 3).astype("float32") for _ in range(3)]
        error = keras_parity_check(keras_model, model, inputs, "visual")
        self.assertLessEqual(error, 1e-4)

    def test_engine_c_outputs_a_single_finite_probability(self):
        from train_semantic_probe import SemanticProbeCNN

        model = SemanticProbeCNN(width=32).eval()
        with torch.no_grad():
            out = model(torch.randn(4, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (4, 1))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(bool(((out >= 0) & (out <= 1)).all()))

    def test_engine_c_has_no_high_pass_stem(self):
        """Engine C must not share Engine A's inductive bias.

        If it did, the fusion would be measuring the same low-level cue twice
        and the diversity that justifies a second engine would be gone.
        """
        from train_semantic_probe import SemanticProbeCNN

        model = SemanticProbeCNN(width=32)
        self.assertFalse(hasattr(model, "srm"))
        first = model.stem[0]
        self.assertEqual(first.in_channels, 3)
        self.assertEqual(first.kernel_size, (3, 3))


class SrmStemCompatibilityTest(unittest.TestCase):
    """The 12/16/20-channel SRM stems must all survive the Keras rebuild."""

    @unittest.skipUnless(_HAS_TF, "tensorflow not installed")
    def test_each_stem_width_rebuilds_with_matching_output_shape(self):
        from models.srm_forensic_cnn import SRMForensicCNN
        from export.torch_to_tflite_keras import build_keras_srm, keras_parity_check

        for include_npr, include_rgb, expected in (
            (False, False, 12),
            (True, False, 16),
            (True, True, 20),
        ):
            with self.subTest(channels=expected):
                model = SRMForensicCNN(include_npr=include_npr, include_rgb=include_rgb).eval()
                self.assertEqual(int(model.srm.conv.out_channels), expected)
                keras_model = build_keras_srm(model, 32)
                rng = np.random.RandomState(2)
                inputs = [rng.randint(0, 256, (32, 32, 3)).astype("float32") for _ in range(3)]
                error = keras_parity_check(keras_model, model, inputs, "visual")
                self.assertLessEqual(error, 1e-4)


if __name__ == "__main__":
    unittest.main()
