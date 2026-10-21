"""Unit tests for Pillar 2: Orthogonal Subspace Probe and Phase Spectrum Reconstruction."""
import unittest
import numpy as np
import torch
from PIL import Image

from ml.src.models.orthogonal_subspace_probe import (
    OrthogonalSubspaceProjector,
    OrthogonalProbeClassifier,
)
from ml.src.meta_classifier.feature_extractor import (
    phase_spectrum_features,
    to_gray_u8,
)
from ml.scripts.train_orthogonal_probe import (
    compute_binary_auc,
    fit_and_evaluate_probe,
)


class OrthogonalProbeTest(unittest.TestCase):

    def test_orthogonal_projector_strips_top_components(self):
        # Create synthetic data with 1 dominant direction
        np.random.seed(42)
        n, d = 100, 32
        v_dom = np.random.randn(d)
        v_dom = v_dom / np.linalg.norm(v_dom)

        # High variance along v_dom
        data = np.outer(np.random.randn(n) * 10.0, v_dom) + np.random.randn(n, d) * 0.1
        data_tensor = torch.from_numpy(data).float()

        projector = OrthogonalSubspaceProjector(embed_dim=d, top_k_remove=1)
        projector.fit_from_covariance(data_tensor)

        # Test projection on a vector along the dominant direction
        test_vec = torch.from_numpy(v_dom).float().unsqueeze(0)
        projected = projector(test_vec)

        # Energy along dominant vector should be strongly suppressed
        orig_norm = float(torch.norm(test_vec))
        proj_norm = float(torch.norm(projected))
        self.assertLess(proj_norm, orig_norm * 0.1)

    def test_orthogonal_probe_classifier_shapes(self):
        batch_size, embed_dim = 8, 64
        model = OrthogonalProbeClassifier(embed_dim=embed_dim, hidden_dim=32, top_k_remove=2)
        x = torch.randn(batch_size, embed_dim)

        logits, emb = model(x)
        self.assertEqual(logits.shape, (batch_size,))
        self.assertEqual(emb.shape, (batch_size, 32))

        probs = model.predict_proba(x)
        self.assertEqual(probs.shape, (batch_size,))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

    def test_phase_spectrum_features(self):
        img = Image.new("RGB", (256, 256), color=(100, 150, 200))
        # Draw some variation
        arr = np.asarray(img).copy()
        arr[50:100, 50:100] = [220, 30, 40]
        test_img = Image.fromarray(arr)

        kurtosis, hf_ratio = phase_spectrum_features(test_img, size=128)
        self.assertFalse(np.isnan(kurtosis))
        self.assertFalse(np.isnan(hf_ratio))
        self.assertGreater(kurtosis, 0.0)
        self.assertGreater(hf_ratio, 0.0)

    def test_fit_and_evaluate_probe_loop(self):
        np.random.seed(99)
        n_train, n_test, d = 40, 20, 32
        train_clean = np.random.randn(n_train, d).astype(np.float32)
        train_deg = train_clean + np.random.randn(n_train, d).astype(np.float32) * 0.05
        train_labels = np.array([0] * (n_train // 2) + [1] * (n_train // 2))

        test_clean = np.random.randn(n_test, d).astype(np.float32)
        test_labels = np.array([0] * (n_test // 2) + [1] * (n_test // 2))
        teacher_scores = np.random.rand(n_test).astype(np.float32)

        summary = fit_and_evaluate_probe(
            train_clean,
            train_deg,
            train_labels,
            test_clean,
            test_labels,
            teacher_scores,
            embed_dim=d,
            top_k_remove=2,
            epochs=2,
        )
        self.assertIn("teacher_test_auc", summary)
        self.assertIn("student_test_auc", summary)
        self.assertIn("distillation_gap_auc", summary)
        self.assertFalse(np.isnan(summary["distillation_gap_auc"]))


if __name__ == "__main__":
    unittest.main()
