"""Unit tests for native patch extraction, Multi-Instance Learning (MIL) pooling, and localization metrics."""

from __future__ import annotations

import unittest
import numpy as np
from PIL import Image

from data.patch_extractor import (
    compute_texture_energy,
    extract_native_patches,
    extract_candidate_patches,
    top_k_pooling,
    log_sum_exp_pooling,
    noisy_or_pooling,
    patch_mask_iou,
    evaluate_small_manipulations,
    _TORCH_AVAILABLE,
)

if _TORCH_AVAILABLE:
    import torch
    from data.patch_extractor import GatedAttentionMIL, npr_residual


class TestPatchExtractor(unittest.TestCase):

    def test_compute_texture_energy_flat_and_noisy(self):
        flat = np.ones((200, 200), dtype=np.uint8) * 128
        top, left = compute_texture_energy(flat, patch_size=64, stride=32)
        self.assertEqual((top, left), (0, 0))

        # Place high noise in bottom-right corner
        np.random.seed(42)
        noisy = flat.copy()
        noisy[100:180, 100:180] = np.random.randint(0, 255, size=(80, 80), dtype=np.uint8)
        top_n, left_n = compute_texture_energy(noisy, patch_size=64, stride=32)
        self.assertGreaterEqual(top_n, 64)
        self.assertGreaterEqual(left_n, 64)

    def test_extract_native_patches_count_and_size(self):
        img = Image.new("RGB", (400, 300), color=(100, 150, 200))
        patches = extract_native_patches(img, patch_size=128, num_patches=2)
        self.assertEqual(len(patches), 2)
        for p in patches:
            self.assertEqual(p.size, (128, 128))

    def test_extract_native_patches_small_image_padding(self):
        small_img = Image.new("RGB", (64, 64), color=(50, 60, 70))
        patches = extract_native_patches(small_img, patch_size=128, num_patches=2)
        self.assertEqual(len(patches), 2)
        for p in patches:
            self.assertEqual(p.size, (128, 128))

    def test_extract_candidate_patches_budget_and_coverage(self):
        img = Image.new("RGB", (500, 500), color=(128, 128, 128))
        res = extract_candidate_patches(
            img, patch_size=224, stride=112, max_patches=4, ranking_metric="gradient"
        )
        self.assertEqual(len(res.patches), 4)
        self.assertEqual(len(res.boxes), 4)
        self.assertEqual(len(res.energies), 4)
        self.assertGreater(res.coverage_ratio, 0.0)
        self.assertLessEqual(res.coverage_ratio, 1.0)
        self.assertEqual(res.image_size, (500, 500))
        for p in res.patches:
            self.assertEqual(p.size, (224, 224))

    def test_top_k_pooling_numpy_and_torch(self):
        logits = [0.1, 0.9, 0.5, 0.2]
        # top 2 are 0.9 and 0.5 -> average = 0.7
        res_np = top_k_pooling(logits, k=2)
        self.assertAlmostEqual(float(res_np), 0.7, places=5)

        if _TORCH_AVAILABLE:
            t_logits = torch.tensor(logits)
            res_torch = top_k_pooling(t_logits, k=2)
            self.assertAlmostEqual(float(res_torch.item()), 0.7, places=5)

    def test_log_sum_exp_pooling_bounded_by_mean_and_max(self):
        logits = [-1.0, 2.0, 0.5, -0.2]
        res = log_sum_exp_pooling(logits, tau=2.0)
        mean_val = np.mean(logits)
        max_val = np.max(logits)
        self.assertGreaterEqual(res, mean_val - 1e-5)
        self.assertLessEqual(res, max_val + 1e-5)

        if _TORCH_AVAILABLE:
            t_logits = torch.tensor(logits)
            res_torch = log_sum_exp_pooling(t_logits, tau=2.0)
            self.assertAlmostEqual(float(res_torch.item()), float(res), places=5)

    def test_noisy_or_pooling_calibration(self):
        probs = [0.1, 0.1, 0.1, 0.1]
        uncal = noisy_or_pooling(probs, calibrate_for_n=False)
        # 1 - (0.9)^4 = 1 - 0.6561 = 0.3439
        self.assertAlmostEqual(uncal, 0.3439, places=3)

        cal = noisy_or_pooling(probs, calibrate_for_n=True)
        # With n-calibration, equal probabilities return 0.1
        self.assertAlmostEqual(cal, 0.1, places=3)

    def test_patch_mask_iou(self):
        gt_mask = np.zeros((200, 200), dtype=bool)
        gt_mask[50:150, 50:150] = True  # 100x100 box = 10000 pixels

        # Exactly matching detection
        boxes = [(50, 50, 150, 150)]
        scores = [0.8]
        iou_perfect = patch_mask_iou(boxes, scores, gt_mask, threshold=0.5)
        self.assertAlmostEqual(iou_perfect, 1.0, places=5)

        # Disjoint detection
        boxes_disjoint = [(0, 0, 50, 50)]
        iou_zero = patch_mask_iou(boxes_disjoint, scores, gt_mask, threshold=0.5)
        self.assertAlmostEqual(iou_zero, 0.0, places=5)

        # Non-detected due to threshold
        iou_subthreshold = patch_mask_iou(boxes, [0.3], gt_mask, threshold=0.5)
        self.assertAlmostEqual(iou_subthreshold, 0.0, places=5)

    def test_evaluate_small_manipulations(self):
        records = [
            # 2% area edit (micro)
            {
                "gt_mask": np.pad(np.ones((10, 20)), ((0, 90), (0, 80))),
                "score": 0.8,
            },
            # 10% area edit (localized)
            {
                "gt_mask": np.pad(np.ones((20, 50)), ((0, 80), (0, 50))),
                "score": 0.4,
            },
            # 50% area edit (extensive)
            {
                "gt_mask": np.pad(np.ones((50, 100)), ((0, 50), (0, 0))),
                "score": 0.9,
            },
        ]
        res = evaluate_small_manipulations(records, threshold=0.5)
        self.assertEqual(res["1%-5%"]["total_samples"], 1)
        self.assertEqual(res["1%-5%"]["detected_samples"], 1)
        self.assertEqual(res["1%-5%"]["tpr"], 1.0)

        self.assertEqual(res["5%-20%"]["total_samples"], 1)
        self.assertEqual(res["5%-20%"]["detected_samples"], 0)
        self.assertEqual(res["5%-20%"]["tpr"], 0.0)

        self.assertEqual(res["20%-100%"]["total_samples"], 1)
        self.assertEqual(res["20%-100%"]["detected_samples"], 1)
        self.assertEqual(res["20%-100%"]["tpr"], 1.0)

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch required for GatedAttentionMIL")
    def test_gated_attention_mil(self):
        mil = GatedAttentionMIL(in_features=32, hidden_dim=16)
        embeddings = torch.randn(2, 5, 32)
        logits = torch.randn(2, 5, 1)
        bag_logit, weights = mil(embeddings, logits)
        self.assertEqual(bag_logit.shape, (2, 1))
        self.assertEqual(weights.shape, (2, 5))
        # Weights must sum to 1 across patches
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
