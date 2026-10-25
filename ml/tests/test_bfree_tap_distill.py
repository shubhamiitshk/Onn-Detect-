"""Unit tests for B-Free Twins, SigLIP-TAP, Dual-SSL Distillation, and AVFF Lip-Audio Synchrony."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data.bfree_paired_corpus import (
    BFreeSyntheticTwinBuilder,
    BFreeTwinPair,
    build_patch_mask_14x14,
    assert_parent_disjoint_splits,
)
from train_bfree_tap_and_ssl_distill import (
    fit_multiwindow_audio_aggregator,
    compute_avff_lip_audio_sync_score,
    compute_avff_lip_audio_sync_correlation,
    DualSSLToLCNNDistiller,
    _TORCH_AVAILABLE,
)

if _TORCH_AVAILABLE:
    import torch
    from train_bfree_tap_and_ssl_distill import SiglipTAPHead, compute_tap_bfree_loss


class TestBFreeTapAndDistill(unittest.TestCase):

    def setUp(self):
        rng = np.random.RandomState(42)
        arr = rng.randint(20, 235, size=(128, 128, 3), dtype=np.uint8)
        self.test_img = Image.fromarray(arr, mode="RGB")

    def test_bfree_synthetic_twin_builder_full_reconstruction(self):
        builder = BFreeSyntheticTwinBuilder(seed=42)
        pair = builder.generate_pair(self.test_img, real_parent_id="parent_01", mode="full_reconstruction")
        self.assertIsInstance(pair, BFreeTwinPair)
        self.assertEqual(pair.real_parent_id, "parent_01")
        self.assertEqual(pair.manipulation_type, "full_reconstruction")
        self.assertEqual(pair.patch_mask_14x14.shape, (196,))
        self.assertEqual(pair.edited_area_ratio, 1.0)
        self.assertEqual(pair.fake_twin_image.size, self.test_img.size)

    def test_bfree_synthetic_twin_builder_local_inpaint(self):
        builder = BFreeSyntheticTwinBuilder(seed=42)
        pair = builder.generate_pair(self.test_img, real_parent_id="parent_02", mode="local_inpaint")
        self.assertEqual(pair.manipulation_type, "local_inpaint")
        self.assertEqual(pair.patch_mask_14x14.shape, (196,))
        self.assertGreater(pair.edited_area_ratio, 0.0)
        self.assertLess(pair.edited_area_ratio, 0.5)

    def test_assert_parent_disjoint_splits(self):
        # Clean splits pass
        assert_parent_disjoint_splits([
            {"real_parent_id": "p1", "split": "train"},
            {"real_parent_id": "p1", "split": "train"},
            {"real_parent_id": "p2", "split": "val"},
            {"real_parent_id": "p3", "split": "test"},
        ])

        # Leaking split raises ValueError
        with self.assertRaises(ValueError):
            assert_parent_disjoint_splits([
                {"real_parent_id": "p_leaked", "split": "train"},
                {"real_parent_id": "p_leaked", "split": "test"},
            ])

    def test_fit_multiwindow_audio_aggregator(self):
        scores = [
            [0.1, 0.15, 0.08] if y == 0 else [0.85, 0.70, 0.90]
            for y in ([0] * 50 + [1] * 50)
        ]
        labels = [0] * 50 + [1] * 50
        res = fit_multiwindow_audio_aggregator(scores, labels, target_fpr=0.01)
        self.assertEqual(res["schema_version"], "audio-multiwindow-v1")
        self.assertEqual(res["windows_per_file"], 3)
        self.assertGreater(res["w_max"], 0.0)
        self.assertTrue(0.0 < res["threshold_at_eer"] < 1.0)

    def test_compute_avff_lip_audio_sync_score(self):
        t = np.linspace(0, 4 * np.pi, 40)
        mar_sync = 0.3 + 0.15 * np.sin(t)
        aud_sync = 0.5 + 0.25 * np.sin(t + 0.05)
        aud_desync = 0.5 + 0.25 * np.cos(3.7 * t)

        score_sync = compute_avff_lip_audio_sync_score(mar_sync, aud_sync)
        score_desync = compute_avff_lip_audio_sync_score(mar_sync, aud_desync)

        # Synchronized speech has low anomaly; desynchronized has high anomaly
        self.assertLess(score_sync, 0.20)
        self.assertGreater(score_desync, 0.60)

        # Slot 8 polarity test: correlation is inverted (authentic speech has high correlation)
        corr_sync = compute_avff_lip_audio_sync_correlation(mar_sync, aud_sync)
        corr_desync = compute_avff_lip_audio_sync_correlation(mar_sync, aud_desync)
        self.assertGreater(corr_sync, 0.80)
        self.assertLess(corr_desync, 0.40)
        self.assertAlmostEqual(corr_sync, 1.0 - score_sync, places=5)

    def test_dual_ssl_distiller_numpy_and_torch(self):
        distiller = DualSSLToLCNNDistiller(alpha=0.5, temperature=2.0)
        s_log = np.array([[1.0], [-1.5]])
        y_lbl = np.array([[1.0], [0.0]])
        w_log = np.array([[1.5], [-1.2]])
        h_log = np.array([[1.2], [-1.6]])

        loss_np, metrics_np = distiller.compute_distillation_loss(s_log, y_lbl, w_log, h_log)
        self.assertGreater(loss_np, 0.0)
        self.assertIn("bce", metrics_np)
        self.assertIn("teacher_kl", metrics_np)

        if _TORCH_AVAILABLE:
            s_t = torch.tensor(s_log, dtype=torch.float32)
            y_t = torch.tensor(y_lbl, dtype=torch.float32)
            w_t = torch.tensor(w_log, dtype=torch.float32)
            h_t = torch.tensor(h_log, dtype=torch.float32)
            loss_t, metrics_t = distiller.compute_distillation_loss(s_t, y_t, w_t, h_t)
            self.assertAlmostEqual(float(loss_t.item()), float(loss_np), places=4)

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch required for SiglipTAPHead")
    def test_siglip_tap_head_forward_and_loss(self):
        tap = SiglipTAPHead(embed_dim=768, num_heads=12, mlp_dim=3072)
        self.assertTrue(hasattr(tap, "export_into_siglip_head"))
        clean_tokens = torch.randn(2, 196, 768)
        degraded_tokens = clean_tokens + 0.1 * torch.randn(2, 196, 768)
        labels = torch.tensor([1.0, 0.0])
        patch_masks = torch.zeros(2, 196)
        patch_masks[0, 20:50] = 1.0  # Simulated inpaint mask

        logits, feat, attn = tap.forward_with_attention(clean_tokens)
        self.assertEqual(logits.shape, (2, 1))
        self.assertEqual(feat.shape, (2, 768))
        self.assertEqual(attn.shape, (2, 196))
        # Attention weights across the 196 tokens must sum to 1.0
        self.assertTrue(torch.allclose(attn.sum(dim=-1), torch.ones(2), atol=1e-5))

        # Joint TAP B-Free + DCPT loss
        total_loss, loss_dict = compute_tap_bfree_loss(
            tap, clean_tokens, degraded_tokens, labels, patch_masks_14x14=patch_masks
        )
        self.assertGreater(float(total_loss.item()), 0.0)
        self.assertIn("dcpt_kl", loss_dict)
        self.assertIn("dcpt_cos", loss_dict)
        self.assertIn("tap_attn_kl", loss_dict)

    def test_multiwindow_aggregator_empty_and_short_windows(self):
        # Test empty list of window scores handled gracefully without IndexError
        empty_scores = [[], [0.1, 0.2], []]
        labels = [0, 1, 0]
        result = fit_multiwindow_audio_aggregator(empty_scores, labels)
        self.assertIn("threshold_at_eer", result)
        self.assertIn("threshold_at_fpr_1pct", result)

    @unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch required for SiglipTAPHead")
    def test_train_siglip_tap_head_execution(self):
        import tempfile
        from ml.scripts.train_bfree_tap_and_ssl_distill import train_siglip_tap_head
        with tempfile.TemporaryDirectory() as td:
            ckpt_path = Path(td) / "test_tap_head.pt"
            tap = train_siglip_tap_head(
                features_npz=Path(td) / "non_existent.npz",
                output_ckpt=ckpt_path,
                epochs=1,
                batch_size=8,
            )
            self.assertTrue(ckpt_path.is_file())
            self.assertIsNotNone(tap)


if __name__ == "__main__":
    unittest.main()
