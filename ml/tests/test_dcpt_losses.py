"""Tests for the DCPT degradation-consistency losses.

The point of these tests is the BINARY KL term. `F.kl_div(log_q, p)` on a
single probability evaluates only the positive-class term
``p * log(p / q)`` and silently drops the negative-class term
``(1 - p) * log((1 - p) / (1 - q))``.

For a real image (p_clean ~ 0.01) against a degraded view the model is
confident about (p_deg ~ 0.40), that positive-only term is NEGATIVE:
``0.01 * ln(0.01 / 0.40) = -0.0369``. A loss that goes negative when the
two views disagree most is not a consistency constraint.
"""
import unittest

import torch

from training.dcpt_losses import (
    dcpt_consistency_loss,
    dcpt_feature_consistency_loss,
    dcpt_total_loss,
)


def _logit(p: float) -> float:
    import math
    return math.log(p / (1.0 - p))


class DcptConsistencyLossTest(unittest.TestCase):
    def test_it_is_zero_for_identical_views(self):
        logits = torch.tensor([[_logit(0.2)], [_logit(0.8)]])
        self.assertAlmostEqual(float(dcpt_consistency_loss(logits, logits)), 0.0, places=6)

    def test_it_is_symmetric(self):
        clean = torch.tensor([[_logit(0.01)]])
        degraded = torch.tensor([[_logit(0.40)]])
        self.assertAlmostEqual(
            float(dcpt_consistency_loss(clean, degraded)),
            float(dcpt_consistency_loss(degraded, clean)),
            places=6,
        )

    def test_it_matches_the_explicit_binary_divergence(self):
        """The loss is the SYMMETRIC binary KL, so both directions count."""
        import math
        p, q = 0.01, 0.40
        kl_pq = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
        kl_qp = q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))
        expected = 0.5 * (kl_pq + kl_qp)
        loss = dcpt_consistency_loss(
            torch.tensor([[_logit(p)]]), torch.tensor([[_logit(q)]])
        )
        self.assertAlmostEqual(float(loss), expected, places=6)

    def test_it_includes_the_negative_class_term(self):
        """The dropped term is the dominant one on a real image.

        ``F.kl_div`` on the positive class alone is NEGATIVE here, which is
        the signature of the bug this module exists to prevent.
        """
        import torch.nn.functional as F
        p, q = 0.01, 0.40
        clean = torch.tensor([[_logit(p)]])
        degraded = torch.tensor([[_logit(q)]])
        positive_only = float(F.kl_div(torch.log(torch.sigmoid(degraded) + 1e-8), torch.sigmoid(clean)))
        full = float(dcpt_consistency_loss(clean, degraded))
        self.assertLess(positive_only, 0.0, "the positive-only term should be negative here")
        self.assertGreater(full, 0.0)
        self.assertGreater(full, abs(positive_only))

    def test_it_grows_with_disagreement(self):
        near = float(dcpt_consistency_loss(
            torch.tensor([[_logit(0.40)]]), torch.tensor([[_logit(0.45)]])))
        far = float(dcpt_consistency_loss(
            torch.tensor([[_logit(0.40)]]), torch.tensor([[_logit(0.95)]])))
        self.assertGreater(far, near)

    def test_it_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            dcpt_consistency_loss(torch.zeros(1, 1), torch.zeros(1, 2))


class DcptFeatureConsistencyLossTest(unittest.TestCase):
    def test_it_is_zero_for_identical_embeddings(self):
        z = torch.randn(4, 16)
        self.assertAlmostEqual(float(dcpt_feature_consistency_loss(z, z)), 0.0, places=6)

    def test_it_is_two_for_opposite_embeddings(self):
        z = torch.randn(4, 16)
        self.assertAlmostEqual(float(dcpt_feature_consistency_loss(z, -z)), 2.0, places=5)

    def test_it_is_one_for_orthogonal_embeddings(self):
        z = torch.tensor([[1.0, 0.0]])
        w = torch.tensor([[0.0, 1.0]])
        self.assertAlmostEqual(float(dcpt_feature_consistency_loss(z, w)), 1.0, places=6)

    def test_it_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            dcpt_feature_consistency_loss(torch.zeros(2, 4), torch.zeros(2, 5))


class DcptTotalLossTest(unittest.TestCase):
    def test_it_combines_both_terms(self):
        clean = torch.tensor([[_logit(0.2)]])
        degraded = torch.tensor([[_logit(0.6)]])
        zc = torch.randn(1, 8)
        zd = torch.randn(1, 8)
        total = float(dcpt_total_loss(clean, degraded, zc, zd))
        parts = float(dcpt_consistency_loss(clean, degraded)) + float(
            dcpt_feature_consistency_loss(zc, zd))
        self.assertAlmostEqual(total, parts, places=6)

    def test_it_rejects_a_half_supplied_embedding(self):
        with self.assertRaises(ValueError):
            dcpt_total_loss(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 4), None)

    def test_the_feature_term_is_optional(self):
        clean = torch.tensor([[_logit(0.2)]])
        degraded = torch.tensor([[_logit(0.6)]])
        total = float(dcpt_total_loss(clean, degraded))
        self.assertAlmostEqual(total, float(dcpt_consistency_loss(clean, degraded)), places=6)


if __name__ == "__main__":
    unittest.main()
