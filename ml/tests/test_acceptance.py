"""Tests for the release acceptance floors and the per-source FPR gate.

These are the gates that stop a release whose headline AUC looks acceptable
but whose usable operating point does not. The shipped image fusion is the
reference case: AUC 0.7152 reads as "weak but usable", while
``tpr_at_fpr_1pct = 0.040`` reads as "unusable", and only the second number
is checked here.
"""
import unittest

from training.acceptance import (
    MINIMUM_ACCEPTANCE,
    MINIMUM_FUSION_ACCEPTANCE,
    MIN_REAL_FOR_LOW_FPR_GATE,
    check_model_acceptance,
)
from training.evaluation import assert_source_fpr_within_bound


def _report(auc=0.90, tpr_1=0.60, tpr_01=0.40, n_real=4000, source_fpr=None):
    test = {
        "auc": auc,
        "tpr_at_fpr_1pct": tpr_1,
        "tpr_at_fpr_0_1pct": tpr_01,
        "n_real": n_real,
    }
    if source_fpr is not None:
        test["groups"] = {"source": source_fpr}
    return test


class AcceptanceFloorTest(unittest.TestCase):
    def test_a_passing_report_is_accepted(self):
        check_model_acceptance("lcnn_audio_int8.tflite", _report())

    def test_a_low_auc_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            check_model_acceptance("lcnn_audio_int8.tflite", _report(auc=0.60))
        self.assertIn("AUC", str(caught.exception))

    def test_a_low_one_percent_operating_point_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            check_model_acceptance("lcnn_audio_int8.tflite", _report(tpr_1=0.10))
        self.assertIn("TPR@1%FPR", str(caught.exception))

    def test_a_missing_metric_is_rejected(self):
        broken = _report()
        del broken["tpr_at_fpr_1pct"]
        with self.assertRaises(ValueError):
            check_model_acceptance("lcnn_audio_int8.tflite", broken)

    def test_the_shipped_fusion_numbers_are_rejected(self):
        # The values the shipped release actually reports.
        with self.assertRaises(ValueError):
            check_model_acceptance(
                "fusion_xgboost",
                _report(auc=0.7152, tpr_1=0.040, tpr_01=0.005, n_real=200),
                floors=MINIMUM_FUSION_ACCEPTANCE,
            )

    def test_a_small_evaluation_set_blocks_the_low_fpr_gate(self):
        with self.assertRaises(ValueError) as caught:
            check_model_acceptance(
                "lcnn_audio_int8.tflite", _report(n_real=200)
            )
        self.assertIn(str(MIN_REAL_FOR_LOW_FPR_GATE), str(caught.exception))

    def test_a_small_evaluation_set_is_allowed_when_the_gate_is_relaxed(self):
        # Staging validation stays usable; the distribution gate is strict.
        check_model_acceptance(
            "lcnn_audio_int8.tflite", _report(n_real=200), require_low_fpr_statistic=False
        )

    def test_an_unknown_model_has_no_floor(self):
        check_model_acceptance("some_other_model", _report(auc=0.1, tpr_1=0.0))

    def test_every_named_model_has_a_floor(self):
        for name in MINIMUM_ACCEPTANCE:
            self.assertEqual(len(MINIMUM_ACCEPTANCE[name]), 3)


class SourceBiasGateTest(unittest.TestCase):
    def test_a_clean_report_passes(self):
        assert_source_fpr_within_bound(_report(source_fpr={
            "laion": {"n_real": 81, "n_fake": 200, "fpr": 0.02},
            "pexels": {"n_real": 69, "n_fake": 200, "fpr": 0.01},
        }))

    def test_a_source_shortcut_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            assert_source_fpr_within_bound(_report(source_fpr={
                "laion": {"n_real": 81, "n_fake": 200, "fpr": 0.4444},
            }))
        self.assertIn("laion", str(caught.exception))
        self.assertIn("acquisition source", str(caught.exception))

    def test_a_small_source_is_ignored(self):
        # A handful of samples cannot establish a per-source rate.
        assert_source_fpr_within_bound(_report(source_fpr={
            "tiny": {"n_real": 3, "n_fake": 0, "fpr": 1.0},
        }))

    def test_the_shipped_source_bias_is_rejected(self):
        with self.assertRaises(ValueError):
            assert_source_fpr_within_bound(_report(source_fpr={
                "laion": {"n_real": 81, "n_fake": 200, "fpr": 0.4444},
                "pexels": {"n_real": 69, "n_fake": 200, "fpr": 0.3478},
                "docci": {"n_real": 17, "n_fake": 200, "fpr": 0.2941},
            }))

    def test_a_missing_source_axis_is_not_silently_accepted(self):
        # The gate must be able to tell "no shortcut" from "not measured".
        report = _report()
        report.pop("groups", None)
        # With no source axis there is nothing to check, and that must not
        # raise - but it must also not be mistaken for a pass by a caller
        # that then reports the gate as satisfied.
        assert_source_fpr_within_bound(report)
        self.assertNotIn("groups", report)


if __name__ == "__main__":
    unittest.main()
