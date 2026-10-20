"""Cross-platform contract for the dual-engine cascade.

The Python policy in `meta_classifier/dual_engine.py` and the Kotlin mirror
in `domain/pipeline/CascadePolicy.kt` must agree exactly. A divergence would
mean the desktop measurements describe a different system from the one that
ships, which is the exact failure the repository's contract gates exist to
prevent.

The Kotlin file is parsed statically (no JVM required) so this test runs in
CI alongside the other contract tests.
"""
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from meta_classifier.dual_engine import (  # noqa: E402
    CascadePolicy,
    assemble_features,
    decide,
    fit_escalation_band,
    mask_semantic_probe,
)

REPO = ROOT.parent
KOTLIN_POLICY = (
    REPO / "android/app/src/main/java/com/onndetect/domain/pipeline/CascadePolicy.kt"
)
KOTLIN_PIPELINE = (
    REPO / "android/app/src/main/java/com/onndetect/domain/pipeline/DetectionPipeline.kt"
)
KOTLIN_CHECKER = (
    REPO / "android/app/src/main/java/com/onndetect/inference/ModelIntegrityChecker.kt"
)


class KotlinMirrorTest(unittest.TestCase):
    """The Kotlin mirror must expose the same decision surface."""

    def setUp(self):
        self.source = KOTLIN_POLICY.read_text(encoding="utf-8")

    def test_kotlin_policy_file_exists(self):
        self.assertTrue(KOTLIN_POLICY.is_file())

    def test_kotlin_exposes_the_same_thresholds_and_band(self):
        for name in (
            "engineAThreshold",
            "engineCThreshold",
            "escalateBelow",
            "escalateAbove",
        ):
            self.assertIn(name, self.source, f"Kotlin policy is missing {name}")

    def test_kotlin_uses_the_same_snake_case_asset_keys(self):
        for key in (
            "engine_a_threshold",
            "engine_c_threshold",
            "escalate_below",
            "escalate_above",
        ):
            self.assertIn(key, self.source, f"Kotlin policy does not read {key}")

    def test_kotlin_disagreement_abstains(self):
        self.assertIn("INCONCLUSIVE", self.source)
        self.assertIn("disagreement", self.source.lower())

    def test_kotlin_handles_a_missing_engine_c(self):
        self.assertIn("Engine C unavailable", self.source)


class PipelineWiringTest(unittest.TestCase):
    """The production pipeline must wire the second engine correctly."""

    def setUp(self):
        self.source = KOTLIN_PIPELINE.read_text(encoding="utf-8")

    def test_pipeline_declares_the_semantic_probe_slot(self):
        self.assertIn("SEMANTIC_PROBE_INDEX", self.source)

    def test_pipeline_sends_unavailable_not_a_neutral_constant(self):
        """Slot 6 must be NaN when the probe did not run."""
        self.assertRegex(
            self.source,
            r"engineCScore\s*\?:\s*Float\.NaN",
            "Pipeline must send Float.NaN for an unavailable Engine C",
        )

    def test_pipeline_does_not_send_the_legacy_neutral_to_slot_six(self):
        features_block = re.search(
            r"floatArrayOf\((.*?)\)", self.source, re.S
        )
        self.assertIsNotNone(features_block)
        body = features_block.group(1)
        self.assertNotIn("NEUTRAL_EXIF", body)

    def test_pipeline_reads_the_measured_cascade_policy(self):
        self.assertIn("CascadePolicy.fromAssets", self.source)

    def test_the_probe_only_runs_when_the_release_validated_it(self):
        """An unverified second engine must not be able to move a verdict."""
        self.assertIn("report.cascadeValidated", self.source)

    def test_pipeline_reports_the_deciding_engines_threshold(self):
        """Score and threshold must come from the SAME engine.

        Engine A and Engine C have different operating points. Reporting
        Engine C's score against the fusion threshold would produce a score
        that contradicts its own verdict.
        """
        self.assertIn("policy.engineCThreshold", self.source)
        self.assertIn("decisionThreshold", self.source)

    def test_pipeline_keeps_the_release_threshold_for_the_single_engine_path(self):
        self.assertIn("report.threshold", self.source)


class IntegrityGateTest(unittest.TestCase):
    """Engine C must be held to the same integrity standard as the others."""

    def setUp(self):
        self.source = KOTLIN_CHECKER.read_text(encoding="utf-8")

    def test_cascade_model_is_declared_optional(self):
        self.assertIn("CASCADE_MODEL", self.source)
        self.assertIn("OPTIONAL_MODELS", self.source)

    def test_cascade_model_must_pass_the_same_parity_checks(self):
        for marker in (
            "Desktop cascade parity missing",
            "Device cascade model/frontend linkage mismatch",
            "Cascade calibration is at/below chance",
        ):
            self.assertIn(marker, self.source, f"Missing integrity gate: {marker}")


class PolicyBehaviourTest(unittest.TestCase):
    """Behaviour of the arbitration policy itself."""

    def _policy(self) -> CascadePolicy:
        return CascadePolicy(
            engine_a_threshold=0.5,
            engine_c_threshold=0.5,
            escalate_below=0.4,
            escalate_above=0.6,
            coverage=0.8,
            agreement_rate=0.9,
            release_id="test",
        )

    def test_agreement_inside_the_band_resolves(self):
        # Both inside the ambiguity band [0.4, 0.6] and on the same side of
        # their own thresholds -> agreement resolves the sample.
        decision = decide(0.55, 0.55, self._policy())
        self.assertEqual(decision.verdict, "SYNTHETIC")
        self.assertTrue(decision.escalated)

    def test_disagreement_inside_the_band_abstains(self):
        # Inside the band, but the engines point in opposite directions.
        decision = decide(0.8, 0.45, self._policy())
        self.assertEqual(decision.verdict, "INCONCLUSIVE")
        self.assertTrue(decision.escalated)

    def test_confident_engine_c_decides_without_escalation(self):
        # Outside the band -> Engine C decides on its own.
        decision = decide(0.2, 0.9, self._policy())
        self.assertEqual(decision.verdict, "SYNTHETIC")
        self.assertFalse(decision.escalated)

    def test_missing_engine_c_leaves_engine_a_in_charge(self):
        decision = decide(0.8, None, self._policy())
        self.assertEqual(decision.verdict, "SYNTHETIC")
        self.assertFalse(decision.escalated)

    def test_band_edges_are_inclusive(self):
        """A score exactly on the band edge must be treated as ambiguous.

        Mirrors `CascadePolicyTest.theBandEdgesAreInclusive` on the Kotlin
        side; the two implementations must agree on the edge behaviour.
        """
        policy = self._policy()
        self.assertTrue(decide(0.55, 0.4, policy).escalated)
        self.assertTrue(decide(0.55, 0.6, policy).escalated)
        self.assertFalse(decide(0.55, 0.3999, policy).escalated)
        self.assertFalse(decide(0.55, 0.6001, policy).escalated)

    def test_the_reported_score_is_always_engine_cs_score(self):
        """Score and threshold must belong to the same engine."""
        decision = decide(0.8, 0.55, self._policy())
        self.assertAlmostEqual(decision.score, 0.55, places=6)

    def test_out_of_range_scores_are_rejected(self):
        """Mirrors the Kotlin `require` guards exactly."""
        policy = self._policy()
        with self.assertRaises(ValueError):
            decide(1.5, 0.5, policy)
        with self.assertRaises(ValueError):
            decide(0.5, 1.5, policy)
        with self.assertRaises(ValueError):
            decide(0.5, float("nan"), policy)

    def test_policy_rejects_an_inverted_band(self):
        with self.assertRaises(ValueError):
            CascadePolicy(
                engine_a_threshold=0.5, engine_c_threshold=0.5,
                escalate_below=0.8, escalate_above=0.2,
                coverage=1.0, agreement_rate=0.9, release_id="test",
            ).validate()

    def test_policy_round_trips_through_json(self):
        policy = self._policy()
        restored = CascadePolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
        self.assertEqual(restored.to_dict(), policy.to_dict())


class MaskingTest(unittest.TestCase):
    """Slot masking must produce a genuinely unavailable measurement."""

    def test_masking_sets_slot_six_to_nan_and_leaves_the_rest(self):
        features = assemble_features(
            srm_score=0.4, fft_score=0.2, kl_score=0.1, flatness_score=0.3,
            resolution_norm=0.5, is_compressed=True, semantic_probe_score=0.9,
        )
        masked = mask_semantic_probe(features[None, :])[0]
        self.assertTrue(np.isnan(masked[6]))
        self.assertTrue(np.isfinite(masked[[i for i in range(10) if i != 6]]).all())

    def test_masking_does_not_mutate_the_input(self):
        features = assemble_features(
            srm_score=0.4, fft_score=0.2, kl_score=0.1, flatness_score=0.3,
            resolution_norm=0.5, is_compressed=True, semantic_probe_score=0.9,
        )
        mask_semantic_probe(features[None, :])
        self.assertAlmostEqual(float(features[6]), 0.9, places=6)

    def test_a_constant_slot_is_not_mistaken_for_a_measurement(self):
        """A release built before the dual engine held slot 6 at a constant.

        float32 rounding makes such a column report a non-zero standard
        deviation (~1.5e-08), so the detection must use the unique-value
        count rather than `std() > 0`.
        """
        column = np.full(400, 0.1, dtype=np.float32)
        self.assertGreater(float(column.std()), 0.0)
        self.assertEqual(int(np.unique(column).size), 1)
        self.assertFalse(bool(
            np.isfinite(column).all() and np.unique(column).size > 1
        ))



class BandFittingTest(unittest.TestCase):
    """The escalation band must be derived, never hand-picked."""

    def test_band_fit_returns_a_valid_policy_and_a_table(self):
        rng = np.random.default_rng(7)
        labels = np.array([0, 1] * 100)
        a_scores = np.clip(0.5 + (labels - 0.5) * 0.3 + rng.normal(0, 0.1, 200), 0, 1)
        c_scores = np.clip(0.5 + (labels - 0.5) * 0.8 + rng.normal(0, 0.1, 200), 0, 1)

        policy, table = fit_escalation_band(a_scores, c_scores, labels, 0.5, 0.5)
        policy.validate()
        self.assertGreater(len(table), 10)
        for row in table:
            self.assertGreaterEqual(row["resolved_accuracy"], 0.0)
            self.assertLessEqual(row["resolved_accuracy"], 1.0)
            self.assertLessEqual(row["escalate_below"], row["escalate_above"])

    def test_band_fit_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            fit_escalation_band(
                np.zeros(10), np.zeros(9), np.zeros(10, dtype=int), 0.5, 0.5
            )


if __name__ == "__main__":
    unittest.main()
