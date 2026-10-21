package com.onndetect

import com.onndetect.domain.models.DetectionVerdict
import com.onndetect.domain.pipeline.CascadePolicy
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Cross-platform contract for the dual-engine cascade.
 *
 * These cases are the SAME ones asserted by
 * `ml/tests/test_dual_engine_contract.py::PolicyBehaviourTest`. The Python
 * policy in `meta_classifier/dual_engine.py` and the Kotlin policy in
 * `domain/pipeline/CascadePolicy.kt` must agree exactly: a divergence would
 * mean the desktop measurements describe a different system from the one
 * that ships.
 *
 * Run with: gradlew.bat testReleaseUnitTest
 */
class CascadePolicyTest {

    private fun policy() = CascadePolicy(
        engineAThreshold = 0.5f,
        engineCThreshold = 0.5f,
        escalateBelow = 0.4f,
        escalateAbove = 0.6f,
        coverage = 0.8f,
        agreementRate = 0.9f,
        releaseId = "test",
    )

    @Test
    fun agreementInsideTheBandResolves() {
        val decision = policy().decide(0.55f, 0.55f)
        assertEquals(DetectionVerdict.SYNTHETIC, decision.verdict)
        assertTrue(decision.escalated)
    }

    @Test
    fun disagreementInsideTheBandAbstains() {
        val decision = policy().decide(0.8f, 0.45f)
        assertEquals(DetectionVerdict.INCONCLUSIVE, decision.verdict)
        assertTrue(decision.escalated)
    }

    @Test
    fun confidentFusionDecidesWithoutEscalation() {
        // Outside the ambiguity band the FUSION decides. The fusion is the
        // stronger signal (sealed test AUC 0.7152 vs Engine C 0.6871), so
        // Engine C must not override it there.
        val decision = policy().decide(0.2f, 0.9f)
        assertEquals(DetectionVerdict.AUTHENTIC, decision.verdict)
        assertFalse(decision.escalated)
    }

    @Test
    fun confidentFusionDecidesSyntheticWithoutEscalation() {
        // Fusion says fake, Engine C says authentic, and Engine C is outside
        // the ambiguity band. The fusion is the stronger signal, so it
        // decides. This is the case the old code got backwards.
        val decision = policy().decide(0.9f, 0.1f)
        assertEquals(DetectionVerdict.SYNTHETIC, decision.verdict)
        assertFalse(decision.escalated)
    }

    @Test
    fun missingEngineCLetsEngineADecide() {
        val decision = policy().decide(0.8f, null)
        assertEquals(DetectionVerdict.SYNTHETIC, decision.verdict)
        assertFalse(decision.escalated)
    }

    @Test
    fun theReportedScoreIsAlwaysTheFusionScore() {
        // The threshold the pipeline reports must belong to the same engine
        // as the score, or the verdict can contradict the score. The cascade
        // reports the fusion score, so the matching threshold is
        // engineAThreshold.
        val p = policy()
        val decision = p.decide(0.8f, 0.55f)
        assertEquals(0.8f, decision.score, 1e-6f)
    }

    @Test
    fun anInvertedBandIsRejected() {
        val failure = runCatching {
            CascadePolicy(
                engineAThreshold = 0.5f, engineCThreshold = 0.5f,
                escalateBelow = 0.8f, escalateAbove = 0.2f,
                coverage = 1.0f, agreementRate = 0.9f, releaseId = "test",
            )
        }
        assertTrue(failure.isFailure)
    }

    @Test
    fun anOutOfRangeThresholdIsRejected() {
        val failure = runCatching {
            CascadePolicy(
                engineAThreshold = 1.5f, engineCThreshold = 0.5f,
                escalateBelow = 0.4f, escalateAbove = 0.6f,
                coverage = 1.0f, agreementRate = 0.9f, releaseId = "test",
            )
        }
        assertTrue(failure.isFailure)
    }

    @Test
    fun theBandEdgesAreInclusive() {
        // A score exactly on the band edge must be treated as ambiguous.
        val p = policy()
        assertTrue(p.decide(0.55f, 0.4f).escalated)
        assertTrue(p.decide(0.55f, 0.6f).escalated)
        assertFalse(p.decide(0.55f, 0.3999f).escalated)
        assertFalse(p.decide(0.55f, 0.6001f).escalated)
    }

    @Test
    fun theDecisionStatusIsInconclusiveOnlyWhenTheVerdictIs() {
        val abstained = policy().decide(0.8f, 0.45f)
        assertEquals(
            com.onndetect.domain.models.AnalysisStatus.INCONCLUSIVE,
            abstained.status,
        )
        val resolved = policy().decide(0.55f, 0.55f)
        assertEquals(
            com.onndetect.domain.models.AnalysisStatus.SUCCESS,
            resolved.status,
        )
    }
}
