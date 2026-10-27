package com.onndetect.detection.video

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

/**
 * Adversarial stress harness for [VideoScorePolicy].
 * Specifically verifies truthfulness contracts, IEEE 754 NaN propagation,
 * out-of-bounds containment, and multi-modal escalation logic.
 */
class VideoScorePolicyAdversarialTest {

    @Test
    fun visualNanAlwaysReturnsNanRegardlessOfOtherModalities() {
        val thresholds = listOf(0.0f, 0.355f, 0.50f, 0.5782f, 0.99f, 1.0f)
        val audioScores = listOf(Float.NaN, 0.0f, 0.355f, 0.50f, 0.99f, 1.0f, Float.POSITIVE_INFINITY)
        val temporalScores = listOf(Float.NaN, 0.0f, 0.30f, 0.70f, 1.0f, Float.POSITIVE_INFINITY)
        val lipSyncAnomalies = listOf(Float.NaN, 0.0f, 0.50f, 1.0f, Float.POSITIVE_INFINITY)
        val validatedFlags = listOf(true, false)

        for (vThresh in thresholds) {
            for (audio in audioScores) {
                for (temp in temporalScores) {
                    for (lip in lipSyncAnomalies) {
                        for (validated in validatedFlags) {
                            val res = VideoScorePolicy.combineMultimodalScore(
                                visualScore = Float.NaN,
                                visualThreshold = vThresh,
                                audioScore = audio,
                                audioValidatedSuccess = validated,
                                temporalScore = temp,
                                lipSyncAnomaly = lip,
                            )
                            assertTrue(
                                "visual NaN must propagate to return NaN! Got: $res with audio=$audio temp=$temp lip=$lip",
                                res.isNaN()
                            )
                        }
                    }
                }
            }
        }
    }

    @Test
    fun visualInfiniteOrOutOfRangeReturnsNan() {
        val invalidVisuals = listOf(
            Float.NEGATIVE_INFINITY,
            -100.0f,
            -1.0f,
            -0.00001f,
            1.00001f,
            2.0f,
            100.0f,
            Float.POSITIVE_INFINITY
        )
        for (invalid in invalidVisuals) {
            val res = VideoScorePolicy.combineMultimodalScore(
                visualScore = invalid,
                visualThreshold = 0.50f,
                audioScore = 0.95f,
                audioValidatedSuccess = true,
                temporalScore = 0.95f,
                lipSyncAnomaly = 0.95f,
            )
            assertTrue("Visual score $invalid must yield NaN, got $res", res.isNaN())
        }
    }

    @Test
    fun audioNanWithValidVisualSkipsAudioEscalationWithoutError() {
        val visualScores = listOf(0.0f, 0.15f, 0.40f, 0.5782f, 0.85f, 1.0f)
        for (visual in visualScores) {
            val scoreWithoutAudio = VideoScorePolicy.combineMultimodalScore(
                visualScore = visual,
                visualThreshold = 0.50f,
                audioScore = Float.NaN,
                audioValidatedSuccess = true,
                temporalScore = Float.NaN,
                lipSyncAnomaly = 0.0f,
            )
            assertEquals("Audio NaN must not alter visual score", visual, scoreWithoutAudio, 1e-6f)
        }
    }

    @Test
    fun audioInvalidOrOutOfBoundsSkipsAudioEscalation() {
        val visual = 0.40f // below threshold 0.50f
        val invalidAudios = listOf(
            Float.NaN,
            Float.NEGATIVE_INFINITY,
            -1.0f,
            -0.001f,
            1.001f,
            5.0f,
            Float.POSITIVE_INFINITY
        )
        for (audio in invalidAudios) {
            val score = VideoScorePolicy.combineMultimodalScore(
                visualScore = visual,
                visualThreshold = 0.50f,
                audioScore = audio,
                audioValidatedSuccess = true,
                temporalScore = Float.NaN,
                lipSyncAnomaly = 0.0f,
            )
            assertEquals("Audio score $audio must be skipped", visual, score, 1e-6f)
        }
    }

    @Test
    fun authenticAudioDoesNotPenalizeSyntheticVisual() {
        val visual = 0.85f // Synthetic visual
        val authenticAudios = listOf(0.0f, 0.05f, 0.10f, 0.25f, 0.355f)
        for (audio in authenticAudios) {
            val score = VideoScorePolicy.combineMultimodalScore(
                visualScore = visual,
                visualThreshold = 0.50f,
                audioScore = audio,
                audioValidatedSuccess = true,
                audioThreshold = 0.35546875f,
                temporalScore = Float.NaN,
                lipSyncAnomaly = 0.0f,
            )
            assertTrue("Authentic audio ($audio) must not penalize visual score ($visual), got: $score", score >= visual)
        }
    }

    @Test
    fun temporalNanWithValidVisualSkipsTemporalBoost() {
        val visualScores = listOf(0.0f, 0.25f, 0.50f, 0.75f, 1.0f)
        for (visual in visualScores) {
            val score = VideoScorePolicy.combineMultimodalScore(
                visualScore = visual,
                visualThreshold = 0.50f,
                audioScore = Float.NaN,
                audioValidatedSuccess = false,
                temporalScore = Float.NaN,
                lipSyncAnomaly = 0.0f,
            )
            assertEquals("Temporal NaN must produce zero boost", visual, score, 1e-6f)
        }
    }

    @Test
    fun temporalInvalidOrSubthresholdYieldsZeroBoost() {
        val visual = 0.50f
        val invalidTemporals = listOf(
            Float.NaN,
            Float.NEGATIVE_INFINITY,
            -1.0f,
            0.0f,
            0.15f,
            0.30f, // threshold is > 0.30f
            Float.POSITIVE_INFINITY
        )
        for (temp in invalidTemporals) {
            val score = VideoScorePolicy.combineMultimodalScore(
                visualScore = visual,
                visualThreshold = 0.50f,
                audioScore = Float.NaN,
                audioValidatedSuccess = false,
                temporalScore = temp,
                lipSyncAnomaly = 0.0f,
            )
            assertEquals("Temporal score $temp must produce zero boost", visual, score, 1e-6f)
        }
    }

    @Test
    fun bothAudioAndTemporalNanReturnsVisualScoreUnmodified() {
        for (i in 0..100) {
            val visual = i / 100.0f
            val score = VideoScorePolicy.combineMultimodalScore(
                visualScore = visual,
                visualThreshold = 0.50f,
                audioScore = Float.NaN,
                audioValidatedSuccess = false,
                temporalScore = Float.NaN,
                lipSyncAnomaly = 0.0f,
            )
            assertEquals("Visual score $visual must be unmodified when audio and temporal are NaN", visual, score, 1e-6f)
        }
    }

    @Test
    fun lipSyncAnomalyNanOrInvalidYieldsZeroBoost() {
        val visual = 0.42f
        val invalidLip = listOf(
            Float.NaN,
            Float.NEGATIVE_INFINITY,
            -1.0f,
            -0.01f,
            0.0f,
            Float.POSITIVE_INFINITY
        )
        for (lip in invalidLip) {
            val score = VideoScorePolicy.combineMultimodalScore(
                visualScore = visual,
                visualThreshold = 0.50f,
                audioScore = Float.NaN,
                audioValidatedSuccess = false,
                temporalScore = Float.NaN,
                lipSyncAnomaly = lip,
            )
            assertEquals("LipSync score $lip must produce zero boost", visual, score, 1e-6f)
        }
    }

    @Test
    fun extremeFuzzingMatrixNeverThrowsAndInvariantsHold() {
        val rand = Random(42)
        val edgeFloats = listOf(
            Float.NaN,
            Float.NEGATIVE_INFINITY,
            Float.POSITIVE_INFINITY,
            -1.0f,
            -0.001f,
            0.0f,
            0.001f,
            0.5f,
            0.999f,
            1.0f,
            1.001f,
            100.0f,
            Float.MIN_VALUE,
            Float.MAX_VALUE
        )

        for (step in 0..2000) {
            val vScore = if (rand.nextBoolean()) edgeFloats.random(rand) else rand.nextFloat()
            val vThresh = if (rand.nextBoolean()) edgeFloats.random(rand) else rand.nextFloat()
            val aScore = if (rand.nextBoolean()) edgeFloats.random(rand) else rand.nextFloat()
            val aThresh = if (rand.nextBoolean()) edgeFloats.random(rand) else rand.nextFloat()
            val aVal = rand.nextBoolean()
            val tScore = if (rand.nextBoolean()) edgeFloats.random(rand) else rand.nextFloat()
            val lip = if (rand.nextBoolean()) edgeFloats.random(rand) else rand.nextFloat()

            val result = VideoScorePolicy.combineMultimodalScore(
                visualScore = vScore,
                visualThreshold = vThresh,
                audioScore = aScore,
                audioValidatedSuccess = aVal,
                audioThreshold = aThresh,
                temporalScore = tScore,
                lipSyncAnomaly = lip,
            )

            if (!vScore.isFinite() || vScore !in 0f..1f) {
                assertTrue("Invalid visual score must return NaN, got $result", result.isNaN())
            } else {
                assertTrue("Valid visual score must return in [0, 1], got $result", result.isFinite() && result in 0f..1f)
            }
        }
    }

    @Test
    fun visualThresholdNanWithValidVisualPreservesBaseScoreAndNeverReturnsNan() {
        val visual = 0.65f
        val result = VideoScorePolicy.combineMultimodalScore(
            visualScore = visual,
            visualThreshold = Float.NaN,
            audioScore = 0.90f,
            audioValidatedSuccess = true,
            audioThreshold = 0.35546875f,
        )
        // With NaN visualThreshold, audio escalation into visual threshold space cannot be mapped,
        // so base visual score must be preserved intact without IEEE 754 Math.max NaN contamination.
        assertEquals(visual, result, 1e-6f)
    }

    @Test
    fun audioThresholdNanWithValidVisualPreservesBaseScoreAndNeverReturnsNan() {
        val visual = 0.65f
        val result = VideoScorePolicy.combineMultimodalScore(
            visualScore = visual,
            visualThreshold = 0.50f,
            audioScore = 0.90f,
            audioValidatedSuccess = true,
            audioThreshold = Float.NaN,
        )
        assertEquals(visual, result, 1e-6f)
    }
}
