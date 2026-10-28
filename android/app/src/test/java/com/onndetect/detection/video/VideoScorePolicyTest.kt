package com.onndetect.detection.video

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class VideoScorePolicyTest {

    @Test
    fun `visual synthetic score dominates combination`() {
        val score = VideoScorePolicy.combineMultimodalScore(
            visualScore = 0.95f,
            visualThreshold = 0.50f,
            audioScore = 0.10f,
            audioValidatedSuccess = true,
        )
        assertTrue(score >= 0.95f)
    }

    @Test
    fun `synthetic audio escalates visual score across threshold`() {
        val score = VideoScorePolicy.combineMultimodalScore(
            visualScore = 0.40f,
            visualThreshold = 0.50f,
            audioScore = 0.90f,
            audioValidatedSuccess = true,
            audioThreshold = 0.35546875f,
        )
        assertTrue(score >= 0.50f)
    }

    @Test
    fun `production combineMultimodalScore decouples unvalidated heuristics from final score`() {
        val baseScore = VideoScorePolicy.combineMultimodalScore(
            visualScore = 0.50f,
            visualThreshold = 0.50f,
            audioScore = Float.NaN,
            audioValidatedSuccess = false,
            temporalScore = Float.NaN,
            lipSyncAnomaly = 0f,
        )
        val scoreWithHeuristics = VideoScorePolicy.combineMultimodalScore(
            visualScore = 0.50f,
            visualThreshold = 0.50f,
            audioScore = Float.NaN,
            audioValidatedSuccess = false,
            temporalScore = 0.85f,
            lipSyncAnomaly = 0.70f,
            rppgPhaseAnomaly = 0.90f,
            enableExperimentalHeuristicBoost = false,
        )
        assertEquals("Production score must preserve base without heuristic distortion", baseScore, scoreWithHeuristics, 1e-6f)
    }

    @Test
    fun `temporal and lip sync anomaly boosts score when experimental boost explicitly enabled`() {
        val baseScore = VideoScorePolicy.combineMultimodalScore(
            visualScore = 0.50f,
            visualThreshold = 0.50f,
            audioScore = Float.NaN,
            audioValidatedSuccess = false,
            temporalScore = Float.NaN,
            lipSyncAnomaly = 0f,
            enableExperimentalHeuristicBoost = true,
        )
        val boostedScore = VideoScorePolicy.combineMultimodalScore(
            visualScore = 0.50f,
            visualThreshold = 0.50f,
            audioScore = Float.NaN,
            audioValidatedSuccess = false,
            temporalScore = 0.85f,
            lipSyncAnomaly = 0.70f,
            enableExperimentalHeuristicBoost = true,
        )
        assertTrue(boostedScore > baseScore)
    }

    @Test
    fun `authentic audio cannot drag visual fake score below threshold`() {
        val score = VideoScorePolicy.combineMultimodalScore(
            visualScore = 0.82f,
            visualThreshold = 0.50f,
            audioScore = 0.05f,
            audioValidatedSuccess = true,
            audioThreshold = 0.35546875f,
        )
        assertTrue(score >= 0.82f)
    }

    @Test
    fun `silent video visual only fallback`() {
        val score = VideoScorePolicy.combineMultimodalScore(
            visualScore = 0.68f,
            visualThreshold = 0.50f,
            audioScore = Float.NaN,
            audioValidatedSuccess = false,
        )
        assertEquals(0.68f, score, 1e-4f)
    }

    @Test
    fun `nan visual score preserves nan without collapsing to zero`() {
        val score = VideoScorePolicy.combineMultimodalScore(
            visualScore = Float.NaN,
            visualThreshold = 0.50f,
            audioScore = 0.90f,
            audioValidatedSuccess = true,
        )
        assertTrue(score.isNaN())
    }

    @Test
    fun `multimodal score preserves nan when visual score is nan or negative`() {
        assertTrue(VideoScorePolicy.combineMultimodalScore(
            visualScore = Float.NaN, visualThreshold = 0.50f, audioScore = 0.90f, audioValidatedSuccess = true
        ).isNaN())
        assertTrue(VideoScorePolicy.combineMultimodalScore(
            visualScore = -1f, visualThreshold = 0.50f, audioScore = 0.90f, audioValidatedSuccess = true
        ).isNaN())
    }

    @Test
    fun `nan audio score does not trigger audio escalation`() {
        val visual = 0.40f
        val score = VideoScorePolicy.combineMultimodalScore(
            visualScore = visual, visualThreshold = 0.50f,
            audioScore = Float.NaN, audioValidatedSuccess = true
        )
        assertEquals(visual, score, 1e-4f)
    }

    @Test
    fun `nan temporal score yields zero temporal boost`() {
        val visual = 0.60f
        val score = VideoScorePolicy.combineMultimodalScore(
            visualScore = visual, visualThreshold = 0.50f,
            temporalScore = Float.NaN, lipSyncAnomaly = 0f
        )
        assertEquals(visual, score, 1e-4f)
    }

    @Test
    fun `unspecified optional modalities default to nan without affecting base score`() {
        val visual = 0.72f
        val score = VideoScorePolicy.combineMultimodalScore(
            visualScore = visual, visualThreshold = 0.50f
        )
        assertEquals(visual, score, 1e-4f)
    }

    @Test
    fun `maximum anomaly boost is strictly capped at 30 percent`() {
        val score = VideoScorePolicy.combineMultimodalScore(
            visualScore = 0.0f,
            visualThreshold = 0.50f,
            audioScore = Float.NaN,
            audioValidatedSuccess = false,
            temporalScore = 1.0f,
            lipSyncAnomaly = 1.0f,
            enableExperimentalHeuristicBoost = true,
        )
        // Max boost = 0.30 -> final = 0.0 + (1.0 - 0.0) * 0.30 = 0.30
        assertEquals(0.30f, score, 1e-4f)
    }
}
