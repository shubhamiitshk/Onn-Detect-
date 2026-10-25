package com.onndetect

import com.onndetect.detection.video.VideoScorePolicy
import com.onndetect.detection.visual.PerioralLandmarkAnalyzer
import com.onndetect.detection.visual.ResidualHeatmapExtractor
import com.onndetect.detection.visual.SRMForensicAnalyzer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Unit tests for the v10.1 Deep-Think algorithmic enhancements:
 * 1. PatchCraft Rich-vs-Poor Texture Contrast Calibration (Portrait-Mode Bokeh Guard & Diffusion Reinforcement)
 * 2. Saliency-Guided Localized Hotspot ROI Extraction (32x32 Residual Heatmap Z-Score Locator)
 * 3. Bilateral Malar rPPG Cardiac Pulse Phase-Locking Correlation
 */
class DeepThinkAlgorithmsTest {

    @Test
    fun `portrait mode bokeh with low poor-mosaic residual is dampened`() {
        val rich = 0.58f
        val poor = 0.10f
        val calibrated = SRMForensicAnalyzer.computeContrastCalibratedScore(rich, poor)
        assertTrue("Expected bokeh dampening below raw richScore ($rich), got $calibrated", calibrated < rich)
        assertEquals(0.58f - 0.15f * 0.48f, calibrated, 1e-4f)
    }

    @Test
    fun `uniform diffusion residual across rich and poor mosaics is reinforced`() {
        val rich = 0.62f
        val poor = 0.78f
        val calibrated = SRMForensicAnalyzer.computeContrastCalibratedScore(rich, poor)
        assertTrue("Expected co-synthetic reinforcement above richScore ($rich), got $calibrated", calibrated > rich)
        assertEquals(0.75f * 0.62f + 0.25f * 0.78f, calibrated, 1e-4f)
    }

    @Test
    fun `hotspot extractor pinpoints localized 8x8 tampering cluster with high z-score`() {
        val grid = 32
        val heatmap = FloatArray(grid * grid) { 0.12f }
        // Inject localized tampering spike at rows 8..15, cols 16..23
        for (r in 8 until 16) {
            for (c in 16 until 24) {
                heatmap[r * grid + c] = 0.94f
            }
        }
        val hotspot = ResidualHeatmapExtractor.extractHotspotBoundingBox(heatmap, grid, windowCells = 8)
        assertNotNull(hotspot)
        assertEquals(8, hotspot!!.gridRow0)
        assertEquals(16, hotspot.gridCol0)
        assertEquals(16, hotspot.gridRow1)
        assertEquals(24, hotspot.gridCol1)
        assertTrue("Expected high Z-score for localized spike, got ${hotspot.zScore}", hotspot.zScore > 3.0f)
    }

    @Test
    fun `uniform heatmap produces near-zero z-score`() {
        val grid = 32
        val heatmap = FloatArray(grid * grid) { 0.50f }
        val hotspot = ResidualHeatmapExtractor.extractHotspotBoundingBox(heatmap, grid, windowCells = 8)
        assertNotNull(hotspot)
        assertEquals(0f, hotspot!!.zScore, 1e-4f)
    }

    @Test
    fun `bilateral rppg phase-locked malar pulses yield low anomaly`() {
        val left = doubleArrayOf(0.340, 0.344, 0.349, 0.343, 0.339, 0.345, 0.350, 0.342)
        val right = doubleArrayOf(0.338, 0.342, 0.347, 0.341, 0.337, 0.343, 0.348, 0.340)
        val anomaly = PerioralLandmarkAnalyzer.computeRppgAnomalyFromSeries(left, right)
        assertTrue("Phase-locked bilateral cheeks should have 0 anomaly, got $anomaly", anomaly <= 0.05f)
    }

    @Test
    fun `desynchronized synthetic malar chromaticity yields high rppg anomaly`() {
        val left = doubleArrayOf(0.340, 0.346, 0.339, 0.348, 0.337, 0.349, 0.338, 0.347)
        val right = doubleArrayOf(0.345, 0.339, 0.347, 0.338, 0.348, 0.336, 0.347, 0.338)
        val anomaly = PerioralLandmarkAnalyzer.computeRppgAnomalyFromSeries(left, right)
        assertTrue("Anti-phase synthetic cheeks should trigger high rPPG anomaly, got $anomaly", anomaly >= 0.85f)
    }

    @Test
    fun `ai video with real music and smooth motion is not dragged below visual threshold`() {
        val threshold = 0.5782f
        val score = VideoScorePolicy.combineMultimodalScore(
            visualScore = 0.78f,
            visualThreshold = threshold,
            audioScore = 0.05f,
            audioValidatedSuccess = true,
            temporalScore = 0.02f,
            lipSyncAnomaly = 0f,
        )
        assertTrue("AI visual score 0.78 must remain >= threshold ($threshold) even with real audio, got $score", score >= 0.78f)
    }

    @Test
    fun `real video with ai voice clone escalates above visual threshold`() {
        val threshold = 0.5782f
        val score = VideoScorePolicy.combineMultimodalScore(
            visualScore = 0.12f,
            visualThreshold = threshold,
            audioScore = 0.88f,
            audioValidatedSuccess = true,
            audioThreshold = 0.3555f,
            temporalScore = 0.04f,
            lipSyncAnomaly = 0f,
        )
        assertTrue("Real video dubbed with AI voice (0.88) should escalate above $threshold, got $score", score >= threshold)
    }

    @Test
    fun `authentic video with authentic audio stays authentic`() {
        val threshold = 0.5782f
        val score = VideoScorePolicy.combineMultimodalScore(
            visualScore = 0.14f,
            visualThreshold = threshold,
            audioScore = 0.08f,
            audioValidatedSuccess = true,
            audioThreshold = 0.3555f,
            temporalScore = 0.05f,
            lipSyncAnomaly = 0f,
        )
        assertEquals(0.14f, score, 1e-4f)
    }
}
