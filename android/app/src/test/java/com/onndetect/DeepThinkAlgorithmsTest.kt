package com.onndetect

import com.onndetect.detection.audio.WindowPooling
import com.onndetect.detection.video.VideoScorePolicy
import com.onndetect.detection.visual.PerioralLandmarkAnalyzer
import com.onndetect.detection.visual.ResidualHeatmapExtractor
import com.onndetect.detection.visual.SRMForensicAnalyzer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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

    @Test
    fun `v11 localized ROI splice with high hotspot z-score boosts calibrated SRM score`() {
        val rich = 0.36f
        val poor = 0.19f
        val roi = 0.89f
        val zScore = 3.2f
        val base = SRMForensicAnalyzer.computeContrastCalibratedScore(rich, poor)
        val boosted = SRMForensicAnalyzer.computeContrastAndRoiCalibratedScore(rich, poor, roi, zScore)
        assertTrue("Expected ROI splice boost above base ($base), got $boosted", boosted > base + 0.15f)
    }

    @Test
    fun `v11 bilateral rppg phase anomaly boosts borderline talking-head video score`() {
        val threshold = 0.5782f
        val withoutRppg = VideoScorePolicy.combineMultimodalScore(
            visualScore = 0.53f,
            visualThreshold = threshold,
            audioScore = 0.10f,
            audioValidatedSuccess = true,
            temporalScore = 0.10f,
            lipSyncAnomaly = 0.10f,
            rppgPhaseAnomaly = 0.05f,
        )
        val withRppg = VideoScorePolicy.combineMultimodalScore(
            visualScore = 0.53f,
            visualThreshold = threshold,
            audioScore = 0.10f,
            audioValidatedSuccess = true,
            temporalScore = 0.10f,
            lipSyncAnomaly = 0.10f,
            rppgPhaseAnomaly = 0.92f,
        )
        assertTrue("Expected rPPG phase anomaly to boost video score above $withoutRppg, got $withRppg", withRppg > withoutRppg)
        assertTrue("Expected rPPG anomaly to push borderline 0.53 video >= threshold ($threshold), got $withRppg", withRppg >= threshold)
    }

    @Test
    fun `v11 multi-window audio combiner protects low-anomaly noisy speech and boosts synthetic`() {
        val noisyScores = floatArrayOf(0.42f, 0.38f, 0.45f)
        val noisyWithCleanBg = WindowPooling.combineMultiWindowAndNonVocalAnomaly(
            probs = noisyScores,
            nonVocalAnomaly = 0.12f,
        )
        val baselinePooled = WindowPooling.poolProbabilities(noisyScores)
        assertTrue("Clean non-vocal background should pull borderline score down, got $noisyWithCleanBg vs $baselinePooled", noisyWithCleanBg < baselinePooled)

        val synthAnomaly = WindowPooling.combineMultiWindowAndNonVocalAnomaly(
            probs = noisyScores,
            nonVocalAnomaly = 0.88f,
        )
        assertTrue("High non-vocal quantization anomaly should boost score, got $synthAnomaly vs $baselinePooled", synthAnomaly > baselinePooled)
    }

    @Test
    fun `v11 VFM temporal acceleration detects oscillating jerky synthetic trajectories`() {
        val smooth = (0 until 10).map { t ->
            val th = t * 0.05
            floatArrayOf(kotlin.math.cos(th).toFloat(), kotlin.math.sin(th).toFloat())
        }
        val smoothAnomaly = VideoScorePolicy.computeVfmTemporalAccelerationAnomaly(smooth)
        assertEquals(0.0f, smoothAnomaly, 1e-4f)

        val jerky = (0 until 10).map { t ->
            val th = t * 0.05
            val flutter = if (t % 2 == 0) 0.12f else -0.12f
            floatArrayOf(
                (kotlin.math.cos(th).toFloat() + flutter),
                (kotlin.math.sin(th).toFloat() - flutter),
            )
        }
        val jerkyAnomaly = VideoScorePolicy.computeVfmTemporalAccelerationAnomaly(jerky)
        assertTrue("Expected oscillating trajectory to yield high anomaly, got $jerkyAnomaly", jerkyAnomaly > 0.50f)
    }

    @Test
    fun onDemandForensicVlmRoutesAndFusesDialectically() {
        // 1. Confident fast scan exits immediately without invoking VLM (0 ms overhead)
        val fastRoute = com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer.routeAsPerNeed(
            fastFusionProb = 0.95f,
            decisionThreshold = 0.5782f,
            srmCalibratedScore = 0.89f,
            vfmProbeScore = 0.94f,
            hotspotZScore = 0.8f,
        )
        assertFalse(fastRoute.invokeOnDeviceVlm)
        assertEquals(
            com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer.EscalationMode.FAST_EXIT_CONFIDENT,
            fastRoute.mode,
        )

        // 2. Ambiguous / cross-engine conflict + ROI hotspot escalates to on-device VLM
        val ambigRoute = com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer.routeAsPerNeed(
            fastFusionProb = 0.53f,
            decisionThreshold = 0.5782f,
            srmCalibratedScore = 0.24f,
            vfmProbeScore = 0.84f,
            hotspotZScore = 2.65f,
            activeDomain = "document_text",
            docTamperScore = 0.62f,
        )
        assertTrue(ambigRoute.invokeOnDeviceVlm)
        assertTrue(ambigRoute.expectedVoiScore >= 0.38f)

        // 3. Dialectical Contrastive Logit Probing resolves ambiguous splice and emits <think> CoT
        val outcome = com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer.fuseDialecticalLogits(
            fastFusionProb = 0.53f,
            forgeryLogits6d = floatArrayOf(0.4f, 0.8f, 0.5f, 1.2f, 2.8f, 0.1f),
            authenticCounterLogits6d = floatArrayOf(0.1f, -0.2f, 0.0f, -0.4f, -0.9f, 0.2f),
            roiForgeryLogits6d = floatArrayOf(0.2f, 0.5f, 0.3f, 1.1f, 3.1f, 0.0f),
            hotspotZScore = 2.65f,
            activeDomain = "document_text",
            decisionThreshold = 0.5782f,
        )
        assertTrue("Expected posterior >= threshold, got ${outcome.posteriorProb}", outcome.posteriorProb >= 0.5782f)
        assertEquals("typography_document_layout", outcome.dominantFlawAxis)
        assertTrue(outcome.dialecticalExplanation.contains("<think>"))
        assertTrue(outcome.dialecticalExplanation.contains("<answer>Fake</answer>"))

        // 4. Tier-2 Model Pack selector & Signed Pack Verification (Qwen3.5-2B / Qwen3.5-0.8B / SmolVLM2-500M)
        val pack6gb = com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer.selectOptimalTier2ModelPack(
            totalDeviceRamMb = 6144,
            freeStorageMb = 1200,
        )
        assertEquals(
            com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer.Tier2ModelPack.QWEN3_5_0_8B_Q4_K_M,
            pack6gb,
        )
        assertEquals(738, pack6gb.combinedDownloadMb)

        val pack8gbDeep = com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer.selectOptimalTier2ModelPack(
            totalDeviceRamMb = 8192,
            freeStorageMb = 3000,
            allowLarge2GbPack = true,
        )
        assertEquals(
            com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer.Tier2ModelPack.QWEN3_5_2B_Q4_K_M,
            pack8gbDeep,
        )
        assertEquals(1948, pack8gbDeep.combinedDownloadMb)

        // Verify diagnosticOnly shadow mode preserves fastFusionProb while recording VLM probability
        val shadowOutcome = com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer.fuseDialecticalLogits(
            fastFusionProb = 0.44f,
            forgeryLogits6d = floatArrayOf(Float.NaN, 0.8f, 0.5f, 1.2f, 2.8f, Float.NaN),
            authenticCounterLogits6d = floatArrayOf(Float.NaN, -0.2f, 0.0f, -0.4f, -0.9f, Float.NaN),
            activeDomain = "document_text",
            diagnosticOnly = true,
        )
        assertEquals(0.44f, shadowOutcome.posteriorProb, 1e-5f)
        assertTrue(shadowOutcome.vlmDialecticalProb > 0.70f)

        // 5. Verify measurable original-source crop specification & downscaleRatio telemetry
        val cropSpec = com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer.computeOriginalSourceCropSpec(
            sourceWidth = 4000,
            sourceHeight = 3000,
            normX0 = 0.40f,
            normY0 = 0.40f,
            normX1 = 0.512f,
            normY1 = 0.54933333f,
        )
        assertEquals(448, cropSpec.width)
        assertEquals(448, cropSpec.targetEncodedEdgePx)
        assertEquals(1.0f, cropSpec.downscaleRatio, 1e-4f)

        // 6. Verify narrowed label taxonomy (ISP denoising & conventional edits = 0, generative = 1)
        assertEquals(
            0,
            com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer.LabelTaxonomyCategory.AUTHENTIC_CAMERA_OR_ISP.binaryLabel,
        )
        assertEquals(
            0,
            com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer.LabelTaxonomyCategory.AUTHENTIC_CONVENTIONAL_EDIT_OR_SCREENSHOT.binaryLabel,
        )
        assertEquals(
            1,
            com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer.LabelTaxonomyCategory.GENERATIVE_LOCAL_EDIT.binaryLabel,
        )
        assertEquals(
            1,
            com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer.LabelTaxonomyCategory.GENERATIVE_FULL_SYNTHESIS.binaryLabel,
        )

        // 7. Verify audio calibration threshold constant and frontend ID match assets/constants/lcnn_audio_threshold.json
        assertEquals(0.35546875f, com.onndetect.detection.audio.AudioClassifier.DEFAULT_THRESHOLD, 1e-7f)
        assertEquals(1, com.onndetect.detection.audio.AudioClassifier.WINDOWS_PER_FILE)
        assertEquals(
            "lfcc-v3-sr16000-f128-c40-nfft512-hop160-topdbnone-resampler-quadratic_window_sinc_v1-float64stft-float32",
            com.onndetect.detection.audio.LFCCExtractor.FRONTEND_ID,
        )

        // 8. Verify EvictionReport distinguishes cache removal from native resource closure
        val partialFailureReport = com.onndetect.inference.LiteRTInferenceEngine.EvictionReport(
            modelFile = "siglip_probe_int8.tflite",
            wasCached = true,
            removedFromCache = true,
            interpreterClosed = true,
            delegateClosed = false,
            mappingClosed = true,
            closeError = RuntimeException("Delegate close failed"),
        )
        assertFalse(partialFailureReport.resourcesCleanlyReleased)

        val cleanReport = com.onndetect.inference.LiteRTInferenceEngine.EvictionReport(
            modelFile = "siglip_probe_int8.tflite",
            wasCached = true,
            removedFromCache = true,
            interpreterClosed = true,
            delegateClosed = true,
            mappingClosed = true,
            closeError = null,
        )
        assertTrue(cleanReport.resourcesCleanlyReleased)

        // 9. Verify FusionFeatureSchema rejects NaN or negative SRM score in slot 0 while permitting NaN in slot 6
        val validVec = com.onndetect.domain.pipeline.FusionFeatureSchema.build(
            0.42f, 0.31f, 0.25f, 0.18f, 1.0f, 1.0f, Float.NaN, 0.5f, 0.5f, 0.15f,
        )
        assertEquals(10, validVec.size)
        val nanSrmRejected = runCatching {
            com.onndetect.domain.pipeline.FusionFeatureSchema.build(
                Float.NaN, 0.31f, 0.25f, 0.18f, 1.0f, 1.0f, Float.NaN, 0.5f, 0.5f, 0.15f,
            )
        }.isFailure
        assertTrue("Expected Float.NaN SRM score in slot 0 to be rejected", nanSrmRejected)
        val negSrmRejected = runCatching {
            com.onndetect.domain.pipeline.FusionFeatureSchema.build(
                -1f, 0.31f, 0.25f, 0.18f, 1.0f, 1.0f, Float.NaN, 0.5f, 0.5f, 0.15f,
            )
        }.isFailure
        assertTrue("Expected -1f SRM score in slot 0 to be rejected", negSrmRejected)
    }
}
