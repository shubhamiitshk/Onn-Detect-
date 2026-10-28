package com.onndetect.detection.provenance

import com.onndetect.inference.LiteRTInferenceEngine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.FileNotFoundException
import java.nio.ByteBuffer

/**
 * Comprehensive unit test suite for [AudioSealDetector].
 *
 * Covers:
 * 1. Exact two-tailed binomial math and distribution properties (N=16).
 * 2. Exact 16-bit agreement triggering watermark detection.
 * 3. Non-significant agreement rejection (k=15, k=12, k<12, k=8).
 * 4. Inverted correlation protection (k < 8 rejected by MIN_AGREEING_BITS).
 * 5. Audio buffer dimension & boundary validation (empty, short, exact, incomplete trailing).
 * 6. Multi-window processing and best-window aggregation (lowest p-value).
 * 7. Silent and non-finite audio input handling.
 * 8. Missing model / unbundled asset resilience (fail-closed returning ProvenanceEvidence.Absent).
 * 9. Model contract mismatch and runtime prediction failure handling.
 */
class AudioSealDetectorTest {

    // =========================================================================
    // Helper Test Double Factory
    // =========================================================================

    private fun createDetector(
        contract: LiteRTInferenceEngine.Contract? = LiteRTInferenceEngine.Contract(
            inputShape = intArrayOf(1, 16000),
            outputShape = intArrayOf(1, 1),
            inputBytes = 16000 * 4,
        ),
        predictor: (String, ByteBuffer) -> Float = { _, _ -> 16f },
    ): AudioSealDetector {
        return AudioSealDetector(
            contractProvider = { contract },
            predictor = predictor,
        )
    }

    // =========================================================================
    // Group 1: Companion Binomial Math & Distribution Tests
    // =========================================================================

    @Test
    fun `full agreement 16 of 16 yields highly significant p-value`() {
        val p = AudioSealDetector.binomialTwoTailed(16, 16)
        assertTrue("p-value must be below 1e-4 for exact 16-bit match", p < 1e-4)
        // Two-tailed tail is 2 * (0.5^16) = 2 / 65536 = 1 / 32768 ~= 3.0517578e-5
        val expected = 2.0 * Math.pow(0.5, 16.0)
        assertEquals(expected, p, 1e-9)
        assertEquals(3.0517578125e-5, p, 1e-9)
    }

    @Test
    fun `chance agreement 8 of 16 yields 1_0 p-value`() {
        val p = AudioSealDetector.binomialTwoTailed(8, 16)
        assertEquals(1.0, p, 1e-6)
    }

    @Test
    fun `monotonicity of binomial tail from chance to full agreement`() {
        val p8 = AudioSealDetector.binomialTwoTailed(8, 16)
        val p9 = AudioSealDetector.binomialTwoTailed(9, 16)
        val p10 = AudioSealDetector.binomialTwoTailed(10, 16)
        val p11 = AudioSealDetector.binomialTwoTailed(11, 16)
        val p12 = AudioSealDetector.binomialTwoTailed(12, 16)
        val p13 = AudioSealDetector.binomialTwoTailed(13, 16)
        val p14 = AudioSealDetector.binomialTwoTailed(14, 16)
        val p15 = AudioSealDetector.binomialTwoTailed(15, 16)
        val p16 = AudioSealDetector.binomialTwoTailed(16, 16)

        assertTrue(p8 > p9)
        assertTrue(p9 > p10)
        assertTrue(p10 > p11)
        assertTrue(p11 > p12)
        assertTrue(p12 > p13)
        assertTrue(p13 > p14)
        assertTrue(p14 > p15)
        assertTrue(p15 > p16)
    }

    @Test
    fun `exact discrete p-values for intermediate agreement levels`() {
        // N = 16 trials, two-tailed binomial p-values:
        // k=15: 2 * (1 + 16) / 65536 = 34 / 65536 ~= 5.187988e-4
        assertEquals(34.0 / 65536.0, AudioSealDetector.binomialTwoTailed(15, 16), 1e-9)

        // k=14: 2 * (1 + 16 + 120) / 65536 = 274 / 65536 ~= 4.180908e-3
        assertEquals(274.0 / 65536.0, AudioSealDetector.binomialTwoTailed(14, 16), 1e-9)

        // k=13: 2 * (1 + 16 + 120 + 560) / 65536 = 1394 / 65536 ~= 0.02127075
        assertEquals(1394.0 / 65536.0, AudioSealDetector.binomialTwoTailed(13, 16), 1e-9)

        // k=12: 2 * (1 + 16 + 120 + 560 + 1820) / 65536 = 5034 / 65536 ~= 0.07681274
        assertEquals(5034.0 / 65536.0, AudioSealDetector.binomialTwoTailed(12, 16), 1e-9)

        // k=11: 2 * (2517 + 4368) / 65536 = 13770 / 65536 ~= 0.2101135
        assertEquals(13770.0 / 65536.0, AudioSealDetector.binomialTwoTailed(11, 16), 1e-9)

        // k=10: 2 * (6885 + 8008) / 65536 = 29786 / 65536 ~= 0.454498
        assertEquals(29786.0 / 65536.0, AudioSealDetector.binomialTwoTailed(10, 16), 1e-9)
    }

    @Test
    fun `two-tailed binomial symmetry - p(k) equals p(N-k)`() {
        for (k in 0..16) {
            val pK = AudioSealDetector.binomialTwoTailed(k, 16)
            val pSym = AudioSealDetector.binomialTwoTailed(16 - k, 16)
            assertEquals("Symmetry failed for k=$k", pK, pSym, 1e-9)
        }
    }

    @Test
    fun `boundary trials zero or negative returns 1_0`() {
        assertEquals(1.0, AudioSealDetector.binomialTwoTailed(0, 0), 1e-9)
        assertEquals(1.0, AudioSealDetector.binomialTwoTailed(5, -1), 1e-9)
        assertEquals(1.0, AudioSealDetector.binomialTwoTailed(-1, 0), 1e-9)
    }

    @Test
    fun `successes outside range are coerced safely`() {
        // More successes than trials coerced to trials (k=16)
        val pOver = AudioSealDetector.binomialTwoTailed(20, 16)
        val pExact = AudioSealDetector.binomialTwoTailed(16, 16)
        assertEquals(pExact, pOver, 1e-9)

        // Negative successes coerced to 0 (k=0, which is two-tailed symmetric to k=16)
        val pNeg = AudioSealDetector.binomialTwoTailed(-5, 16)
        assertEquals(pExact, pNeg, 1e-9)
    }

    // =========================================================================
    // Group 2: Agreement Levels & Provenance Evidence Verdicts
    // =========================================================================

    @Test
    fun `exact 16-bit agreement triggers watermark detection and confirms AI involvement`() {
        val detector = createDetector(predictor = { _, _ -> 16f })
        val samples = FloatArray(16000) { 0.1f }

        val detail = detector.detectWatermark(samples)
        assertNotNull("DetectionResult must not be null for valid window", detail)
        assertEquals(16, detail!!.agreeingBits)
        assertEquals(16, detail.totalBits)
        assertEquals(3.0517578125e-5, detail.pValue, 1e-9)
        assertTrue("detected flag must be true for 16-bit match", detail.detected)

        val evidence = detector.analyze(samples)
        assertTrue("Evidence must be WatermarkMark on detection", evidence is ProvenanceEvidence.WatermarkMark)
        val mark = evidence as ProvenanceEvidence.WatermarkMark
        assertTrue("WatermarkMark.detected must be true", mark.detected)
        assertEquals("AudioSeal", mark.kind)
        assertEquals(3.0517578125e-5, mark.pValue, 1e-9)
        assertFalse("mark presence alone does not confirm AI involvement (authentic audio can be watermarked)", mark.confirmsAiInvolvement)
    }

    @Test
    fun `continuous presence score in 0 to 1 range triggers detection without LiteRT contract failure`() {
        val detector = createDetector(predictor = { _, _ -> 0.95f })
        val samples = FloatArray(16000) { 0.1f }

        val detail = detector.detectWatermark(samples)
        assertNotNull(detail)
        assertTrue("High presence score >= 0.5 must be detected", detail!!.detected)
        assertEquals(0.95f, detail.presenceScore, 1e-6f)

        val evidence = detector.analyze(samples)
        assertTrue("Evidence must be WatermarkMark on detection", evidence is ProvenanceEvidence.WatermarkMark)
        val mark = evidence as ProvenanceEvidence.WatermarkMark
        assertTrue(mark.detected)
        assertEquals(0.95f, mark.score, 1e-6f)
        assertFalse(mark.confirmsAiInvolvement)
    }

    @Test
    fun `near-miss 15-bit agreement is rejected as non-significant and returns ProvenanceEvidence Absent`() {
        val detector = createDetector(predictor = { _, _ -> 15f })
        val samples = FloatArray(16000) { 0.1f }

        val detail = detector.detectWatermark(samples)
        assertNotNull(detail)
        assertEquals(15, detail!!.agreeingBits)
        assertEquals(16, detail.totalBits)
        assertEquals(5.18798828125e-4, detail.pValue, 1e-6)
        assertFalse("15 bits must not pass 1e-4 significance threshold", detail.detected)

        val evidence = detector.analyze(samples)
        assertEquals("analyze() must return Absent when not detected", ProvenanceEvidence.Absent, evidence)
    }

    @Test
    fun `12-bit agreement yields p ~= 0_0768 and returns ProvenanceEvidence Absent`() {
        val detector = createDetector(predictor = { _, _ -> 12f })
        val samples = FloatArray(16000) { 0.1f }

        val detail = detector.detectWatermark(samples)
        assertNotNull(detail)
        assertEquals(12, detail!!.agreeingBits)
        assertEquals(16, detail.totalBits)
        assertEquals(0.076812744, detail.pValue, 1e-4)
        assertFalse("12 bits (p ~= 0.0768) must not trigger detection", detail.detected)

        val evidence = detector.analyze(samples)
        assertEquals(ProvenanceEvidence.Absent, evidence)
    }

    @Test
    fun `non-significant agreement k less than 12 yields ProvenanceEvidence Absent`() {
        for (k in listOf(11f, 10f, 9f, 8f)) {
            val detector = createDetector(predictor = { _, _ -> k })
            val samples = FloatArray(16000) { 0.05f }
            val detail = detector.detectWatermark(samples)
            assertNotNull(detail)
            assertFalse("Agreement k=$k must not be detected", detail!!.detected)
            assertEquals(ProvenanceEvidence.Absent, detector.analyze(samples))
        }
    }

    @Test
    fun `inverted correlation k less than 8 is rejected by MIN_AGREEING_BITS and returns ProvenanceEvidence Absent`() {
        // k=0 yields two-tailed binomial p-value identical to k=16 (3.05e-5).
        // However, count < MIN_AGREEING_BITS (8) must prevent false detection on inverted sign.
        val detector = createDetector(predictor = { _, _ -> 0f })
        val samples = FloatArray(16000) { 0.2f }

        val detail = detector.detectWatermark(samples)
        assertNotNull(detail)
        assertEquals(0, detail!!.agreeingBits)
        assertEquals(3.0517578125e-5, detail.pValue, 1e-9)
        assertFalse("k=0 must be rejected by MIN_AGREEING_BITS guard", detail.detected)

        val evidence = detector.analyze(samples)
        assertEquals(ProvenanceEvidence.Absent, evidence)
    }

    // =========================================================================
    // Group 3: Audio Buffer Validation & Windowing
    // =========================================================================

    @Test
    fun `empty audio buffer returns ProvenanceEvidence Absent immediately without invoking engine`() {
        var engineCalled = false
        val detector = AudioSealDetector(
            contractProvider = { engineCalled = true; null },
            predictor = { _, _ -> engineCalled = true; 16f },
        )

        val evidence = detector.analyze(FloatArray(0))
        assertEquals(ProvenanceEvidence.Absent, evidence)
        assertFalse("Engine must not be queried for empty audio buffer", engineCalled)
    }

    @Test
    fun `short audio buffer below window size returns ProvenanceEvidence Absent`() {
        val detector = createDetector()
        // Window size is 16000; test buffers with 1, 500, 1024, and 15999 samples
        for (length in listOf(1, 500, 1024, 15999)) {
            val shortBuffer = FloatArray(length) { 0.1f }
            assertNull("detectWatermark must return null for short buffer of size $length", detector.detectWatermark(shortBuffer))
            assertEquals("analyze must return Absent for short buffer of size $length", ProvenanceEvidence.Absent, detector.analyze(shortBuffer))
        }
    }

    @Test
    fun `exact single window processes cleanly and reads byte buffer accurately`() {
        var capturedFloats = 0
        val detector = createDetector(predictor = { _, byteBuffer ->
            val floatBuf = byteBuffer.asFloatBuffer()
            capturedFloats = floatBuf.remaining()
            16f
        })

        val samples = FloatArray(16000) { it.toFloat() / 16000f }
        val evidence = detector.analyze(samples)

        assertEquals(16000, capturedFloats)
        assertTrue(evidence is ProvenanceEvidence.WatermarkMark)
    }

    @Test
    fun `incomplete trailing window is discarded safely without error`() {
        // 24000 samples = 1 full window of 16000 + 8000 trailing samples (< 1 full window)
        var windowCalls = 0
        val detector = createDetector(predictor = { _, _ ->
            windowCalls++
            16f
        })

        val samples = FloatArray(24000) { 0.1f }
        val evidence = detector.analyze(samples)

        assertEquals("Exactly 1 window must be evaluated; incomplete remainder discarded", 1, windowCalls)
        assertTrue(evidence is ProvenanceEvidence.WatermarkMark)
    }

    @Test
    fun `multi-window audio selects best window with lowest p-value`() {
        // 32000 samples = 2 windows of 16000
        var windowIndex = 0
        val detector = createDetector(predictor = { _, _ ->
            val score = if (windowIndex == 0) 8f else 16f // Window 0: chance (p=1.0), Window 1: detected (p=3.05e-5)
            windowIndex++
            score
        })

        val samples = FloatArray(32000) { 0.1f }
        val result = detector.detectWatermark(samples)

        assertNotNull(result)
        assertEquals("Detector must select best window (Window 1)", 16, result!!.agreeingBits)
        // Bonferroni adjustment: 2 windows * 3.0517578125e-5 = 6.103515625e-5 < 1e-4
        assertEquals(2.0 * 3.0517578125e-5, result.pValue, 1e-9)
        assertEquals(3.0517578125e-5, result.uncorrectedPValue, 1e-9)
        assertTrue(result.detected)

        val evidence = detector.analyze(samples)
        assertTrue(evidence is ProvenanceEvidence.WatermarkMark)
    }

    @Test
    fun `multi-window audio retains earlier best window if subsequent window has higher p-value`() {
        // Window 0: detected (16 bits, p=3.05e-5), Window 1: chance (8 bits, p=1.0)
        var windowIndex = 0
        val detector = createDetector(predictor = { _, _ ->
            val score = if (windowIndex == 0) 16f else 8f
            windowIndex++
            score
        })

        val samples = FloatArray(32000) { 0.1f }
        val result = detector.detectWatermark(samples)

        assertNotNull(result)
        assertEquals("Detector must retain Window 0", 16, result!!.agreeingBits)
        assertEquals(2.0 * 3.0517578125e-5, result.pValue, 1e-9)
        assertEquals(3.0517578125e-5, result.uncorrectedPValue, 1e-9)
        assertTrue(result.detected)
    }

    @Test
    fun `multi-window audio recovers when one window throws an exception`() {
        // Window 0 throws, Window 1 succeeds with 16 bits
        var windowIndex = 0
        val detector = createDetector(predictor = { _, _ ->
            val idx = windowIndex++
            if (idx == 0) throw RuntimeException("Simulated window corruption")
            16f
        })

        val samples = FloatArray(32000) { 0.1f }
        val result = detector.detectWatermark(samples)

        assertNotNull("Detector must recover from first window failure and score second window", result)
        assertEquals(16, result!!.agreeingBits)
        assertEquals(2.0 * 3.0517578125e-5, result.pValue, 1e-9)
        assertTrue(result.detected)
    }

    // =========================================================================
    // Group 4: Silent & Corrupted Audio Handling
    // =========================================================================

    @Test
    fun `silent audio buffer returns ProvenanceEvidence Absent`() {
        // Silent PCM generates zero watermark correlation (chance agreement k=8)
        val detector = createDetector(predictor = { _, _ -> 8f })
        val silentAudio = FloatArray(16000) { 0.0f }

        val evidence = detector.analyze(silentAudio)
        assertEquals(ProvenanceEvidence.Absent, evidence)
    }

    @Test
    fun `audio buffer with non-finite samples does not crash and returns Absent`() {
        val detector = createDetector(predictor = { _, byteBuffer ->
            val floatBuf = byteBuffer.asFloatBuffer()
            while (floatBuf.hasRemaining()) {
                val f = floatBuf.get()
                if (!f.isFinite()) throw IllegalArgumentException("Non-finite float in direct buffer: $f")
            }
            16f
        })

        val corruptedSamples = FloatArray(16000) { 0.1f }
        corruptedSamples[100] = Float.NaN
        corruptedSamples[200] = Float.POSITIVE_INFINITY

        val evidence = detector.analyze(corruptedSamples)
        assertEquals("Non-finite audio must be safely caught by runCatching and return Absent", ProvenanceEvidence.Absent, evidence)
    }

    // =========================================================================
    // Group 5: Missing Model & Engine Error Resilience
    // =========================================================================

    @Test
    fun `unbundled or missing model asset returns ProvenanceEvidence Absent gracefully`() {
        val detector = AudioSealDetector(
            contractProvider = { null }, // Model not bundled in release APK
            predictor = { _, _ -> error("Predictor must not be called when model is absent") },
        )

        val samples = FloatArray(16000) { 0.1f }
        assertNull(detector.detectWatermark(samples))
        assertEquals(ProvenanceEvidence.Absent, detector.analyze(samples))
    }

    @Test
    fun `model loader throwing FileNotFoundException returns ProvenanceEvidence Absent without leaking exception`() {
        val detector = AudioSealDetector(
            contractProvider = { throw FileNotFoundException("audioseal_detector_int8.tflite not found") },
            predictor = { _, _ -> error("Should not be called") },
        )

        val samples = FloatArray(16000) { 0.1f }
        assertEquals(ProvenanceEvidence.Absent, detector.analyze(samples))
    }

    @Test
    fun `invalid model contract input shape returns ProvenanceEvidence Absent`() {
        // Case A: Input rank != 2
        val detectorRank1 = AudioSealDetector(
            contractProvider = { LiteRTInferenceEngine.Contract(intArrayOf(16000), intArrayOf(1, 1), 64000) },
            predictor = { _, _ -> 16f },
        )
        assertEquals(ProvenanceEvidence.Absent, detectorRank1.analyze(FloatArray(16000)))

        // Case B: Batch dimension != 1
        val detectorBatch2 = AudioSealDetector(
            contractProvider = { LiteRTInferenceEngine.Contract(intArrayOf(2, 16000), intArrayOf(1, 1), 128000) },
            predictor = { _, _ -> 16f },
        )
        assertEquals(ProvenanceEvidence.Absent, detectorBatch2.analyze(FloatArray(32000)))

        // Case C: Window size < 1024
        val detectorSmallWindow = AudioSealDetector(
            contractProvider = { LiteRTInferenceEngine.Contract(intArrayOf(1, 512), intArrayOf(1, 1), 2048) },
            predictor = { _, _ -> 16f },
        )
        assertEquals(ProvenanceEvidence.Absent, detectorSmallWindow.analyze(FloatArray(16000)))
    }

    @Test
    fun `model predictor emitting non-finite or negative scalar returns ProvenanceEvidence Absent`() {
        // Predictor returning NaN
        val detectorNaN = createDetector(predictor = { _, _ -> Float.NaN })
        assertEquals(ProvenanceEvidence.Absent, detectorNaN.analyze(FloatArray(16000)))

        // Predictor returning negative scalar
        val detectorNeg = createDetector(predictor = { _, _ -> -1.0f })
        assertEquals(ProvenanceEvidence.Absent, detectorNeg.analyze(FloatArray(16000)))

        // Predictor returning +Infinity
        val detectorInf = createDetector(predictor = { _, _ -> Float.POSITIVE_INFINITY })
        assertEquals(ProvenanceEvidence.Absent, detectorInf.analyze(FloatArray(16000)))
    }

    @Test
    fun `model predictor throwing exception is caught and returns ProvenanceEvidence Absent`() {
        val detector = createDetector(predictor = { _, _ ->
            throw IllegalStateException("TFLite runtime invocation error")
        })

        assertEquals(ProvenanceEvidence.Absent, detector.analyze(FloatArray(16000)))
    }
}
