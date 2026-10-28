package com.onndetect.detection.visual

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ForensicFeatureHeadTest {

    @Test
    fun `predict computes exact affine sigmoid score and respects threshold`() {
        val dim = 4
        val weights = floatArrayOf(0.5f, -0.5f, 1.0f, -1.0f)
        val bias = 0.0f
        val threshold = 0.50f

        val head = ForensicFeatureHead(
            modelId = "google/embeddinggemma-2-test",
            featureDim = dim,
            weights = weights,
            bias = bias,
            threshold = threshold,
            diagnosticOnly = true,
        )

        // Balanced vector: dot product = 0.5 - 0.5 + 1.0 - 1.0 = 0.0 -> prob = 0.50
        val featuresZero = floatArrayOf(1f, 1f, 1f, 1f)
        val resZero = head.predict(featuresZero)
        assertEquals(0.50f, resZero.probability, 1e-5f)
        assertTrue(resZero.isSynthetic)
        assertTrue(resZero.diagnosticOnly)

        // Positive vector: dot product = 0.5*2 - 0 = 1.0 -> prob = 1 / (1 + exp(-1)) ~= 0.731058
        val featuresPos = floatArrayOf(2f, 0f, 0f, 0f)
        val resPos = head.predict(featuresPos)
        assertTrue(resPos.probability > 0.70f)
        assertTrue(resPos.isSynthetic)

        // Negative vector: dot product = -1.0 -> prob = 1 / (1 + exp(1)) ~= 0.26894
        val featuresNeg = floatArrayOf(0f, 2f, 0f, 0f)
        val resNeg = head.predict(featuresNeg)
        assertTrue(resNeg.probability < 0.30f)
        assertFalse(resNeg.isSynthetic)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `rejects mismatched feature dimensions`() {
        val head = ForensicFeatureHead(
            modelId = "test",
            featureDim = 4,
            weights = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f),
            bias = 0.0f,
            threshold = 0.5f,
        )
        head.predict(floatArrayOf(1f, 2f)) // 2 elements instead of 4
    }
}
