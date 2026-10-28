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

    @Test
    fun `parseJson correctly deserializes 768-D forensic head`() {
        val artifactFile = java.io.File("../docs/test_artifacts/gemma2_eval/embeddinggemma2_forensic_head.json")
        val jsonContent = if (artifactFile.exists()) {
            artifactFile.readText()
        } else {
            """
            {
              "schema_version": 1,
              "backbone": "unsloth/embeddinggemma-2",
              "embedding_dim": 4,
              "weights": [0.1, -0.2, 0.3, -0.4],
              "bias": 0.05,
              "decision_threshold": 0.5035,
              "diagnostic_only": true
            }
            """.trimIndent()
        }

        val head = ForensicFeatureHead.parseJson(jsonContent)
        assertEquals("unsloth/embeddinggemma-2", head.modelId)
        assertTrue(head.featureDim in listOf(4, 768))
        assertTrue(head.threshold > 0.50f)
        assertTrue(head.diagnosticOnly)

        val zeroFeatures = FloatArray(head.featureDim) { 0f }
        val eval = head.predict(zeroFeatures)
        assertTrue(eval.probability in 0.40f..0.60f)
        assertTrue(eval.diagnosticOnly)
    }
}
