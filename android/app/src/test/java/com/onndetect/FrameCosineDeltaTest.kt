package com.onndetect

import com.onndetect.detection.temporal.FrameCosineDelta
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FrameCosineDeltaTest {

    private fun solidPixels(gray: Int): Pair<IntArray, Int> {
        val size = 32
        val px = IntArray(size * size) { gray shl 16 or (gray shl 8) or gray }
        return px to size
    }

    /** Horizontal gradient - gives frames a DIRECTION, unlike solids. */
    private fun gradientPixels(inverted: Boolean): Pair<IntArray, Int> {
        val size = 32
        val px = IntArray(size * size) { i ->
            val x = i % size
            val v = if (inverted) 255 - x * 8 else x * 8
            (v.coerceIn(0, 255)).let { g -> g shl 16 or (g shl 8) or g }
        }
        return px to size
    }

    @Test
    fun `identical frames have cosine one`() {
        val (px, w) = solidPixels(100)
        val a = FrameCosineDelta.descriptorFromPixels(px, w, w)
        assertEquals(1f, FrameCosineDelta.cosine(a, a), 1e-5f)
    }

    @Test
    fun `perfectly frozen sequence saturates anomaly`() {
        assertEquals(1f, FrameCosineDelta.anomalyScore(listOf(1f, 1f, 1f)), 1e-4f)
        // Just inside the freeze band maps proportionally
        assertEquals(
            0.98f,
            FrameCosineDelta.anomalyScore(listOf(0.9999f, 0.9999f)),
            1e-3f,
        )
    }

    @Test
    fun `orthogonal descriptors have zero cosine`() {
        val n = FrameCosineDelta.GRID * FrameCosineDelta.GRID
        val a = FloatArray(n).also { it[0] = 1f }
        val b = FloatArray(n).also { it[1] = 1f }
        assertEquals(0f, FrameCosineDelta.cosine(a, b), 1e-6f)
    }

    @Test
    fun `natural-range dynamics score low`() {
        val score = FrameCosineDelta.anomalyScore(listOf(0.97f, 0.96f, 0.975f, 0.965f))
        assertTrue("natural motion should be low, was $score", score < 0.35f)
    }

    @Test
    fun `flickering sequence scores high`() {
        val score = FrameCosineDelta.anomalyScore(listOf(0.55f, 0.62f, 0.50f))
        assertEquals(1f, score, 1e-4f)
    }

    @Test
    fun `opposite gradients are anti-correlated`() {
        val (p1, w) = gradientPixels(false)
        val (p2, _) = gradientPixels(true)
        val d1 = FrameCosineDelta.descriptorFromPixels(p1, w, w)
        val d2 = FrameCosineDelta.descriptorFromPixels(p2, w, w)
        assertTrue(
            "expected strong decorrelation, got ${FrameCosineDelta.cosine(d1, d2)}",
            FrameCosineDelta.cosine(d1, d2) < 0.99f,
        )
    }
}
