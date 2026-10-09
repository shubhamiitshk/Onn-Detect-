package com.onndetect

import com.onndetect.detection.visual.KLCooccurrenceAnalyzer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class KLCooccurrenceTest {

    private val analyzer = KLCooccurrenceAnalyzer()

    private fun baseline(): FloatArray = floatArrayOf(
        0.005f, 0.007f, 0.012f, 0.018f, 0.025f, 0.038f, 0.055f, 0.082f,
        0.120f, 0.150f, 0.180f, 0.150f, 0.120f, 0.082f, 0.055f, 0.038f,
        0.025f, 0.018f, 0.012f, 0.007f, 0.005f,
    ).let { arr ->
        FloatArray(arr.size) { i -> arr[i] / arr.sum() }
    }

    @Test
    fun `histogram identical to baseline gives zero KL`() {
        val base = baseline()
        val score = analyzer.scoreFromHistogram(base.copyOf(), base)
        assertTrue("identical distributions must score ~0, was $score", score < 1e-3f)
    }

    @Test
    fun `shifted distribution gives positive KL and saturates`() {
        val base = baseline()
        // All mass in one far-from-center bin: huge divergence
        val spike = FloatArray(21) { i ->
            if (i == 20) 1f - 21 * 1e-6f else 1e-6f
        }
        assertEquals(1.0f, analyzer.scoreFromHistogram(spike, base), 1e-4f)

        // Mild perturbation: small but positive KL
        val mild = base.copyOf()
        mild[10] += 0.02f
        mild[9] -= 0.01f
        mild[11] -= 0.01f
        val s2 = analyzer.scoreFromHistogram(mild, base)
        assertTrue("mild shift should be positive, was $s2", s2 > 0f)
        assertTrue("mild shift should not saturate, was $s2", s2 < 1f)
    }

    @Test
    fun `flat image concentrates mass in center bin and scores high`() {
        val w = 16
        val h = 16
        val gray = IntArray(w * h) { 77 } // every diff == 0
        val hist = analyzer.diffHistogramFromGray(gray, w, h)
        assertNotNull(hist)
        // Center bin (index 10) holds essentially all mass; edge bins ~eps
        assertTrue("center bin should dominate, got ${hist!![10]}", hist[10] > 0.95f)
        val score = analyzer.scoreFromGray(gray, w, h, baseline())
        assertTrue("flat image should score high, was $score", score > 0.5f)
    }

    @Test
    fun `extreme diffs are clamped into edge bins`() {
        val w = 8
        val h = 8
        // Alternating 255/0 -> |diff| = 255, clamped to +/-10 edge bins
        val gray = IntArray(w * h) { i ->
            if ((i % w + i / w) % 2 == 0) 255 else 0
        }
        val hist = analyzer.diffHistogramFromGray(gray, w, h)!!
        assertTrue("no center mass expected, got ${hist[10]}", hist[10] < 1e-4f)
        // All mass sits on the two edge bins (signs alternate by column)
        assertEquals(1f, hist[0] + hist[20], 1e-3f)
        assertEquals(1.0f, analyzer.scoreFromGray(gray, w, h, baseline()))
    }

    @Test
    fun `histogram sums to one`() {
        val w = 12
        val h = 12
        val gray = IntArray(w * h) { it * 7 % 256 }
        val hist = analyzer.diffHistogramFromGray(gray, w, h)!!
        assertEquals(1f, hist.sum(), 1e-5f)
    }

    @Test(expected = IllegalStateException::class)
    fun `degenerate inputs are unavailable`() {
        val base = baseline()
        assertNull(analyzer.diffHistogramFromGray(IntArray(4), 1, 4))
        analyzer.scoreFromGray(IntArray(4), 1, 4, base)
    }
    @Test(expected = IllegalArgumentException::class)
    fun `nonfinite histogram is rejected`() {
        analyzer.scoreFromHistogram(baseline().apply { this[0] = Float.NaN }, baseline())
    }
    @Test(expected = IllegalArgumentException::class)
    fun `zero baseline is rejected`() { analyzer.scoreFromHistogram(baseline(), FloatArray(21)) }
    @Test
    fun `zero probability bins do not generate NaN`() {
        val spike = FloatArray(21).apply { this[10] = 1f }
        assertTrue(analyzer.scoreFromHistogram(spike, baseline()).isFinite())
    }
}
