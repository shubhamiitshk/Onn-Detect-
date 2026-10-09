package com.onndetect

import com.onndetect.detection.visual.KLCooccurrenceAnalyzer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the shared BT.601 grayscale convention used by both fusion inputs.
 *
 * Kl_score and fft_score both enter the ten-slot feature vector, and both
 * derived their grayscale from the same bitmap. They previously used DIFFERENT
 * arithmetic: Kl used integer weights while the FFT path used fractional double
 * coefficients. Those constants are not exact binary fractions, so the double
 * sum lands just below the integer for roughly 2e-4 of RGB triples and truncates
 * one lower - e.g. (130,182,64) yields 152.99999999999997 and therefore 152,
 * where the integer form gives 153.
 *
 * Bitmap cannot be constructed under plain JUnit, so the production conversion
 * cannot be exercised here. These tests pin the arithmetic both analyzers must
 * perform, and `scripts/check_android_consistency.py` separately asserts the
 * source text matches across both platforms and Python. Neither is a native
 * bitmap-decoding parity measurement.
 */
class GrayscaleConventionTest {

    private val analyzer = KLCooccurrenceAnalyzer()

    @Test
    fun `fractional double weights disagree with integer weights on real triples`() {
        // This is WHY both analyzers were changed to integer weights. The two
        // forms are NOT equivalent: the fractional constants are not exact
        // binary fractions, so the double sum lands just below the integer for
        // a small but non-zero fraction of the colour cube and truncates one
        // lower. A previous version of this test asserted they agreed and
        // therefore could not detect the divergence it was meant to guard.
        var disagreements = 0
        val samples = 200_000
        val rng = java.util.Random(20261008L)
        repeat(samples) {
            val r = rng.nextInt(256)
            val g = rng.nextInt(256)
            val b = rng.nextInt(256)
            if ((r * 299 + g * 587 + b * 114) / 1000 != (0.299 * r + 0.587 * g + 0.114 * b).toInt()) {
                disagreements++
            }
        }
        val rate = disagreements.toDouble() / samples
        assertTrue("expected a measurable divergence rate, got $rate", rate > 1e-4)
        assertTrue("divergence should stay rare, got $rate", rate < 1e-3)
    }

    @Test
    fun `known fractional-double divergence is pinned`() {
        // If this ever stops holding, the convention changed and every stored
        // calibration is suspect.
        assertEquals(152, (0.299 * 130 + 0.587 * 182 + 0.114 * 64).toInt())
        assertEquals(153, (130 * 299 + 182 * 587 + 64 * 114) / 1000)
    }

    @Test
    fun `luma stays inside uint8 range for the full colour cube`() {
        var minimum = 255
        var maximum = 0
        for (r in 0..255 step 17) {
            for (g in 0..255 step 17) {
                for (b in 0..255 step 17) {
                    val luma = (r * 299 + g * 587 + b * 114) / 1000
                    if (luma < minimum) minimum = luma
                    if (luma > maximum) maximum = luma
                }
            }
        }
        assertEquals(0, minimum)
        assertEquals(255, maximum)
    }

    @Test
    fun `single column grayscale has no adjacent pairs and must be refused`() {
        assertEquals(null, analyzer.diffHistogramFromGray(IntArray(16) { 100 }, 1, 16))
        assertEquals(null, analyzer.diffHistogramFromGray(IntArray(8) { 100 }, 0, 8))
    }
}