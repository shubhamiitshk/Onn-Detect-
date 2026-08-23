package com.onndetect

import com.onndetect.detection.visual.FFTUtil
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.abs

class FFTUtilTest {

    @Test
    fun `impulse input yields flat spectrum`() {
        val n = 8
        val re = DoubleArray(n).also { it[0] = 1.0 }
        val im = DoubleArray(n)
        FFTUtil.transform(re, im)
        for (i in 0 until n) {
            assertEquals(1.0, abs(complex(re[i], im[i])), 1e-9)
        }
    }

    @Test
    fun `single tone peaks at expected bin`() {
        val n = 64
        val bin = 7
        val re = DoubleArray(n) { i -> cos(2.0 * PI * bin * i / n) }
        val im = DoubleArray(n)
        FFTUtil.transform(re, im)

        // A real cosine splits energy equally between bin k and its mirror
        // n-k; the two largest magnitudes must be exactly those bins.
        val mags = DoubleArray(n) { kotlin.math.abs(complex(re[it], im[it])) }
        val top2 = (0 until n).sortedByDescending { mags[it] }.take(2).sorted()
        assertEquals(listOf(bin, n - bin), top2)
    }

    @Test
    fun `constant signal has energy only at DC`() {
        val n = 16
        val re = DoubleArray(n) { 3.0 }
        val im = DoubleArray(n)
        FFTUtil.transform(re, im)
        assertEquals(n * 3.0, abs(complex(re[0], im[0])), 1e-9)
        for (i in 1 until n) {
            assertTrue(abs(complex(re[i], im[i])) < 1e-9)
        }
    }

    @Test
    fun `2D power spectrum of constant image is DC-only after shift`() {
        val n = 8
        val pixels = DoubleArray(n * n) { 0.5 }
        val ps = FFTUtil.powerSpectrum2D(pixels, n)
        val center = ps[(n / 2) * n + n / 2]
        // DC magnitude^2 = (n*n*0.5)^2
        assertEquals((n * n * 0.5) * (n * n * 0.5), center, 1e-6)
        var offCenterSum = 0.0
        for (i in ps.indices) if (i != (n / 2) * n + n / 2) offCenterSum += ps[i]
        assertEquals(0.0, offCenterSum, 1e-6)
    }

    private fun complex(re: Double, im: Double): Double =
        kotlin.math.sqrt(re * re + im * im)
}
