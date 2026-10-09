package com.onndetect.detection.visual

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Zero-dependency radix-2 Cooley-Tukey FFT used by the spectral forensics
 * analyzers. All methods operate in-place on real/imaginary pairs.
 */
object FFTUtil {

    /** In-place complex FFT. Length must be a power of two. */
    fun transform(real: DoubleArray, imag: DoubleArray) {
        val n = real.size
        require(n == imag.size && n and (n - 1) == 0 && n > 0) { "length must be a power of two" }

        // Bit-reversal permutation
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j or bit
            if (i < j) {
                val tr = real[i]; real[i] = real[j]; real[j] = tr
                val ti = imag[i]; imag[i] = imag[j]; imag[j] = ti
            }
        }

        // Butterfly passes
        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wStepRe = cos(ang)
            val wStepIm = sin(ang)
            var start = 0
            while (start < n) {
                var wRe = 1.0
                var wIm = 0.0
                val half = len shr 1
                for (k in 0 until half) {
                    val iEven = start + k
                    val iOdd = iEven + half
                    val uRe = real[iEven]
                    val uIm = imag[iEven]
                    val vRe = real[iOdd] * wRe - imag[iOdd] * wIm
                    val vIm = real[iOdd] * wIm + imag[iOdd] * wRe
                    real[iEven] = uRe + vRe
                    imag[iEven] = uIm + vIm
                    real[iOdd] = uRe - vRe
                    imag[iOdd] = uIm - vIm
                    val nextRe = wRe * wStepRe - wIm * wStepIm
                    wIm = wRe * wStepIm + wIm * wStepRe
                    wRe = nextRe
                }
                start += len
            }
            len = len shl 1
        }
    }

    /**
     * 2D FFT power spectrum of an n x n grayscale image (row-major), returned
     * fftshift-ed so DC sits at the center, matching the azimuthal average.
     */
    fun powerSpectrum2D(pixels: DoubleArray, n: Int): DoubleArray {
        require(n > 0 && n and (n - 1) == 0 && n.toLong() * n == pixels.size.toLong() && pixels.all { it.isFinite() })
        val re = pixels.copyOf()
        val im = DoubleArray(n * n)

        for (y in 0 until n) {
            val off = y * n
            transform(re, im, off, n)
        }
        // Column transforms via strided access
        val colRe = DoubleArray(n)
        val colIm = DoubleArray(n)
        for (x in 0 until n) {
            for (y in 0 until n) {
                colRe[y] = re[y * n + x]
                colIm[y] = im[y * n + x]
            }
            transform(colRe, colIm)
            for (y in 0 until n) {
                re[y * n + x] = colRe[y]
                im[y * n + x] = colIm[y]
            }
        }

        // |F|^2 with fftshift
        val spectrum = DoubleArray(n * n)
        val half = n / 2
        for (y in 0 until n) {
            val sy = (y + half) % n
            for (x in 0 until n) {
                val sx = (x + half) % n
                val r = re[y * n + x]
                val i = im[y * n + x]
                spectrum[sy * n + sx] = r * r + i * i
            }
        }
        return spectrum
    }

    private fun transform(re: DoubleArray, im: DoubleArray, offset: Int, length: Int) {
        val tmpRe = re.copyOfRange(offset, offset + length)
        val tmpIm = im.copyOfRange(offset, offset + length)
        transform(tmpRe, tmpIm)
        System.arraycopy(tmpRe, 0, re, offset, length)
        System.arraycopy(tmpIm, 0, im, offset, length)
    }
}
