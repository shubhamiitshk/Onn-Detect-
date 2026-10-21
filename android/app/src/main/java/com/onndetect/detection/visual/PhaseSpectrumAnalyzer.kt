package com.onndetect.detection.visual

import android.graphics.Bitmap
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Phase-only spectral descriptors.
 *
 * ## Why this exists
 *
 * The shipped `fft_score` measures the slope of an azimuthally averaged
 * POWER spectrum. JPEG quantization destroys high-frequency amplitude, so
 * that slope collapses under re-encoding: the shipped feature's standalone
 * AUC is 0.5013 - chance - and it is used in 26 of the fusion tree's
 * splits.
 *
 * JPEG leaves the PHASE angles intact. Li et al., CVPR 2026 ("Detecting
 * Compressed AI-Generated Images via Phase Spectrum Robustness") show that
 * a phase-only reconstruction is stable under compression.
 *
 * ## Two details that are easy to get wrong
 *
 *  * the DC term must be moved to the centre with an FFT shift before any
 *    radial statistic is computed, otherwise the radii are centred on the
 *    Nyquist corner rather than on DC;
 *  * the phase must be turned back into an IMAGE (the phase-only
 *    reconstruction) rather than reduced to `cos(phase)` directly, because a
 *    translated image acquires a linear phase ramp that `cos(phase)` does
 *    not tolerate.
 *
 * This is the Kotlin counterpart of
 * `ml/src/meta_classifier/feature_extractor.py::phase_spectrum_features`.
 */
class PhaseSpectrumAnalyzer {

    companion object {
        /** Analysis resolution. Matches the Python implementation. */
        const val SIZE = 256
        /** Inner radius of the high-frequency band, as a fraction of Nyquist. */
        const val HF_INNER_FRACTION = 0.25
    }

    data class PhaseResult(val kurtosis: Float, val highFrequencyRatio: Float)

    /**
     * Compute the phase-only descriptors for [bitmap].
     *
     * Returns `null` when the image is too small to analyse.
     */
    fun analyze(bitmap: Bitmap, checkStop: () -> Unit = {}): PhaseResult? {
        checkStop()
        if (bitmap.isRecycled || bitmap.width < 2 || bitmap.height < 2) return null
        val size = SIZE
        val scaled = Bitmap.createScaledBitmap(bitmap, size, size, true)
        val pixels = IntArray(size * size)
        try {
            scaled.getPixels(pixels, 0, size, 0, 0, size, size)
        } finally {
            if (scaled !== bitmap) scaled.recycle()
        }

        // Grey, mean-removed.
        val gray = DoubleArray(size * size)
        var mean = 0.0
        for (i in pixels.indices) {
            val c = pixels[i]
            val g = ((c shr 16 and 255) * 299 + (c shr 8 and 255) * 587 + (c and 255) * 114) / 1000.0 / 255.0
            gray[i] = g
            mean += g
        }
        mean /= gray.size
        for (i in gray.indices) gray[i] -= mean

        checkStop()

        // 2D FFT (row passes then column passes), then the unit-magnitude
        // phase of the spectrum.
        val re = gray.copyOf()
        val im = DoubleArray(gray.size)
        val rowRe = DoubleArray(size)
        val rowIm = DoubleArray(size)
        for (y in 0 until size) {
            for (x in 0 until size) {
                rowRe[x] = re[y * size + x]
                rowIm[x] = im[y * size + x]
            }
            FFTUtil.transform(rowRe, rowIm)
            for (x in 0 until size) {
                re[y * size + x] = rowRe[x]
                im[y * size + x] = rowIm[x]
            }
        }
        val colRe = DoubleArray(size)
        val colIm = DoubleArray(size)
        for (x in 0 until size) {
            for (y in 0 until size) {
                colRe[y] = re[y * size + x]
                colIm[y] = im[y * size + x]
            }
            FFTUtil.transform(colRe, colIm)
            for (y in 0 until size) {
                re[y * size + x] = colRe[y]
                im[y * size + x] = colIm[y]
            }
        }

        val unitRe = DoubleArray(gray.size)
        val unitIm = DoubleArray(gray.size)
        for (i in re.indices) {
            val magnitude = sqrt(re[i] * re[i] + im[i] * im[i])
            if (magnitude < 1e-12) {
                unitRe[i] = 1.0
                unitIm[i] = 0.0
            } else {
                unitRe[i] = re[i] / magnitude
                unitIm[i] = im[i] / magnitude
            }
        }

        checkStop()

        // Inverse FFT of the unit phase: the phase-only reconstruction.
        val inv = FFTUtil.inverse2d(unitRe, unitIm, size)

        // The same reconstruction with only the high-frequency band kept.
        // The DC term is shifted to the centre FIRST so the radii are
        // measured from DC.
        val shiftedRe = DoubleArray(gray.size)
        val shiftedIm = DoubleArray(gray.size)
        val half = size / 2
        for (y in 0 until size) {
            val sy = (y + half) % size
            for (x in 0 until size) {
                val sx = (x + half) % size
                shiftedRe[sy * size + sx] = unitRe[y * size + x]
                shiftedIm[sy * size + sx] = unitIm[y * size + x]
            }
        }
        val hfRe = DoubleArray(gray.size)
        val hfIm = DoubleArray(gray.size)
        val cy = size / 2
        val cx = size / 2
        val inner = HF_INNER_FRACTION * (size / 2.0)
        for (y in 0 until size) {
            for (x in 0 until size) {
                val radius = sqrt((x - cx).toDouble() * (x - cx) + (y - cy).toDouble() * (y - cy))
                if (radius >= inner) {
                    hfRe[y * size + x] = shiftedRe[y * size + x]
                    hfIm[y * size + x] = shiftedIm[y * size + x]
                }
            }
        }
        val hfInv = FFTUtil.inverse2d(hfRe, hfIm, size)

        // Kurtosis of the phase-only reconstruction and the high-frequency
        // energy ratio.
        var m2 = 0.0
        var m4 = 0.0
        for (v in inv) {
            m2 += v * v
            m4 += v * v * v * v
        }
        m2 /= inv.size
        m4 /= inv.size
        val kurtosis = (m4 / (m2 * m2 + 1e-12)).toFloat()

        var sdAll = 0.0
        for (v in inv) sdAll += (v - 0.0) * (v - 0.0)
        sdAll = sqrt(sdAll / inv.size)
        var sdHf = 0.0
        for (v in hfInv) sdHf += v * v
        sdHf = sqrt(sdHf / hfInv.size)
        val ratio = (sdHf / (sdAll + 1e-12)).toFloat()

        checkStop()
        if (!kurtosis.isFinite() || !ratio.isFinite()) return null
        return PhaseResult(kurtosis, ratio)
    }
}
