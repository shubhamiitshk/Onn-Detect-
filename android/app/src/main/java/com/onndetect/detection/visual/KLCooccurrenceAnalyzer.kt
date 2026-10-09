package com.onndetect.detection.visual

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.ln

/** Horizontal adjacent-pixel difference histogram descriptor, not proof of AI origin. */
class KLCooccurrenceAnalyzer(private val context: Context? = null) {
    
    companion object {
        // Fallback prior; superseded at runtime by the calibrated baseline in
        // assets/constants/natural_diff_histogram.json when available.
        // 21 bins covering pixel differences from -10 to +10.
        val NATURAL_DIFF_HISTOGRAM = floatArrayOf(
            0.005f, 0.007f, 0.012f, 0.018f, 0.025f, 0.038f, 0.055f, 0.082f, 0.120f, 0.150f, 
            0.180f, // Center (0 difference)
            0.150f, 0.120f, 0.082f, 0.055f, 0.038f, 0.025f, 0.018f, 0.012f, 0.007f, 0.005f
        ).apply {
            // Ensure exact sum to 1.0 for probability distributions
            val sum = this.sum()
            for (i in this.indices) { this[i] /= sum }
        }
    }

    fun analyze(bitmap: Bitmap): Float {
        val baseline = context?.let { CalibrationStore.naturalHistogram(it) }
            ?.takeIf { it.size == 21 }
            ?.also(::normalizeInPlace)
            ?: NATURAL_DIFF_HISTOGRAM

        // 1. Get Grayscale Pixels
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val gray = IntArray(width * height)
        for (i in pixels.indices) {
            val c = pixels[i]
            // Standard luminosity formula
            // Integer BT.601 luma, shared with FFTSpectralAnalyzer and with
            // meta_classifier/feature_extractor.py to_gray_u8(). The Int
            // division truncates; that truncation IS the convention.
            gray[i] = ((Color.red(c) * 299 + Color.green(c) * 587 + Color.blue(c) * 114) / 1000)
        }

        return scoreFromGray(gray, width, height, baseline)
    }

    /** Pure, unit-testable: clamped adjacent-diff histogram (normalized, eps). */
    fun diffHistogramFromGray(gray: IntArray, width: Int, height: Int): FloatArray? {
        if (width < 2 || height < 1 || width.toLong() * height > gray.size) return null
        val histogram = FloatArray(21)
        var validPixels = 0
        // Out-of-range diffs are CLAMPED into the edge bins - this matches
        // ml/src/meta_classifier/feature_extractor.py exactly.
        for (y in 0 until height) {
            for (x in 0 until width - 1) {
                val diff = (gray[y * width + x] - gray[y * width + x + 1])
                    .coerceIn(-10, 10)
                histogram[diff + 10]++
                validPixels++
            }
        }
        if (validPixels == 0) return null
        for (i in histogram.indices) histogram[i] = histogram[i] / validPixels + 1e-6f
        val sum = histogram.sum()
        for (i in histogram.indices) histogram[i] /= sum
        return histogram
    }

    /** Pure, unit-testable core: histogram -> KL vs baseline -> [0,1] score. */
    fun scoreFromGray(
        gray: IntArray,
        width: Int,
        height: Int,
        baseline: FloatArray,
    ): Float {
        require(baseline.size == 21) { "baseline must have 21 bins" }
        val histogram = diffHistogramFromGray(gray, width, height) ?: error("Insufficient pixel differences")
        return scoreFromHistogram(histogram, baseline)
    }

    fun scoreFromHistogram(histogram: FloatArray, baseline: FloatArray): Float {
        require(histogram.size == 21 && baseline.size == 21)
        require(histogram.all { it.isFinite() && it >= 0f } && histogram.sum() > 0f && baseline.all { it.isFinite() && it > 0f }) { "Invalid histogram" }
        // Empirical mapping: KL of 0.3 saturates the score.
        return (klDivergence(histogram, baseline) / 0.3f).coerceIn(0.0f, 1.0f)
    }

    private fun klDivergence(p: FloatArray, q: FloatArray): Float {
        var kl = 0f
        for (i in p.indices) if (p[i] > 0f) kl += p[i] * ln(p[i] / q[i])
        return kl
    }

    private fun normalizeInPlace(baseline: FloatArray) {
        var sum = 0f
        for (v in baseline) sum += v
        require(sum.isFinite() && sum > 0f && baseline.all { it.isFinite() && it > 0f })
        if (sum > 0f) for (i in baseline.indices) baseline[i] /= sum
    }
}
