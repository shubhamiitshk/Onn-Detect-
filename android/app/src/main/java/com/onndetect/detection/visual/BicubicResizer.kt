package com.onndetect.detection.visual

import android.graphics.Bitmap
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Separable bicubic resampler matching Pillow's `Image.Resampling.BICUBIC`.
 *
 * ## Why this exists
 *
 * The Engine C frontend contract is
 * `clip-appearance-bicubic224-clipnorm-meanstd-float32`, and the Python
 * training/calibration path resizes with `Image.Resampling.BICUBIC`
 * (`ml/src/data/image_frontend.py:53`, `semantic_image_tensor`).
 *
 * The Android side used `Bitmap.createScaledBitmap(bitmap, size, size, true)`,
 * which is **bilinear** - the same filter Engine A uses. So the device fed
 * Engine C a bilinear tensor while every calibration and threshold in the
 * release was computed from a bicubic tensor. That is a train/serve skew
 * inside a contract the whole release hashes bind to, and it is exactly the
 * kind of difference the docstring at
 * `detection/visual/VfmProbeAnalyzer.kt` claimed did not exist.
 *
 * Android's `Bitmap` has no bicubic filter, so this implements the Keys
 * cubic convolution (a = -0.5) with support 2.0 - the same kernel and the
 * same scale-aware filter support Pillow uses for downscaling.
 *
 * ## Scope
 *
 * Only RGB is resampled. Both analyzers consume R, G and B only
 * (`SRMForensicAnalyzer.kt:30-34`, `VfmProbeAnalyzer.kt`), and the
 * input bitmap has already had its alpha flattened onto white upstream by
 * `ScannerViewModel.decodeScaled`. Output is therefore opaque ARGB_8888.
 */
object BicubicResizer {

    /** Keys cubic convolution parameter. Matches Pillow. */
    private const val A = -0.5

    /** Filter support radius. Matches Pillow's BICUBIC filter_support. */
    private const val SUPPORT = 2.0

    /** Keys cubic convolution kernel. */
    private fun cubic(x: Double): Double {
        val ax = abs(x)
        return when {
            ax <= 1.0 -> (A + 2.0) * ax * ax * ax - (A + 3.0) * ax * ax + 1.0
            ax < 2.0 -> A * ax * ax * ax - 5.0 * A * ax * ax + 8.0 * A * ax - 4.0 * A
            else -> 0.0
        }
    }

    /**
     * Per-output-pixel source weights for one axis.
     *
     * @return first contributing source index per output pixel, and the
     *   normalized weights per output pixel.
     */
    private fun axisWeights(inSize: Int, outSize: Int): Pair<IntArray, Array<DoubleArray>> {
        val scale = inSize.toDouble() / outSize.toDouble()
        // Pillow widens the filter support when downscaling so the kernel
        // still covers the full source footprint of one output pixel.
        val filterScale = max(scale, 1.0)
        val support = SUPPORT * filterScale
        val starts = IntArray(outSize)
        val weights = Array(outSize) { DoubleArray(0) }
        for (o in 0 until outSize) {
            val center = (o + 0.5) * scale
            var xmin = floor(center - support + 0.5).toInt()
            var xmax = ceil(center + support - 0.5).toInt()
            if (xmin < 0) xmin = 0
            if (xmax > inSize - 1) xmax = inSize - 1
            val n = xmax - xmin + 1
            val w = DoubleArray(n)
            var sum = 0.0
            for (k in 0 until n) {
                val v = cubic((xmin + k - center + 0.5) / filterScale)
                w[k] = v
                sum += v
            }
            if (sum != 0.0) {
                for (k in 0 until n) w[k] /= sum
            }
            starts[o] = xmin
            weights[o] = w
        }
        return starts to weights
    }

    /**
     * Resize [src] to [dstWidth] x [dstHeight] with bicubic filtering.
     *
     * Returns [src] unchanged when it is already the requested size.
     */
    fun resize(src: Bitmap, dstWidth: Int, dstHeight: Int): Bitmap {
        require(src.width > 0 && src.height > 0 && dstWidth > 0 && dstHeight > 0) { "Invalid resize geometry" }
        if (src.width == dstWidth && src.height == dstHeight) return src
        val srcW = src.width
        val srcH = src.height
        val srcPixels = IntArray(srcW * srcH)
        src.getPixels(srcPixels, 0, srcW, 0, 0, srcW, srcH)
        val out = resizePixels(srcPixels, srcW, srcH, dstWidth, dstHeight)
        return Bitmap.createBitmap(out, dstWidth, dstHeight, Bitmap.Config.ARGB_8888)
    }

    /**
     * Bicubic resize over a raw ARGB pixel array.
     *
     * Kept separate from the Bitmap overload so it can be unit tested on
     * the JVM against a Pillow-generated reference fixture. The JVM unit
     * test source set has no Robolectric, so anything that touches
     * `android.graphics.Bitmap` cannot be asserted there.
     */
    fun resizePixels(src: IntArray, srcW: Int, srcH: Int, dstW: Int, dstH: Int): IntArray {
        require(srcW > 0 && srcH > 0 && dstW > 0 && dstH > 0) { "Invalid resize geometry" }
        require(src.size >= srcW * srcH) { "Source pixel array is smaller than the declared geometry" }

        // Horizontal pass. The intermediate is srcH * dstW pixels wide, so
        // it is kept as one flat FloatArray rather than an array of small
        // arrays - the per-pixel-object version allocates hundreds of
        // thousands of short-lived objects on a 1024x1024 source.
        val (xStart, xWeights) = axisWeights(srcW, dstW)
        val horiz = FloatArray(srcH * dstW * 3)
        for (y in 0 until srcH) {
            val rowBase = y * srcW
            val outBase = y * dstW * 3
            for (x in 0 until dstW) {
                val w = xWeights[x]
                val s = xStart[x]
                var r = 0f
                var g = 0f
                var b = 0f
                for (k in w.indices) {
                    val c = src[rowBase + s + k]
                    val wk = w[k].toFloat()
                    r += wk * ((c shr 16) and 0xFF)
                    g += wk * ((c shr 8) and 0xFF)
                    b += wk * (c and 0xFF)
                }
                val o = outBase + x * 3
                // Pillow's resample keeps an 8-BIT intermediate between the
                // horizontal and vertical passes, so the horizontal result is
                // rounded and SATURATED to 0..255 before the vertical pass.
                // Without this the two implementations diverge by up to 26
                // levels on upscaling, because the cubic kernel overshoots
                // and Pillow saturates there.
                horiz[o] = r.roundToInt().coerceIn(0, 255).toFloat()
                horiz[o + 1] = g.roundToInt().coerceIn(0, 255).toFloat()
                horiz[o + 2] = b.roundToInt().coerceIn(0, 255).toFloat()
            }
        }

        // Vertical pass.
        val (yStart, yWeights) = axisWeights(srcH, dstH)
        val out = IntArray(dstW * dstH)
        for (oy in 0 until dstH) {
            val w = yWeights[oy]
            val s = yStart[oy]
            for (ox in 0 until dstW) {
                var r = 0f
                var g = 0f
                var b = 0f
                for (k in w.indices) {
                    val base = (s + k) * dstW * 3 + ox * 3
                    val wk = w[k].toFloat()
                    r += wk * horiz[base]
                    g += wk * horiz[base + 1]
                    b += wk * horiz[base + 2]
                }
                val ri = r.roundToInt().coerceIn(0, 255)
                val gi = g.roundToInt().coerceIn(0, 255)
                val bi = b.roundToInt().coerceIn(0, 255)
                out[oy * dstW + ox] = (0xFF shl 24) or (ri shl 16) or (gi shl 8) or bi
            }
        }
        return out
    }
}
