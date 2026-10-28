package com.onndetect.detection.temporal

import kotlin.math.max
import kotlin.math.min

/**
 * Temporal-consistency signal: consecutive sampled frames are reduced to
 * grayscale 16x16 descriptors and compared with cosine similarity.
 *
 * Natural video shows a characteristic band of inter-frame change; both
 * frozen-frame splices and over-smooth generative video deviate. This is a
 * heuristic signal (reported in the breakdown, not part of the XGBoost
 * feature contract).
 */
object FrameCosineDelta {

    const val GRID = 16

    fun descriptor(bmp: android.graphics.Bitmap): FloatArray {
        val w = bmp.width
        val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        return descriptorFromPixels(pixels, w, h)
    }

    /** Pure, unit-testable core over row-major ARGB pixels. */
    fun descriptorFromPixels(pixels: IntArray, w: Int, h: Int): FloatArray {
        require(w > 0 && h > 0 && w.toLong() * h <= pixels.size)
        val out = FloatArray(GRID * GRID)
        for (gy in 0 until GRID) {
            for (gx in 0 until GRID) {
                var sum = 0f
                var count = 0
                val x0 = gx * w / GRID
                val y0 = gy * h / GRID
                val x1 = (gx + 1) * w / GRID
                val y1 = (gy + 1) * h / GRID
                for (y in y0 until y1) {
                    for (x in x0 until x1) {
                        val c = pixels[y * w + x]
                        sum += 0.299f * (c shr 16 and 0xFF) +
                            0.587f * (c shr 8 and 0xFF) + 0.114f * (c and 0xFF)
                        count++
                    }
                }
                out[gy * GRID + gx] = if (count > 0) sum / count else 0f
            }
        }
        return out
    }

    fun cosine(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size && a.isNotEmpty() && a.all { it.isFinite() } && b.all { it.isFinite() })
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            na += a[i].toDouble() * a[i]
            nb += b[i].toDouble() * b[i]
        }
        if (na == 0.0 && nb == 0.0) return 1f
        if (na == 0.0 || nb == 0.0) return 0f
        return (dot / (kotlin.math.sqrt(na) * kotlin.math.sqrt(nb)))
            .toFloat().coerceIn(-1f, 1f)
    }

    /**
     * @param sims cosine similarities between consecutive frame descriptors.
     * @return temporalScore in [0,1]; higher = more anomalous dynamics.
     *         Both extremes of similarity (frozen or wildly flickering) map up.
     */
    fun anomalyScore(sims: List<Float>): Float {
        require(sims.all { it.isFinite() && it in -1f..1f })
        if (sims.isEmpty()) return 0f
        // Filter out abrupt scene cuts (sim < 0.50f) so ordinary camera shot transitions
        // do not masquerade as deepfake temporal flicker.
        val intraShotSims = sims.filter { it >= 0.50f }
        val evalSims = if (intraShotSims.size >= 2) intraShotSims else sims
        val mean = evalSims.sum() / evalSims.size
        var vr = 0.0
        for (s in evalSims) vr += (s - mean).toDouble() * (s - mean)
        val std = kotlin.math.sqrt(vr / evalSims.size).toFloat()

        // Frozen content: similarity pinned near 1.0
        val frozenness = max(0f, (mean - 0.995f) / 0.005f)
        // Instability: low mean similarity or high dispersion
        val instability = max(0f, (0.90f - mean) / 0.30f) +
            max(0f, (std - 0.08f) / 0.20f)

        return min(1f, max(frozenness, min(1f, instability)))
    }

    /** Convenience: descriptors -> per-gap similarities. */
    fun similarities(frames: List<android.graphics.Bitmap>): List<Float> {
        if (frames.size < 2) return emptyList()
        val desc = frames.map { descriptor(it) }
        return (1 until desc.size).map { i -> cosine(desc[i - 1], desc[i]) }
    }
}
