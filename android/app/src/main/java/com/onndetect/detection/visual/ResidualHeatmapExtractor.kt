package com.onndetect.detection.visual

import android.graphics.Bitmap
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Computes a spatial 32x32 residual-energy heatmap across an image canvas.
 *
 * Uses high-pass spatial Laplacian residuals on luma.
 *
 * The map shows the LOCAL RESIDUAL ENERGY of each cell - how much
 * high-frequency energy sits in that part of the image relative to the rest
 * of it. A pasted or in-painted region usually carries energy that does not
 * match its surroundings, so it shows up as a localised spike.
 *
 * It is an ANOMALY map, not a verdict. A uniformly noisy generative image
 * has no localised anomaly at all, so a flat map does NOT mean "natural" -
 * it means "no localised discontinuity". The scale is normalised per image,
 * so it says nothing about absolute noise.
 */
object ResidualHeatmapExtractor {

    const val GRID_SIZE = 32
    private const val TOTAL_CELLS = GRID_SIZE * GRID_SIZE

    fun computeResidualHeatmap(bitmap: Bitmap, gridSize: Int = GRID_SIZE): FloatArray {
        if (bitmap.isRecycled || bitmap.width < 8 || bitmap.height < 8) {
            return FloatArray(0)
        }

        // The ASPECT RATIO is preserved. The viewer maps the grid to
        // normalised coordinates (cellW = width / gridSize), so the cells
        // must be measured on a frame with the same shape as the source.
        // Resizing to a square first stretched every non-square photo, and
        // the highlighted region no longer pointed at the region that had
        // been altered.
        val longSide = 256
        val scale = longSide.toFloat() / max(bitmap.width, bitmap.height)
        // Even dimensions, so the 32x32 grid divides exactly.
        val scaledW = max(2, (bitmap.width * scale).toInt() and 0x7FFFFFFE)
        val scaledH = max(2, (bitmap.height * scale).toInt() and 0x7FFFFFFE)
        val scaled = if (scaledW == bitmap.width && scaledH == bitmap.height) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, scaledW, scaledH, true)
        }

        val pixels = IntArray(scaledW * scaledH)
        try {
            scaled.getPixels(pixels, 0, scaledW, 0, 0, scaledW, scaledH)
        } finally {
            if (scaled !== bitmap) scaled.recycle()
        }

        val blockW = scaledW / gridSize
        val blockH = scaledH / gridSize
        val rawVariances = FloatArray(gridSize * gridSize)

        for (gy in 0 until gridSize) {
            for (gx in 0 until gridSize) {
                var sum = 0.0
                var sumSq = 0.0
                var count = 0

                val startY = gy * blockH
                val endY = min(startY + blockH, scaledH - 1)
                val startX = gx * blockW
                val endX = min(startX + blockW, scaledW - 1)

                for (y in startY + 1 until endY) {
                    for (x in startX + 1 until endX) {
                        val idx = y * scaledW + x
                        val centerLuma = getLuma(pixels[idx])
                        val top = getLuma(pixels[idx - scaledW])
                        val bottom = getLuma(pixels[idx + scaledW])
                        val left = getLuma(pixels[idx - 1])
                        val right = getLuma(pixels[idx + 1])

                        // Discrete Laplacian high-pass filter: 4*center - top - bottom - left - right
                        val residual = (4.0 * centerLuma - top - bottom - left - right).toFloat()

                        sum += residual
                        sumSq += (residual * residual)
                        count++
                    }
                }

                val variance = if (count > 1) {
                    val mean = sum / count
                    max(0.0, (sumSq / count) - (mean * mean)).toFloat()
                } else {
                    0f
                }

                rawVariances[gy * gridSize + gx] = variance
            }
        }

        // Min-Max normalization into [0.0, 1.0]
        var minVal = Float.MAX_VALUE
        var maxVal = Float.MIN_VALUE
        for (v in rawVariances) {
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
        }

        val range = maxVal - minVal
        val normalized = FloatArray(rawVariances.size)
        if (range > 1e-6f) {
            for (i in rawVariances.indices) {
                normalized[i] = ((rawVariances[i] - minVal) / range).coerceIn(0f, 1f)
            }
        } else {
            // Flat canvas -> uniform 0.5
            normalized.fill(0.5f)
        }

        return normalized
    }

    private fun getLuma(rgb: Int): Float {
        val r = (rgb shr 16) and 0xFF
        val g = (rgb shr 8) and 0xFF
        val b = rgb and 0xFF
        // ITU-R BT.601 integer luma
        return (r * 299 + g * 587 + b * 114) / 1000f
    }
}
