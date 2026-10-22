package com.onndetect.detection.visual

import android.graphics.Bitmap
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Computes a spatial 32x32 residual-energy heatmap across an image canvas.
 *
 * Uses high-pass spatial Laplacian residuals on luma:
 * In natural camera captures, sensor noise is spatially uniform. In generative or
 * spliced images, in-painted text, faces, or pasted regions exhibit sharp localized
 * variance discontinuities compared to background noise.
 */
object ResidualHeatmapExtractor {

    const val GRID_SIZE = 32
    private const val TOTAL_CELLS = GRID_SIZE * GRID_SIZE

    fun computeResidualHeatmap(bitmap: Bitmap, gridSize: Int = GRID_SIZE): FloatArray {
        if (bitmap.isRecycled || bitmap.width < 8 || bitmap.height < 8) {
            return FloatArray(0)
        }

        val targetSide = 256
        val scaled = if (bitmap.width != targetSide || bitmap.height != targetSide) {
            Bitmap.createScaledBitmap(bitmap, targetSide, targetSide, true)
        } else {
            bitmap
        }

        val pixels = IntArray(targetSide * targetSide)
        try {
            scaled.getPixels(pixels, 0, targetSide, 0, 0, targetSide, targetSide)
        } finally {
            if (scaled !== bitmap) scaled.recycle()
        }

        val blockSize = targetSide / gridSize
        val rawVariances = FloatArray(gridSize * gridSize)

        for (gy in 0 until gridSize) {
            for (gx in 0 until gridSize) {
                var sum = 0.0
                var sumSq = 0.0
                var count = 0

                val startY = gy * blockSize
                val endY = min(startY + blockSize, targetSide - 1)
                val startX = gx * blockSize
                val endX = min(startX + blockSize, targetSide - 1)

                for (y in startY + 1 until endY) {
                    for (x in startX + 1 until endX) {
                        val idx = y * targetSide + x
                        val centerLuma = getLuma(pixels[idx])
                        val top = getLuma(pixels[idx - targetSide])
                        val bottom = getLuma(pixels[idx + targetSide])
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
