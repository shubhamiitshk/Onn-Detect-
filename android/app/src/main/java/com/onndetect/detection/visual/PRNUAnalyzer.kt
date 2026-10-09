package com.onndetect.detection.visual

import android.graphics.Bitmap
import android.graphics.Color
import javax.inject.Inject
import kotlin.math.abs
import kotlin.math.sqrt

/** Legacy residual-energy proxy. It does not measure camera PRNU or origin. */
class PRNUAnalyzer @Inject constructor() {
    fun analyze(bitmap: Bitmap): Float {
        // Downsample slightly to cap processing time on low-end devices
        val size = 512
        val scaled = Bitmap.createScaledBitmap(bitmap, size, size, true)
        
        // 1. Extract Noise Residual (Simplified High-Pass Filter)
        // In a full production C++ JNI module, this would use a 2D Daubechies 
        // Wavelet Transform (DWT) to isolate the exact sensor pattern.
        val noiseResidual = try { extractNoiseResidual(scaled, size) }
        finally { if (scaled !== bitmap) scaled.recycle() }
        
        // 2. Calculate Peak-to-Correlation Energy (PCE) or general variance
        // Descriptor thresholds are uncalibrated and excluded from verdicts.
        val variance = calculateVariance(noiseResidual)
        
        // Legacy descriptor mapping retained for diagnostics only.
        
        var anomalyScore = 0.0f
        if (variance < 0.8f) {
            // Low residual energy
            anomalyScore = 0.9f 
        } else if (variance > 8.0f) {
            // High residual energy
            anomalyScore = 0.7f
        } else {
            // Intermediate residual energy
            anomalyScore = 0.1f
        }
        
        return anomalyScore
    }
    
    private fun extractNoiseResidual(bitmap: Bitmap, size: Int): FloatArray {
        val pixels = IntArray(size * size)
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)

        // Only the interior is computed; return exactly (size-2)^2 values so
        // the variance does not count the two-pixel uncomputed border as
        // signal (the old layout embedded those as zeros, and then also
        // dropped interior zeros - a double corruption).
        val residual = FloatArray((size - 2) * (size - 2))
        val width = size
        var p = 0

        for (y in 1 until size - 1) {
            for (x in 1 until size - 1) {
                val idx = y * width + x
                val center = getLuma(pixels[idx])
                val left = getLuma(pixels[idx - 1])
                val right = getLuma(pixels[idx + 1])
                val up = getLuma(pixels[idx - width])
                val down = getLuma(pixels[idx + width])
                residual[p++] = 4 * center - left - right - up - down
            }
        }
        return residual
    }
    
    private fun getLuma(color: Int): Float {
        val r = Color.red(color)
        val g = Color.green(color)
        val b = Color.blue(color)
        return (0.299f * r + 0.587f * g + 0.114f * b)
    }
    
    private fun calculateVariance(data: FloatArray): Float {
        if (data.isEmpty()) return 0f

        var sum = 0.0
        for (v in data) sum += v.toDouble()
        val mean = sum / data.size

        var sumSquares = 0.0
        for (v in data) {
            val d = v - mean
            sumSquares += d * d
        }
        return (sumSquares / data.size).toFloat()
    }
}
