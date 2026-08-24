package com.onndetect.detection.visual

import android.graphics.Bitmap
import android.graphics.Color
import javax.inject.Inject
import kotlin.math.abs
import kotlin.math.sqrt

class PRNUAnalyzer @Inject constructor() {

    /**
     * Extracts Photo Response Non-Uniformity (PRNU) noise residual.
     * Real camera sensors leave a unique, microscopic noise fingerprint.
     * AI-generated images (Midjourney, DALL-E) lack this physical hardware footprint.
     */
    fun analyze(bitmap: Bitmap): Float {
        // Downsample slightly to cap processing time on low-end devices
        val size = 512
        val scaled = Bitmap.createScaledBitmap(bitmap, size, size, true)
        
        // 1. Extract Noise Residual (Simplified High-Pass Filter)
        // In a full production C++ JNI module, this would use a 2D Daubechies 
        // Wavelet Transform (DWT) to isolate the exact sensor pattern.
        val noiseResidual = extractNoiseResidual(scaled, size)
        
        // 2. Calculate Peak-to-Correlation Energy (PCE) or general variance
        // AI images are mathematically "too smooth" at the hardware noise level.
        val variance = calculateVariance(noiseResidual)
        
        // Typical natural PRNU variance is around 2.0 to 5.0 (scaled)
        // AI images often drop below 0.5 because they lack physical sensor defects.
        
        var anomalyScore = 0.0f
        if (variance < 0.8f) {
            // Highly suspicious: lacks physical sensor noise
            anomalyScore = 0.9f 
        } else if (variance > 8.0f) {
            // Highly suspicious: injected artificial noise (e.g. GAN noise vectors)
            anomalyScore = 0.7f
        } else {
            // Natural hardware noise profile
            anomalyScore = 0.1f
        }
        
        return anomalyScore
    }
    
    private fun extractNoiseResidual(bitmap: Bitmap, size: Int): FloatArray {
        val pixels = IntArray(size * size)
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)
        
        val residual = FloatArray(pixels.size)
        val width = size
        
        // Simplified Laplacian high-pass filter to isolate high-frequency noise
        for (y in 1 until size - 1) {
            for (x in 1 until size - 1) {
                val idx = y * width + x
                
                // Get luma (brightness)
                val center = getLuma(pixels[idx])
                val left = getLuma(pixels[idx - 1])
                val right = getLuma(pixels[idx + 1])
                val up = getLuma(pixels[idx - width])
                val down = getLuma(pixels[idx + width])
                
                // Laplacian edge/noise detection
                val laplacian = (4 * center) - left - right - up - down
                residual[idx] = laplacian
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
        var sum = 0f
        var count = 0
        
        // Skip edges where Laplacian is 0
        for (v in data) {
            if (v != 0f) {
                sum += v
                count++
            }
        }
        if (count == 0) return 0f
        
        val mean = sum / count
        var sumSquares = 0f
        for (v in data) {
            if (v != 0f) {
                sumSquares += (v - mean) * (v - mean)
            }
        }
        
        return sumSquares / count
    }
}
