package com.onndetect.detection.visual

import android.graphics.Bitmap
import android.graphics.Color
import kotlin.math.*

/**
 * Authentic Research Implementation: "Watch your Up-Conv" (Durall et al., 2020)
 * 
 * Convolutional Neural Networks (GANs, Diffusion) rely on upsampling operations 
 * (Transposed Convolutions, Bilinear upsampling) which mathematically fail to 
 * reproduce the spectral distributions of natural images, particularly in the 
 * high-frequency domain.
 * 
 * Natural images follow a predictable 1/f^2 power-law decay in their power spectrum.
 * AI-generated images exhibit spectral anomalies (grid-like artifacts or sudden drop-offs).
 * This analyzer computes the 2D Fast Fourier Transform (FFT), extracts the 1D 
 * azimuthal average, and fits a log-log regression to measure deviation from nature.
 */
class FFTSpectralAnalyzer {
    
    fun analyze(bitmap: Bitmap): SpectralResult {
        // 1. Resize to power of 2 for Radix-2 FFT (e.g., 256x256)
        val size = 256
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, size, size, true)

        // 2. Extract grayscale pixels [0, 1]
        val pixels = IntArray(size * size)
        scaledBitmap.getPixels(pixels, 0, size, 0, 0, size, size)
        if (scaledBitmap !== bitmap) scaledBitmap.recycle()
        
        val grayData = DoubleArray(size * size)
        for (i in pixels.indices) {
            val color = pixels[i]
            val r = Color.red(color)
            val g = Color.green(color)
            val b = Color.blue(color)
            grayData[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        }
        
        // 3. Compute 2D Power Spectrum
        val powerSpectrum = compute2DPowerSpectrum(grayData, size)
        
        // 4. Azimuthal Integration (Radial averaging)
        val radialProfile = computeAzimuthalAverage(powerSpectrum, size)
        
        // 5. Power Law Regression (log-log space)
        val slope = computePowerLawSlope(radialProfile)
        
        // Natural images typically have a slope around -2.0.
        // Deviations indicate generative upsampling artifacts.
        val slopeDeviation = abs(slope - (-2.0)).toFloat()
        
        // Normalize score (0 = authentic, 1 = heavily anomalous)
        val score = (slopeDeviation / 2.0f).coerceIn(0.0f, 1.0f)
        
        return SpectralResult(
            slopeDeviation = slopeDeviation,
            spectralFlatness = computeWienerEntropy(powerSpectrum).toFloat(),
            score = score
        )
    }

    private fun compute2DPowerSpectrum(realInput: DoubleArray, n: Int): DoubleArray {
        // Real radix-2 FFT (row-column passes) via FFTUtil; output is already
        // fftshift-ed so DC sits at the center for the azimuthal average.
        return FFTUtil.powerSpectrum2D(realInput, n)
    }

    private fun computeAzimuthalAverage(powerSpectrum: DoubleArray, n: Int): DoubleArray {
        val center = n / 2
        val maxRadius = (n / 2.0 * sqrt(2.0)).toInt()
        val radialSum = DoubleArray(maxRadius + 1)
        val radialCount = IntArray(maxRadius + 1)

        for (y in 0 until n) {
            for (x in 0 until n) {
                val r = sqrt(((x - center) * (x - center) + (y - center) * (y - center)).toDouble()).toInt()
                if (r <= maxRadius) {
                    radialSum[r] += powerSpectrum[y * n + x]
                    radialCount[r]++
                }
            }
        }

        val radialAvg = DoubleArray(maxRadius + 1)
        for (i in radialAvg.indices) {
            if (radialCount[i] > 0) {
                radialAvg[i] = radialSum[i] / radialCount[i]
            }
        }
        return radialAvg
    }

    private fun computePowerLawSlope(radialProfile: DoubleArray): Double {
        var sumX = 0.0
        var sumY = 0.0
        var sumXY = 0.0
        var sumXX = 0.0
        var count = 0

        // Skip DC component (0) and extremely low frequencies
        for (i in 2 until radialProfile.size) {
            if (radialProfile[i] > 1e-10) { // avoid log(0)
                val logFreq = ln(i.toDouble())
                val logPower = ln(radialProfile[i])
                
                sumX += logFreq
                sumY += logPower
                sumXY += logFreq * logPower
                sumXX += logFreq * logFreq
                count++
            }
        }
        
        if (count < 2) return -2.0 // Fallback

        // Ordinary Least Squares (OLS) regression
        val meanX = sumX / count
        val meanY = sumY / count
        val numerator = sumXY - count * meanX * meanY
        val denominator = sumXX - count * meanX * meanX
        
        return if (denominator == 0.0) -2.0 else numerator / denominator
    }

    private fun computeWienerEntropy(spectrum: DoubleArray): Double {
        var sumLn = 0.0
        var sumLinear = 0.0
        var count = 0
        
        for (v in spectrum) {
            if (v > 1e-10) {
                sumLn += ln(v)
                sumLinear += v
                count++
            }
        }
        if (count == 0 || sumLinear == 0.0) return 0.0
        
        val geomMean = exp(sumLn / count)
        val arithMean = sumLinear / count
        return geomMean / arithMean
    }
}

data class SpectralResult(
    val slopeDeviation: Float,
    val spectralFlatness: Float, 
    val score: Float
)
