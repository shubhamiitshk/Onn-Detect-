package com.onndetect.detection.document

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import java.io.ByteArrayOutputStream
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Result of document and payment screenshot forensic analysis.
 */
data class DocumentForgeryReport(
    val isDocumentOrScreenshot: Boolean,
    val forgeryScore: Float, // 0.0 (Clean) to 1.0 (Manipulated)
    val isForged: Boolean,
    val elaPeakRatio: Float,
    val noiseDiscontinuity: Float,
    val edgeInconsistency: Float,
    val layoutAnomaly: Boolean,
    val tamperedTileIndices: List<Int>,
    val indicators: List<String>,
)

/**
 * On-device forensic analyzer for edited UPI payment screenshots (Google Pay, PhonePe, Paytm),
 * forged Indian KYC documents (Aadhaar, PAN cards), and digitally altered invoices.
 *
 * Employs three complementary forensic checks:
 * 1. Error Level Analysis (ELA): Identifies mismatched JPEG compression histories in spliced text/numbers.
 * 2. Spatial Noise-Floor Discontinuity: Compares content-tile variances to content baseline, and background
 *    tiles to background baseline, flagging isolated outliers (copy-paste bounding boxes).
 * 3. Manhattan-Grid & Antialiasing Profiler: Validates typography consistency and layout geometry against authentic mobile app viewports.
 */
class DocumentForgeryAnalyzer {

    companion object {
        const val GRID_SIZE = 16
        const val TOTAL_TILES = GRID_SIZE * GRID_SIZE
        const val FORGERY_THRESHOLD = 0.65f
    }

    /**
     * Analyzes an Android [Bitmap] for digital splicing, fake payment templates, and document forgery.
     */
    fun analyze(bitmap: Bitmap): DocumentForgeryReport {
        if (bitmap.isRecycled || bitmap.width < 32 || bitmap.height < 32) {
            return DocumentForgeryReport(
                isDocumentOrScreenshot = false,
                forgeryScore = 0f,
                isForged = false,
                elaPeakRatio = 1f,
                noiseDiscontinuity = 0f,
                edgeInconsistency = 0f,
                layoutAnomaly = false,
                tamperedTileIndices = emptyList(),
                indicators = listOf("Invalid or undersized image"),
            )
        }

        val width = bitmap.width
        val height = bitmap.height
        val scaledWidth = min(width, 512)
        val scaledHeight = min(height, 512)

        val scaled = if (width != scaledWidth || height != scaledHeight) {
            Bitmap.createScaledBitmap(bitmap, scaledWidth, scaledHeight, true)
        } else {
            bitmap
        }

        val pixels = IntArray(scaledWidth * scaledHeight)
        try {
            scaled.getPixels(pixels, 0, scaledWidth, 0, 0, scaledWidth, scaledHeight)
        } finally {
            if (scaled !== bitmap) scaled.recycle()
        }

        // BT.601 integer grayscale convention
        val gray = IntArray(pixels.size) { i ->
            val c = pixels[i]
            val r = c shr 16 and 255
            val g = c shr 8 and 255
            val b = c and 255
            (r * 299 + g * 587 + b * 114) / 1000
        }

        val elaPeakRatio = runCatching { computeEla(bitmap) }.getOrDefault(1f)
        return analyzeGrayscale(
            gray = gray,
            width = scaledWidth,
            height = scaledHeight,
            elaPeakRatio = elaPeakRatio,
            originalWidth = width,
            originalHeight = height,
        )
    }

    /**
     * Pure array-level analysis for JVM unit testing and core forensic mathematics.
     */
    fun analyzeGrayscale(
        gray: IntArray,
        width: Int,
        height: Int,
        elaPeakRatio: Float = 1.0f,
        originalWidth: Int = width,
        originalHeight: Int = height,
    ): DocumentForgeryReport {
        require(gray.size == width * height) { "Grayscale array size mismatch" }
        if (width < 16 || height < 16) {
            return DocumentForgeryReport(
                isDocumentOrScreenshot = false,
                forgeryScore = 0f,
                isForged = false,
                elaPeakRatio = 1f,
                noiseDiscontinuity = 0f,
                edgeInconsistency = 0f,
                layoutAnomaly = false,
                tamperedTileIndices = emptyList(),
                indicators = emptyList(),
            )
        }

        // 1. Modality Profile: Is this a structured UI screen, receipt, or KYC document?
        val (isDoc, manhattanRatio) = checkDocumentModality(gray, width, height)

        // 2. Spatial Noise-Floor Discontinuity across 16x16 tiles
        val (noiseDiscontinuity, kurtosis, tamperedTiles) = computeNoiseDiscontinuity(gray, width, height)

        // 3. Rectangular Splicing Boundary Edge Inconsistency
        val edgeInconsistency = computeEdgeBoundaryInconsistency(gray, width, height)

        // 4. Layout & Aspect Ratio Verification
        val aspectRatio = max(originalWidth, originalHeight).toFloat() / min(originalWidth, originalHeight)
        val isMobileScreenshotRatio = aspectRatio in 1.65f..2.45f
        val layoutAnomaly = isDoc && !isMobileScreenshotRatio && (aspectRatio < 1.3f || aspectRatio > 2.6f)

        // 5. Indicators compilation
        val indicators = mutableListOf<String>()
        if (elaPeakRatio > 3.0f) {
            indicators.add("Discontinuous JPEG compression error levels (ELA peak: %.2fx)".format(java.util.Locale.US, elaPeakRatio))
        }
        if (noiseDiscontinuity > 0.50f || kurtosis > 7.0f) {
            indicators.add("Localized noise variance anomaly across text/digits (Kurtosis: %.1f)".format(java.util.Locale.US, kurtosis))
        }
        if (edgeInconsistency > 0.55f) {
            indicators.add("Rectangular splicing boundary gradient detected")
        }
        if (layoutAnomaly) {
            indicators.add("Non-standard viewport aspect ratio for mobile payment receipt (%.2f:1)".format(java.util.Locale.US, aspectRatio))
        }

        // 6. Calibrated Forgery Score Fusion
        val elaTerm = ((elaPeakRatio - 1.0f) / 3.0f).coerceIn(0f, 1f)
        val noiseTerm = noiseDiscontinuity.coerceIn(0f, 1f)
        val edgeTerm = edgeInconsistency.coerceIn(0f, 1f)
        val layoutTerm = if (layoutAnomaly) 0.5f else 0.0f

        var rawScore = (0.35f * elaTerm) + (0.35f * noiseTerm) + (0.20f * edgeTerm) + (0.10f * layoutTerm)

        // If not a document or screenshot, suppress document-specific false alarms
        if (!isDoc && rawScore < 0.80f) {
            rawScore *= 0.5f
        }

        val finalScore = rawScore.coerceIn(0f, 1f)
        val isForged = finalScore >= FORGERY_THRESHOLD

        return DocumentForgeryReport(
            isDocumentOrScreenshot = isDoc,
            forgeryScore = finalScore,
            isForged = isForged,
            elaPeakRatio = elaPeakRatio,
            noiseDiscontinuity = noiseDiscontinuity,
            edgeInconsistency = edgeInconsistency,
            layoutAnomaly = layoutAnomaly,
            tamperedTileIndices = tamperedTiles,
            indicators = indicators,
        )
    }

    private fun checkDocumentModality(gray: IntArray, width: Int, height: Int): Pair<Boolean, Float> {
        var manhattanEdges = 0
        var totalEdges = 0
        var flatPixelCount = 0

        for (y in 1 until height - 1) {
            val row = y * width
            for (x in 1 until width - 1) {
                val idx = row + x
                val dx = abs(gray[idx + 1] - gray[idx - 1])
                val dy = abs(gray[idx + width] - gray[idx - width])
                val grad = dx + dy

                if (grad < 10) {
                    flatPixelCount++
                } else {
                    totalEdges++
                    val maxAxis = max(dx, dy)
                    val minAxis = min(dx, dy)
                    if (maxAxis > 0 && (minAxis.toFloat() / maxAxis) < 0.25f) {
                        manhattanEdges++
                    }
                }
            }
        }

        val totalPixels = (width - 2) * (height - 2)
        val flatRatio = flatPixelCount.toFloat() / totalPixels
        val manhattanRatio = if (totalEdges > 0) manhattanEdges.toFloat() / totalEdges else 0f

        val isDoc = (manhattanRatio > 0.55f && flatRatio > 0.25f) || flatRatio > 0.60f
        return Pair(isDoc, manhattanRatio)
    }

    /**
     * Partitions tiles into Content (text/graphics) and Background (whitespace/solid fill).
     * Compares content tiles to content baseline, and background tiles to background baseline.
     */
    private fun computeNoiseDiscontinuity(
        gray: IntArray,
        width: Int,
        height: Int,
    ): Triple<Float, Float, List<Int>> {
        val blockW = width / GRID_SIZE
        val blockH = height / GRID_SIZE
        if (blockW < 2 || blockH < 2) return Triple(0f, 0f, emptyList())

        val tileVariances = FloatArray(TOTAL_TILES)

        for (gy in 0 until GRID_SIZE) {
            val startY = gy * blockH
            val endY = min(startY + blockH, height - 1)
            for (gx in 0 until GRID_SIZE) {
                val startX = gx * blockW
                val endX = min(startX + blockW, width - 1)

                var sumL = 0.0
                var sumSqL = 0.0
                var count = 0

                for (y in startY + 1 until endY) {
                    val row = y * width
                    for (x in startX + 1 until endX) {
                        val idx = row + x
                        val laplacian = 4 * gray[idx] - gray[idx - width] - gray[idx + width] - gray[idx - 1] - gray[idx + 1]
                        sumL += laplacian
                        sumSqL += laplacian.toDouble() * laplacian
                        count++
                    }
                }

                val tileIdx = gy * GRID_SIZE + gx
                if (count > 4) {
                    val mean = sumL / count
                    val variance = (sumSqL / count - mean * mean).coerceAtLeast(0.0).toFloat()
                    tileVariances[tileIdx] = variance
                }
            }
        }

        // Separate content tiles (text/edges) from flat background tiles
        val contentTiles = mutableListOf<Float>()
        val backgroundTiles = mutableListOf<Float>()
        val contentThreshold = 250.0f // Text/edge tiles have high Laplacian variance

        for (v in tileVariances) {
            if (v >= contentThreshold) {
                contentTiles.add(v)
            } else if (v > 0.5f) {
                backgroundTiles.add(v)
            }
        }

        val outlierIndices = mutableListOf<Int>()
        var maxOutlierRatio = 1.0f

        // 1. Analyze Content Tile Consistency (Detects mismatched spliced fonts/numbers)
        if (contentTiles.size >= 4) {
            contentTiles.sort()
            val medianContent = contentTiles[contentTiles.size / 2]
            val q1 = contentTiles[contentTiles.size / 4]
            val q3 = contentTiles[(contentTiles.size * 3) / 4]
            val iqrContent = max(q3 - q1, medianContent * 0.25f)
            val threshold = medianContent + (2.5f * iqrContent)

            for (i in tileVariances.indices) {
                val v = tileVariances[i]
                if (v >= contentThreshold && v > threshold) {
                    outlierIndices.add(i)
                    val ratio = v / max(medianContent, 1.0f)
                    if (ratio > maxOutlierRatio) maxOutlierRatio = ratio
                }
            }
        }

        // 2. Analyze Background Tile Consistency (Detects in-painted or pasted patch backgrounds)
        if (backgroundTiles.size >= 8) {
            backgroundTiles.sort()
            val medianBg = backgroundTiles[backgroundTiles.size / 2]
            val q1 = backgroundTiles[backgroundTiles.size / 4]
            val q3 = backgroundTiles[(backgroundTiles.size * 3) / 4]
            val iqrBg = max(q3 - q1, medianBg * 0.25f)
            val thresholdBg = medianBg + (3.0f * iqrBg)

            for (i in tileVariances.indices) {
                val v = tileVariances[i]
                if (v < contentThreshold && v > thresholdBg && v > 5.0f) {
                    outlierIndices.add(i)
                    val ratio = v / max(medianBg, 0.5f)
                    if (ratio > maxOutlierRatio) maxOutlierRatio = ratio
                }
            }
        }

        // Compute Kurtosis of content tiles
        val kurtosis = if (contentTiles.size >= 6) {
            var mean = 0.0
            for (v in contentTiles) mean += v
            mean /= contentTiles.size

            var m2 = 0.0
            var m4 = 0.0
            for (v in contentTiles) {
                val diff = v - mean
                val diffSq = diff * diff
                m2 += diffSq
                m4 += diffSq * diffSq
            }
            val variance = m2 / contentTiles.size
            if (variance > 1e-4) {
                ((m4 / contentTiles.size) / (variance * variance) - 3.0).toFloat()
            } else 0f
        } else 0f

        val discontinuityScore = ((maxOutlierRatio - 1.0f) / 4.0f).coerceIn(0f, 1f)
        return Triple(discontinuityScore, kurtosis, outlierIndices)
    }

    private fun computeEdgeBoundaryInconsistency(gray: IntArray, width: Int, height: Int): Float {
        val horizEnergy = FloatArray(height)
        val vertEnergy = FloatArray(width)

        for (y in 1 until height - 1) {
            val row = y * width
            for (x in 1 until width - 1) {
                val idx = row + x
                val dy = abs(gray[idx + width] - gray[idx - width]).toFloat()
                val dx = abs(gray[idx + 1] - gray[idx - 1]).toFloat()
                horizEnergy[y] += dy
                vertEnergy[x] += dx
            }
        }

        fun peakiness(arr: FloatArray): Float {
            // Compare text lines against other text lines
            val activeLines = arr.filter { it > 15.0f }.sorted()
            if (activeLines.size < 6) return 1f
            val median = activeLines[activeLines.size / 2]
            val maxVal = activeLines.last()
            return if (median > 1.0f) maxVal / median else 1f
        }

        val hPeak = peakiness(horizEnergy)
        val vPeak = peakiness(vertEnergy)
        val maxPeak = max(hPeak, vPeak)

        return ((maxPeak - 2.5f) / 5.0f).coerceIn(0f, 1f)
    }

    private fun computeEla(bitmap: Bitmap): Float {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream)
        val bytes = stream.toByteArray()
        val recompressed = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return 1f

        val w = min(bitmap.width, 256)
        val h = min(bitmap.height, 256)

        val srcScaled = Bitmap.createScaledBitmap(bitmap, w, h, true)
        val recScaled = Bitmap.createScaledBitmap(recompressed, w, h, true)
        recompressed.recycle()

        val srcPixels = IntArray(w * h)
        val recPixels = IntArray(w * h)
        srcScaled.getPixels(srcPixels, 0, w, 0, 0, w, h)
        recScaled.getPixels(recPixels, 0, w, 0, 0, w, h)

        if (srcScaled !== bitmap) srcScaled.recycle()
        recScaled.recycle()

        val tileErrors = mutableListOf<Float>()
        val tw = w / 8
        val th = h / 8

        for (ty in 0 until 8) {
            for (tx in 0 until 8) {
                var totalDiff = 0.0
                var count = 0
                for (y in ty * th until (ty + 1) * th) {
                    val row = y * w
                    for (x in tx * tw until (tx + 1) * tw) {
                        val idx = row + x
                        val c1 = srcPixels[idx]
                        val c2 = recPixels[idx]
                        val dr = abs((c1 shr 16 and 255) - (c2 shr 16 and 255))
                        val dg = abs((c1 shr 8 and 255) - (c2 shr 8 and 255))
                        val db = abs((c1 and 255) - (c2 and 255))
                        totalDiff += (dr + dg + db)
                        count++
                    }
                }
                if (count > 0) {
                    tileErrors.add((totalDiff / count).toFloat())
                }
            }
        }

        if (tileErrors.isEmpty()) return 1f
        tileErrors.sort()
        val median = tileErrors[tileErrors.size / 2]
        val peak = tileErrors.last()

        return if (median > 0.2f) peak / median else 1f
    }
}
