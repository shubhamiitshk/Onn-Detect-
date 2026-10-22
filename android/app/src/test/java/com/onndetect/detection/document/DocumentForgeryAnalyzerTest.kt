package com.onndetect.detection.document

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

class DocumentForgeryAnalyzerTest {

    private val analyzer = DocumentForgeryAnalyzer()

    @Test
    fun `clean structured screenshot does not flag forgery`() {
        val width = 128
        val height = 256
        val gray = IntArray(width * height)

        // Generate a clean payment-like UI: white card with black horizontal text lines
        val rng = Random(42)
        for (y in 0 until height) {
            val isTextRow = (y in 40..44) || (y in 80..84) || (y in 120..124)
            for (x in 0 until width) {
                val idx = y * width + x
                val baseline = if (isTextRow && x in 20..100) 30 else 240
                // Uniform mild sensor noise
                val noise = rng.nextInt(-2, 3)
                gray[idx] = (baseline + noise).coerceIn(0, 255)
            }
        }

        val report = analyzer.analyzeGrayscale(
            gray = gray,
            width = width,
            height = height,
            elaPeakRatio = 1.2f,
            originalWidth = 1080,
            originalHeight = 2400,
        )

        assertTrue("Should detect structured document/screenshot modality", report.isDocumentOrScreenshot)
        assertFalse("Clean screenshot should not be flagged as forged", report.isForged)
        assertTrue(
            "Clean screenshot forgery score should be low (< 0.40), got ${report.forgeryScore} (noise=${report.noiseDiscontinuity}, edge=${report.edgeInconsistency}, layout=${report.layoutAnomaly})",
            report.forgeryScore < 0.40f
        )
        assertTrue("No tampered tiles in clean screenshot", report.tamperedTileIndices.isEmpty())
    }

    @Test
    fun `spliced transaction amount triggers noise discontinuity and detects tampered tile`() {
        val width = 128
        val height = 256
        val gray = IntArray(width * height)

        val rng = Random(1337)
        // Background: clean white document
        for (i in gray.indices) {
            gray[i] = (245 + rng.nextInt(-1, 2)).coerceIn(0, 255)
        }

        // Add some normal text lines
        for (y in 40..42) {
            for (x in 20..80) gray[y * width + x] = 30
        }

        // Splicing attack: Paste a high-variance noisy fake amount ("₹50,000") in tile (gy=5, gx=4)
        val blockW = width / DocumentForgeryAnalyzer.GRID_SIZE
        val blockH = height / DocumentForgeryAnalyzer.GRID_SIZE
        val targetTileY = 5
        val targetTileX = 4
        val targetTileIndex = targetTileY * DocumentForgeryAnalyzer.GRID_SIZE + targetTileX

        val startY = targetTileY * blockH
        val startX = targetTileX * blockW

        for (y in startY until startY + blockH) {
            for (x in startX until startX + blockW) {
                // High frequency sharp noise with jagged edits
                gray[y * width + x] = if ((x + y) % 2 == 0) 20 else 230
            }
        }

        val report = analyzer.analyzeGrayscale(
            gray = gray,
            width = width,
            height = height,
            elaPeakRatio = 4.2f, // ELA spike from recompression
            originalWidth = 1080,
            originalHeight = 2400,
        )

        assertTrue("Spliced document should be flagged as forged", report.isForged)
        assertTrue("Forgery score should exceed threshold (>= 0.65)", report.forgeryScore >= DocumentForgeryAnalyzer.FORGERY_THRESHOLD)
        assertTrue("Should detect the tampered tile index ($targetTileIndex)", report.tamperedTileIndices.contains(targetTileIndex))
        assertTrue("Indicators should report localized noise or ELA", report.indicators.isNotEmpty())
    }

    @Test
    fun `manhattan grid edge alignment distinguishes documents from organic photos`() {
        val size = 64

        // 1. Document / UI grid: strictly horizontal & vertical bars
        val docGray = IntArray(size * size) { 250 }
        for (y in 10..12) for (x in 5..55) docGray[y * size + x] = 20
        for (y in 20..50) for (x in 20..22) docGray[y * size + x] = 20

        val docReport = analyzer.analyzeGrayscale(docGray, size, size)
        assertTrue("UI grid should be classified as document", docReport.isDocumentOrScreenshot)

        // 2. Organic / Diagonal photo texture: 45-degree diagonal waves
        val photoGray = IntArray(size * size) { i ->
            val x = i % size
            val y = i / size
            ((kotlin.math.sin((x + y) * 0.3) * 100) + 128).toInt().coerceIn(0, 255)
        }
        val photoReport = analyzer.analyzeGrayscale(photoGray, size, size)
        assertFalse("Diagonal photo texture should not be classified as document layout", photoReport.isDocumentOrScreenshot)
    }

    @Test
    fun `abnormal aspect ratio in payment screenshot triggers layout anomaly`() {
        val width = 128
        val height = 128 // 1:1 Square - abnormal for full-screen UPI transaction
        val gray = IntArray(width * height) { 240 }

        val report = analyzer.analyzeGrayscale(
            gray = gray,
            width = width,
            height = height,
            elaPeakRatio = 1.0f,
            originalWidth = 500,
            originalHeight = 500, // 1:1 aspect ratio
        )

        // In 1:1 square document, layoutAnomaly is true
        assertTrue("Square payment screenshot should flag layout anomaly", report.layoutAnomaly)
    }
}
