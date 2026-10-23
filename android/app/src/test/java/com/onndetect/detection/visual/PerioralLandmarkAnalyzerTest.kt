package com.onndetect.detection.visual

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

class PerioralLandmarkAnalyzerTest {

    private fun generateFacePoints(
        centerX: Float = 200f,
        centerY: Float = 200f,
        eyeDist: Float = 100f,
        lipJitterOffset: Float = 0f,
        marAperture: Float = 20f,
    ): List<PerioralLandmarkAnalyzer.Point> {
        // Create 478 points
        val points = MutableList(478) { PerioralLandmarkAnalyzer.Point(centerX, centerY) }

        // Left eye outer: 33
        points[33] = PerioralLandmarkAnalyzer.Point(centerX - eyeDist / 2f, centerY - 40f)
        // Right eye outer: 263
        points[263] = PerioralLandmarkAnalyzer.Point(centerX + eyeDist / 2f, centerY - 40f)
        // Nose tip: 1
        points[1] = PerioralLandmarkAnalyzer.Point(centerX, centerY)

        // Lip contour markers
        // Upper inner lip: 13
        points[13] = PerioralLandmarkAnalyzer.Point(centerX, centerY + 30f - marAperture / 2f + lipJitterOffset)
        // Lower inner lip: 14
        points[14] = PerioralLandmarkAnalyzer.Point(centerX, centerY + 30f + marAperture / 2f - lipJitterOffset)
        // Left mouth corner: 78
        points[78] = PerioralLandmarkAnalyzer.Point(centerX - 35f, centerY + 30f)
        // Right mouth corner: 308
        points[308] = PerioralLandmarkAnalyzer.Point(centerX + 35f, centerY + 30f)

        // Fill remaining lip indices around mouth center
        for ((idx, lipIdx) in PerioralLandmarkAnalyzer.LIP_INDICES.withIndex()) {
            if (lipIdx != 13 && lipIdx != 14 && lipIdx != 78 && lipIdx != 308) {
                val angle = 2.0 * PI * idx / PerioralLandmarkAnalyzer.LIP_INDICES.size
                val rx = (cos(angle) * 30.0).toFloat()
                val ry = (sin(angle) * 15.0).toFloat() + lipJitterOffset
                points[lipIdx] = PerioralLandmarkAnalyzer.Point(centerX + rx, centerY + 30f + ry)
            }
        }

        return points
    }

    @Test
    fun `insufficient frames return NaN`() {
        val empty = emptyList<List<PerioralLandmarkAnalyzer.Point>?>()
        assertTrue(PerioralLandmarkAnalyzer.computeJitterScore(empty).isNaN())

        val oneFrame = listOf(generateFacePoints())
        assertTrue(PerioralLandmarkAnalyzer.computeJitterScore(oneFrame).isNaN())

        val twoFrames = listOf(generateFacePoints(), generateFacePoints())
        assertTrue(PerioralLandmarkAnalyzer.computeJitterScore(twoFrames).isNaN())
    }

    @Test
    fun `faceless or null frames return NaN`() {
        val nullFrames = listOf(null, null, null)
        assertTrue(PerioralLandmarkAnalyzer.computeJitterScore(nullFrames).isNaN())

        val shortPoints = listOf(
            List(100) { PerioralLandmarkAnalyzer.Point(0f, 0f) },
            List(100) { PerioralLandmarkAnalyzer.Point(0f, 0f) },
            List(100) { PerioralLandmarkAnalyzer.Point(0f, 0f) },
        )
        assertTrue(PerioralLandmarkAnalyzer.computeJitterScore(shortPoints).isNaN())
    }

    @Test
    fun `sub-threshold eye distance returns NaN`() {
        // Eye distance < 12 px
        val tinyFaceFrames = List(5) {
            generateFacePoints(eyeDist = 8f)
        }
        assertTrue(PerioralLandmarkAnalyzer.computeJitterScore(tinyFaceFrames).isNaN())
    }

    @Test
    fun `smooth biological motion yields low natural jitter score`() {
        val numFrames = 10
        val frames = List(numFrames) { t ->
            // Constant smooth translation & slow natural jaw movement
            generateFacePoints(
                centerX = 200f + t * 2f,
                centerY = 200f + t * 1f,
                eyeDist = 100f,
                lipJitterOffset = 0f,
                marAperture = 20f + (t * 0.5f),
            )
        }

        val score = PerioralLandmarkAnalyzer.computeJitterScore(frames)
        assertTrue("Smooth linear trajectory should produce low score (< 0.20, got $score)", score < 0.20f)
    }

    @Test
    fun `high frequency generative micro-jitter yields elevated score`() {
        val numFrames = 10
        val frames = List(numFrames) { t ->
            // Frame-to-frame high-frequency synthetic jitter (+3px / -3px alternating)
            val jitter = if (t % 2 == 0) 3.5f else -3.5f
            val marFlicker = if (t % 2 == 0) 12f else 28f
            generateFacePoints(
                centerX = 200f,
                centerY = 200f,
                eyeDist = 100f,
                lipJitterOffset = jitter,
                marAperture = marFlicker,
            )
        }

        val score = PerioralLandmarkAnalyzer.computeJitterScore(frames)
        assertTrue("Synthetic lip-sync micro-jitter should score high (> 0.50, got $score)", score > 0.50f)
    }
}
