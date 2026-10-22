package com.onndetect.detection.visual

import com.onndetect.domain.pipeline.DetectionPipeline
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class BioLipLandmarkAnalyzerTest {

    private val analyzer = BioLipLandmarkAnalyzer()

    @Test
    fun `insufficient trajectories return neutral jitter`() {
        val emptyTrajectories = emptyArray<Array<FloatArray>>()
        val jitter0 = analyzer.computeJitterFromTrajectories(emptyTrajectories)
        assertEquals(DetectionPipeline.NEUTRAL_JITTER, jitter0, 1e-4f)

        val oneFrame = arrayOf(
            Array(BioLipLandmarkAnalyzer.NUM_PERIORAL_LANDMARKS) { floatArrayOf(0.5f, 0.5f) }
        )
        val jitter1 = analyzer.computeJitterFromTrajectories(oneFrame)
        assertEquals(DetectionPipeline.NEUTRAL_JITTER, jitter1, 1e-4f)

        val twoFrames = arrayOf(
            Array(BioLipLandmarkAnalyzer.NUM_PERIORAL_LANDMARKS) { floatArrayOf(0.5f, 0.5f) },
            Array(BioLipLandmarkAnalyzer.NUM_PERIORAL_LANDMARKS) { floatArrayOf(0.51f, 0.51f) }
        )
        val jitter2 = analyzer.computeJitterFromTrajectories(twoFrames)
        assertEquals(DetectionPipeline.NEUTRAL_JITTER, jitter2, 1e-4f)
    }

    @Test
    fun `static or linear motion yields low natural jitter`() {
        val numFrames = 10
        val numLandmarks = BioLipLandmarkAnalyzer.NUM_PERIORAL_LANDMARKS

        // Constant velocity motion (zero 2nd-order acceleration)
        val linearTrajectories = Array(numFrames) { t ->
            Array(numLandmarks) { p ->
                floatArrayOf(0.5f + t * 0.005f, 0.6f + t * 0.002f)
            }
        }

        val jitter = analyzer.computeJitterFromTrajectories(linearTrajectories)
        assertTrue("Linear motion should produce low jitter (was $jitter)", jitter < 0.20f)
    }

    @Test
    fun `smooth biological oscillation yields low natural jitter`() {
        val numFrames = 30
        val numLandmarks = BioLipLandmarkAnalyzer.NUM_PERIORAL_LANDMARKS

        // 2 Hz speech articulation cadence: smooth jaw/lip opening and closing
        val naturalSpeechTrajectories = Array(numFrames) { t ->
            val apertureDelta = (0.02 * sin(2.0 * PI * 2.0 * (t / 30.0))).toFloat()
            Array(numLandmarks) { p ->
                val baseAngle = 2.0 * PI * p / numLandmarks
                val rx = kotlin.math.cos(baseAngle).toFloat() * 0.05f
                val ry = kotlin.math.sin(baseAngle).toFloat() * (0.025f + apertureDelta)
                floatArrayOf(0.5f + rx, 0.65f + ry)
            }
        }

        val jitter = analyzer.computeJitterFromTrajectories(naturalSpeechTrajectories)
        assertTrue("Smooth biological speech should yield low jitter (was $jitter)", jitter < 0.35f)
    }

    @Test
    fun `high frequency generative micro-jitter yields elevated synthetic jitter`() {
        val numFrames = 30
        val numLandmarks = BioLipLandmarkAnalyzer.NUM_PERIORAL_LANDMARKS

        // Unconstrained frame-to-frame high-frequency jitter characteristic of generative lip-sync
        val syntheticJitterTrajectories = Array(numFrames) { t ->
            val highFreqNoise = if (t % 2 == 0) 0.04f else -0.04f
            Array(numLandmarks) { p ->
                val baseAngle = 2.0 * PI * p / numLandmarks
                val rx = kotlin.math.cos(baseAngle).toFloat() * 0.05f + highFreqNoise
                val ry = kotlin.math.sin(baseAngle).toFloat() * 0.025f - highFreqNoise
                floatArrayOf(0.5f + rx, 0.65f + ry)
            }
        }

        val jitter = analyzer.computeJitterFromTrajectories(syntheticJitterTrajectories)
        assertTrue("High-frequency generative jitter should yield high score > 0.60 (was $jitter)", jitter > 0.60f)
    }
}
