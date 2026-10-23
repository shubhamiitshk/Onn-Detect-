package com.onndetect.detection.visual

import com.onndetect.domain.pipeline.DetectionPipeline
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
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
                floatArrayOf(0.5f + rx, 0.65f + highFreqNoise * 0.5f)
            }
        }

        val jitter = analyzer.computeJitterFromTrajectories(syntheticJitterTrajectories)
        assertTrue("Synthetic high-frequency jitter should be elevated (was $jitter)", jitter > 0.50f)
    }

    // ── F5: MAR flicker, the fixed blend, pose normalization ─────────────

    @Test
    fun `smooth MAR series yields low flicker`() {
        // Natural articulation: 2 Hz sinusoidal aperture drift.
        val mar = FloatArray(30) { t -> 0.35f + (0.08 * sin(2.0 * PI * 2.0 * t / 30.0)).toFloat() }
        val flicker = analyzer.computeMarFlicker(mar)
        assertTrue("smooth aperture should flicker low (was $flicker)", flicker < 0.35f)
    }

    @Test
    fun `frame-to-frame MAR snapping yields high flicker`() {
        // Wav2Lip-style: the aperture teleports between two states each frame.
        val mar = FloatArray(30) { t -> if (t % 2 == 0) 0.15f else 0.75f }
        val flicker = analyzer.computeMarFlicker(mar)
        assertTrue("snapping aperture should flicker high (was $flicker)", flicker > 0.60f)
    }

    @Test
    fun `the blend is exactly 06 acceleration and 04 MAR`() {
        assertEquals(0.6f, BioLipLandmarkAnalyzer.ACCELERATION_WEIGHT, 1e-6f)
        assertEquals(0.4f, BioLipLandmarkAnalyzer.MAR_FLICKER_WEIGHT, 1e-6f)
        assertEquals(0.60f, analyzer.computeBioLipScore(0.50f, 0.75f), 1e-4f) // 0.3 + 0.3
        assertEquals(0.95f, analyzer.computeBioLipScore(0.99f, 0.89f), 1e-3f) // 0.594 + 0.356
        // Slot-9 range is respected even at the extremes.
        assertTrue(analyzer.computeBioLipScore(1f, 1f) <= 0.99f)
        assertTrue(analyzer.computeBioLipScore(0f, 0f) >= 0.01f)
    }

    @Test
    fun `pose normalization removes translation scale and roll`() {
        // A synthetic 478-point mesh: perioral points on a fixed ring around
        // the nose tip, eye corners placed symmetrically.
        fun meshAt(cx: Float, cy: Float, scale: Float, rollDeg: Double): Array<FloatArray> {
            val roll = Math.toRadians(rollDeg)
            val cos = kotlin.math.cos(roll).toFloat()
            val sin = kotlin.math.sin(roll).toFloat()
            fun place(mx: Float, my: Float): FloatArray {
                val dx = (mx - 0.5f) * scale
                val dy = (my - 0.45f) * scale
                return floatArrayOf(cx + dx * cos - dy * sin, cy + dx * sin + dy * cos)
            }
            val mesh = Array(478) { i ->
                // perioral points sit on a ring; others on a wider ring.
                val inner = i % 2 == 0
                val r = if (inner) 0.06f else 0.20f
                val a = 2.0 * PI * i / 478.0
                place(0.5f + (r * kotlin.math.cos(a)).toFloat(), 0.55f + (r * kotlin.math.sin(a)).toFloat())
            }
            // Canonical anchors: 33 right eye outer, 263 left eye outer, 1 nose tip.
            mesh[PerioralTopology.RIGHT_EYE_OUTER] = place(-0.12f, 0.30f)
            mesh[PerioralTopology.LEFT_EYE_OUTER] = place(0.12f, 0.30f)
            mesh[PerioralTopology.NOSE_TIP] = place(0f, 0.02f)
            return mesh
        }

        val base = PerioralTopology.normalizePose(meshAt(0.5f, 0.5f, 1.0f, 0.0))
        assertNotNull("canonical pose must normalize", base)
        assertEquals(64, base!!.size)

        val moved = PerioralTopology.normalizePose(meshAt(0.62f, 0.41f, 1.45f, 17.0))
        assertNotNull(moved)
        var maxDrift = 0f
        for (i in 0 until 64) {
            maxDrift = maxOf(maxDrift, kotlin.math.hypot(base[i][0] - moved!![i][0], base[i][1] - moved!![i][1]))
        }
        assertTrue("pose normalization must remove translation/scale/roll (drift $maxDrift)", maxDrift < 0.02f)
    }

    @Test
    fun `mouthAspectRatio reads the inner-lip ring`() {
        // Build a 64-point set via the topology order, then move the lips.
        val perioral = Array(64) { i ->
            PerioralTopology.PERIORAL_64[i].toFloat().let { idx ->
                floatArrayOf(idx / 100f, 0.5f)
            }
        }
        // upper inner lip (13) and lower (14) sit at y 0.5; corners 78/308 wide apart.
        val upperSlot = PerioralTopology.PERIORAL_64.indexOf(13)
        val lowerSlot = PerioralTopology.PERIORAL_64.indexOf(14)
        val rSlot = PerioralTopology.PERIORAL_64.indexOf(78)
        val lSlot = PerioralTopology.PERIORAL_64.indexOf(308)
        perioral[upperSlot] = floatArrayOf(0.0f, 0.45f)
        perioral[lowerSlot] = floatArrayOf(0.0f, 0.55f)
        perioral[rSlot] = floatArrayOf(-0.30f, 0.5f)
        perioral[lSlot] = floatArrayOf(0.30f, 0.5f)
        val mar = PerioralTopology.mouthAspectRatio(perioral)
        assertEquals("MAR = 0.1 vertical over 0.6 width", 0.1f / 0.6f, mar, 1e-4f)
    }

    @Test
    fun `a degenerate inter-ocular geometry refuses to normalize`() {
        val mesh = Array(478) { floatArrayOf(0.5f, 0.5f) } // all points collapsed
        assertNull(PerioralTopology.normalizePose(mesh))
    }
}
