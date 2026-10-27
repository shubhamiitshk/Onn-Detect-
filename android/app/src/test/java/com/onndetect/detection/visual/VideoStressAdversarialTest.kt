package com.onndetect.detection.visual

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Adversarial stress and temporal anomaly verification suite for M2 video forensics:
 * - computeJitterScore: gap preservation, no false acceleration spikes on head turns across gaps
 * - computeRppgBilateralPhaseAnomaly: contiguous run isolation, Float.NaN on scattered frames
 * - Boundary conditions: 0, 1, 2 frames, all-null frames, rapid alternating nulls, extreme face dimensions, out-of-order inputs
 */
class VideoStressAdversarialTest {

    private data class LandmarkPt(
        val x: Double,
        val y: Double,
        val z: Double = 0.0
    )

    private fun createFaceMesh(
        centerX: Double = 500.0,
        centerY: Double = 500.0,
        eyeDist: Double = 300.0,
        rollAngleRad: Double = 0.0,
        mouthAperture: Double = 1.0
    ): List<LandmarkPt> {
        val cosR = cos(rollAngleRad)
        val sinR = sin(rollAngleRad)
        val halfEye = eyeDist / 2.0

        return (0 until 478).map { idx ->
            when (idx) {
                // RIGHT_EYE_OUTER (33)
                PerioralLandmarkAnalyzer.RIGHT_EYE_OUTER -> {
                    val rx = -halfEye
                    val ry = -100.0
                    LandmarkPt(centerX + rx * cosR - ry * sinR, centerY + rx * sinR + ry * cosR)
                }
                // LEFT_EYE_OUTER (263)
                PerioralLandmarkAnalyzer.LEFT_EYE_OUTER -> {
                    val rx = halfEye
                    val ry = -100.0
                    LandmarkPt(centerX + rx * cosR - ry * sinR, centerY + rx * sinR + ry * cosR)
                }
                // NOSE_TIP (1)
                PerioralLandmarkAnalyzer.NOSE_TIP -> {
                    val rx = 0.0
                    val ry = -20.0
                    LandmarkPt(centerX + rx * cosR - ry * sinR, centerY + rx * sinR + ry * cosR)
                }
                // UPPER_INNER_LIP (13)
                PerioralLandmarkAnalyzer.UPPER_INNER_LIP -> {
                    val rx = 0.0
                    val ry = 70.0 - 5.0 * mouthAperture
                    LandmarkPt(centerX + rx * cosR - ry * sinR, centerY + rx * sinR + ry * cosR)
                }
                // LOWER_INNER_LIP (14)
                PerioralLandmarkAnalyzer.LOWER_INNER_LIP -> {
                    val rx = 0.0
                    val ry = 70.0 + 15.0 * mouthAperture
                    LandmarkPt(centerX + rx * cosR - ry * sinR, centerY + rx * sinR + ry * cosR)
                }
                // LEFT_MOUTH_CORNER (78)
                PerioralLandmarkAnalyzer.LEFT_MOUTH_CORNER -> {
                    val rx = -40.0
                    val ry = 70.0
                    LandmarkPt(centerX + rx * cosR - ry * sinR, centerY + rx * sinR + ry * cosR)
                }
                // RIGHT_MOUTH_CORNER (308)
                PerioralLandmarkAnalyzer.RIGHT_MOUTH_CORNER -> {
                    val rx = 40.0
                    val ry = 70.0
                    LandmarkPt(centerX + rx * cosR - ry * sinR, centerY + rx * sinR + ry * cosR)
                }
                in PerioralLandmarkAnalyzer.LIP_INDICES -> {
                    val lipIdx = PerioralLandmarkAnalyzer.LIP_INDICES.indexOf(idx)
                    val angle = 2.0 * PI * lipIdx / 40.0
                    val rx = 35.0 * cos(angle)
                    val ry = 70.0 + (15.0 * mouthAperture) * sin(angle)
                    LandmarkPt(centerX + rx * cosR - ry * sinR, centerY + rx * sinR + ry * cosR)
                }
                else -> {
                    LandmarkPt(centerX, centerY)
                }
            }
        }
    }

    // =========================================================================
    // Challenge 1: computeJitterScore Gap Preservation & False Spikes
    // =========================================================================

    @Test
    fun `computeJitterScore does not trigger false acceleration spikes on simulated head turns across gaps`() {
        // Run 1: 5 static frames before head turn at centerX = 500.0, mouth closed (aperture = 1.0)
        val beforeTurn = List(5) { createFaceMesh(centerX = 500.0, centerY = 500.0, mouthAperture = 1.0) }

        // Gap: 5 frames where face is turned away or tracking is lost (null)
        val gap = List(5) { null }

        // Run 2: 5 static frames after head turn returns with 150px shift (centerX = 650.0)
        // and mouth changed to wide open (aperture = 3.5) while turned away
        val afterTurn = List(5) { createFaceMesh(centerX = 650.0, centerY = 500.0, mouthAperture = 3.5) }

        val timeline = beforeTurn + gap + afterTurn
        assertEquals(15, timeline.size)

        val jitterWithGap = PerioralLandmarkAnalyzer.computeJitterScore(timeline)

        // Across the preserved gap, valid triplets are only evaluated within run 1 (t=1,2,3)
        // and within run 2 (t=11,12,13). No triplet spans across the gap.
        // Therefore, internal acceleration is strictly zero in both runs.
        assertFalse("Jitter should be finite", jitterWithGap.isNaN())
        assertTrue(
            "Jitter on static face with 150px shift across a 5-frame gap must be near 0.0 and strictly < 0.15 (was $jitterWithGap)",
            jitterWithGap < 0.05f
        )

        // Adversarial contrast: If the timeline was collapsed via filterNotNull() (the old defect),
        // the frame before the gap and after the gap would be adjacent, creating a huge false spike!
        val collapsedTimeline = timeline.filterNotNull()
        val jitterCollapsed = PerioralLandmarkAnalyzer.computeJitterScore(collapsedTimeline)
        assertTrue(
            "Collapsed timeline should falsely detect high jitter from the sudden mouth and pose jump (was $jitterCollapsed)",
            jitterCollapsed > 0.20f
        )
    }

    @Test
    fun `computeJitterScore remains near zero across multiple intermittent gaps and large coordinate shifts`() {
        // Multi-gap scenario:
        // Run 1: 4 frames at centerX = 400
        val run1 = List(4) { createFaceMesh(centerX = 400.0, centerY = 450.0) }
        // Gap 1: 3 nulls
        val gap1 = List(3) { null }
        // Run 2: 4 frames at centerX = 700 (+300px shift)
        val run2 = List(4) { createFaceMesh(centerX = 700.0, centerY = 480.0) }
        // Gap 2: 7 nulls
        val gap2 = List(7) { null }
        // Run 3: 5 frames at centerX = 250 (-450px shift)
        val run3 = List(5) { createFaceMesh(centerX = 250.0, centerY = 520.0) }

        val timeline = run1 + gap1 + run2 + gap2 + run3
        val jitter = PerioralLandmarkAnalyzer.computeJitterScore(timeline)

        assertFalse("Jitter across multiple preserved gaps must not be NaN", jitter.isNaN())
        assertTrue("Jitter should remain near 0.0 across all preserved gaps (was $jitter)", jitter < 0.05f)
    }

    // =========================================================================
    // Challenge 2: computeRppgBilateralPhaseAnomaly Run Length & Pulse Isolation
    // =========================================================================

    @Test
    fun `computeRppgBilateralPhaseAnomaly returns Float_NaN on scattered frames lacking 4 contiguous frames`() {
        // Case A: 30 total frames, but scattered in bursts of at most 3 frames
        val scatteredTimeline = mutableListOf<Double?>()
        while (scatteredTimeline.size < 30) {
            scatteredTimeline.add(0.35)
            scatteredTimeline.add(0.36)
            scatteredTimeline.add(0.37)
            scatteredTimeline.add(null) // interrupts before reaching 4
            scatteredTimeline.add(0.38)
            scatteredTimeline.add(0.39)
            scatteredTimeline.add(null) // interrupts
        }
        val left = scatteredTimeline.take(30)
        val right = scatteredTimeline.take(30)

        val anomaly = PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(left, right)
        assertTrue("Scattered timeline lacking any contiguous run >= 4 must return Float.NaN", anomaly.isNaN())

        // Case B: Alternating valid and null frames [V, N, V, N, ...]
        val alternating = List(20) { idx -> if (idx % 2 == 0) 0.36 else null }
        val anomalyAlt = PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(alternating, alternating)
        assertTrue("Rapid alternating null frames must return Float.NaN", anomalyAlt.isNaN())

        // Case C: Exactly 3 contiguous frames
        val threeFrames = listOf(0.35, 0.36, 0.37)
        val anomalyThree = PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(threeFrames, threeFrames)
        assertTrue("3 contiguous frames (< 4 threshold) must return Float.NaN", anomalyThree.isNaN())

        // Case D: 4 frames where middle frame has NaN
        val nanInMiddle = listOf(0.35, 0.36, Double.NaN, 0.38, 0.39)
        val anomalyNan = PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(nanInMiddle, nanInMiddle)
        assertTrue("NaN in middle splitting runs below 4 must return Float.NaN", anomalyNan.isNaN())

        // Case E: 4 frames where middle frame has Infinity
        val infInMiddle = listOf(0.35, 0.36, Double.POSITIVE_INFINITY, 0.38, 0.39)
        val anomalyInf = PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(infInMiddle, infInMiddle)
        assertTrue("Infinity in middle splitting runs below 4 must return Float.NaN", anomalyInf.isNaN())
    }

    @Test
    fun `computeRppgBilateralPhaseAnomaly correctly isolates in-phase cardiac pulse across continuous runs`() {
        val sampleRate = 30.0
        val pulseFreq = 1.2 // 72 bpm cardiac pulse

        // Generate 12 contiguous frames with clean in-phase bilateral pulse
        val cleanRunLeft = (0 until 12).map { t -> 0.35 + 0.02 * sin(2.0 * PI * pulseFreq * t / sampleRate) }
        val cleanRunRight = (0 until 12).map { t -> 0.35 + 0.02 * sin(2.0 * PI * pulseFreq * t / sampleRate) }

        // Surround by scattered nulls and short runs (< 4)
        val prefix = listOf(0.35, null, 0.36, 0.37, null)
        val suffix = listOf(null, 0.34, 0.35, null, 0.36)

        val leftTimeline = prefix + cleanRunLeft + suffix
        val rightTimeline = prefix + cleanRunRight + suffix

        val anomaly = PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(leftTimeline, rightTimeline)
        assertFalse("Anomaly on contiguous in-phase pulse must not be NaN", anomaly.isNaN())
        assertEquals("Perfect in-phase bilateral cardiac oscillation must yield 0.0 anomaly", 0.0f, anomaly, 0.01f)
    }

    @Test
    fun `computeRppgBilateralPhaseAnomaly correctly detects counterphase synthetic deepfake anomalies`() {
        val sampleRate = 30.0
        val pulseFreq = 1.2

        // Left and right are perfectly inverted (-1.0 correlation)
        val left = (0 until 10).map { t -> 0.35 + 0.02 * sin(2.0 * PI * pulseFreq * t / sampleRate) }
        val right = (0 until 10).map { t -> 0.35 - 0.02 * sin(2.0 * PI * pulseFreq * t / sampleRate) }

        val anomaly = PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(left, right)
        assertFalse("Anomaly must not be NaN", anomaly.isNaN())
        assertEquals("Inverted malar chromaticity must saturate to 1.0 anomaly", 1.0f, anomaly, 0.01f)
    }

    @Test
    fun `computeRppgBilateralPhaseAnomaly selects longest valid run when multiple runs exist`() {
        // Run 1: 4 frames with random noise / zero correlation
        val noiseLeft = listOf(0.35, 0.37, 0.34, 0.36)
        val noiseRight = listOf(0.36, 0.34, 0.37, 0.35)

        // Gap: 3 nulls
        val gap = listOf(null, null, null)

        // Run 2: 8 frames with perfect in-phase pulse
        val pulseLeft = (0 until 8).map { t -> 0.35 + 0.02 * sin(2.0 * PI * 1.2 * t / 30.0) }
        val pulseRight = (0 until 8).map { t -> 0.35 + 0.02 * sin(2.0 * PI * 1.2 * t / 30.0) }

        val leftTimeline = noiseLeft + gap + pulseLeft
        val rightTimeline = noiseRight + gap + pulseRight

        val anomaly = PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(leftTimeline, rightTimeline)
        assertFalse("Must be valid", anomaly.isNaN())
        // Should select the 8-frame run (pulse) over the 4-frame run (noise)
        assertTrue("Anomaly should isolate the longer 8-frame in-phase run (< 0.10f, was $anomaly)", anomaly < 0.10f)
    }

    @Test
    fun `computeRppgBilateralPhaseAnomaly truthfully returns NaN on zero-variance flat signals`() {
        // When malar green-ratio has zero variance (constant value), no rPPG pulse can be inferred.
        // Pearson correlation denominator is zero -> Double.NaN -> method must return Float.NaN.
        val flatSeries = DoubleArray(10) { 0.35 }
        val anomaly = PerioralLandmarkAnalyzer.computeRppgAnomalyFromSeries(flatSeries, flatSeries)
        assertTrue("Flat signal with zero variance must return Float.NaN", anomaly.isNaN())
    }

    // =========================================================================
    // Challenge 3: Edge Cases (0, 1, 2 frames, rapid alternating nulls, extreme coords, temporal ordering)
    // =========================================================================

    @Test
    fun `computeJitterScore boundary frames return Float_NaN`() {
        val valid = createFaceMesh()

        // 0 frames
        assertTrue(PerioralLandmarkAnalyzer.computeJitterScore(emptyList()).isNaN())
        // 1 frame
        assertTrue(PerioralLandmarkAnalyzer.computeJitterScore(listOf(valid)).isNaN())
        // 2 frames
        assertTrue(PerioralLandmarkAnalyzer.computeJitterScore(listOf(valid, valid)).isNaN())
        // 3 frames but middle is null
        assertTrue(PerioralLandmarkAnalyzer.computeJitterScore(listOf(valid, null, valid)).isNaN())
        // 20 all-null frames
        assertTrue(PerioralLandmarkAnalyzer.computeJitterScore(List(20) { null }).isNaN())
    }

    @Test
    fun `computeRppgAnomalyFromTimeline boundary frames return Float_NaN`() {
        // 0 frames
        assertTrue(PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(emptyList(), emptyList()).isNaN())
        // 1 frame
        assertTrue(PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(listOf(0.3), listOf(0.3)).isNaN())
        // 2 frames
        assertTrue(PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(listOf(0.3, 0.4), listOf(0.3, 0.4)).isNaN())
        // 3 frames
        assertTrue(PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(listOf(0.3, 0.4, 0.5), listOf(0.3, 0.4, 0.5)).isNaN())
        // 20 all-null frames
        assertTrue(PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(List(20) { null }, List(20) { null }).isNaN())
    }

    @Test
    fun `computeJitterScore filters out degenerate landmarks and extreme coordinates gracefully`() {
        val valid = createFaceMesh()

        // 1. Partial landmark list (size <= 415)
        val partialPts = (0 until 100).map { LandmarkPt(500.0, 500.0) }
        val partialFrames = listOf(partialPts, partialPts, partialPts, partialPts)
        assertTrue("Landmark sets with size <= 415 must be ignored and return NaN",
            PerioralLandmarkAnalyzer.computeJitterScore(partialFrames).isNaN())

        // 2. Sub-threshold eye distance (< 12px)
        val tinyFace = createFaceMesh(eyeDist = 8.0) // < 12.0px
        val tinyFrames = listOf(tinyFace, tinyFace, tinyFace, tinyFace)
        assertTrue("Eye distance < 12px must be ignored and return NaN",
            PerioralLandmarkAnalyzer.computeJitterScore(tinyFrames).isNaN())

        // 3. Non-finite landmark coordinates (NaN / Inf)
        val nanFace = createFaceMesh().mapIndexed { idx, pt ->
            if (idx == PerioralLandmarkAnalyzer.NOSE_TIP) LandmarkPt(Double.NaN, 500.0) else pt
        }
        val nanFrames = listOf(valid, nanFace, valid, valid)
        // With nanFace ignored as null, valid consecutive run is at most 2 -> returns NaN
        assertTrue("Non-finite coordinates must be discarded as null",
            PerioralLandmarkAnalyzer.computeJitterScore(nanFrames).isNaN())

        // 4. Extreme coordinate range (e.g. 100,000px coordinate offset)
        // Pose normalization should divide by eyeDist without overflow
        val extremeFrames = (0..4).map {
            createFaceMesh(centerX = 100_000.0, centerY = 100_000.0, eyeDist = 1_000.0)
        }
        val extremeJitter = PerioralLandmarkAnalyzer.computeJitterScore(extremeFrames)
        assertFalse("Extreme coordinates should normalize cleanly without NaN", extremeJitter.isNaN())
        assertTrue("Static face at extreme coordinate space should have low jitter", extremeJitter < 0.05f)
    }

    @Test
    fun `computeJitterScore is strictly sensitive to temporal order`() {
        // Natural speech: smooth continuous mouth movement
        val orderedFrames = (0 until 15).map { t ->
            val aperture = 1.0 + 0.3 * sin(2.0 * PI * t / 15.0)
            createFaceMesh(centerX = 500.0, centerY = 500.0, mouthAperture = aperture)
        }
        val orderedJitter = PerioralLandmarkAnalyzer.computeJitterScore(orderedFrames)
        assertTrue("Smooth ordered motion should have low jitter (was $orderedJitter)", orderedJitter < 0.15f)

        // Shuffled out-of-order frames (violating chronological timeline)
        // Interleaving frames: 0, 14, 1, 13, 2, 12...
        val outOfOrderFrames = mutableListOf<List<LandmarkPt>>()
        for (i in 0 until 7) {
            outOfOrderFrames.add(orderedFrames[i])
            outOfOrderFrames.add(orderedFrames[14 - i])
        }
        val outOfOrderJitter = PerioralLandmarkAnalyzer.computeJitterScore(outOfOrderFrames)
        assertTrue(
            "Out-of-order shuffled frames must produce high temporal acceleration jitter (was $outOfOrderJitter)",
            outOfOrderJitter > 0.30f
        )
    }
}
