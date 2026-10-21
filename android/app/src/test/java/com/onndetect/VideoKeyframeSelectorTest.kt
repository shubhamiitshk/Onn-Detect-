package com.onndetect

import com.onndetect.detection.video.VideoForensicAnalyzer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The 4-keyframe video probe selector.
 *
 * The frozen SigLIP-B/16 tower costs ~300-540 ms per frame, and the video
 * route used to run it on EVERY sampled frame. The selector bounds the
 * number of tower invocations while keeping slot 6 a per-frame quantity.
 *
 * The property that matters is not "the tower ran four times" - a frame
 * that a scene cut separates from every keyframe MUST still be scored
 * directly, or slot 6 would carry a measurement of a different frame. The
 * property is: a frame inherits a neighbour's score ONLY when every
 * transition between them is near-identical.
 */
class VideoKeyframeSelectorTest {

    @Test fun `no frames means no keyframes`() {
        assertTrue(VideoForensicAnalyzer.selectProbeKeyframes(0, FloatArray(0)).isEmpty())
    }

    @Test fun `few frames are all keyframes`() {
        for (count in 1..4) {
            val selected = VideoForensicAnalyzer.selectProbeKeyframes(count, FloatArray(count))
            assertEquals(count, selected.size)
        }
    }

    @Test fun `the first, middle and last frames are always keyframes`() {
        val selected = VideoForensicAnalyzer.selectProbeKeyframes(10, FloatArray(9) { 0.99f })
        assertTrue("the first frame must be a keyframe", 0 in selected)
        assertTrue("the middle frame must be a keyframe", 5 in selected)
        assertTrue("the last frame must be a keyframe", 9 in selected)
    }

    @Test fun `the frame after the sharpest transition is a keyframe`() {
        // The lowest adjacent similarity is at index 3, so frame 4 follows
        // the cut.
        val cosines = floatArrayOf(0.99f, 0.99f, 0.99f, 0.10f, 0.99f, 0.99f, 0.99f, 0.99f, 0.99f)
        val selected = VideoForensicAnalyzer.selectProbeKeyframes(10, cosines)
        assertTrue("the frame after the cut must be a keyframe", 4 in selected)
    }

    @Test fun `the keyframe count never exceeds the budget`() {
        val cosines = FloatArray(19) { 0.99f }
        val selected = VideoForensicAnalyzer.selectProbeKeyframes(20, cosines)
        assertTrue(
            "selected ${selected.size} keyframes, the budget is " +
                "${VideoForensicAnalyzer.MAX_PROBE_KEYFRAMES}",
            selected.size <= VideoForensicAnalyzer.MAX_PROBE_KEYFRAMES,
        )
    }

    @Test fun `a near-identical chain inherits the keyframe's score`() {
        // 10 frames, all transitions at 0.99 (near-identical).
        val cosines = FloatArray(9) { 0.99f }
        var towerCalls = 0
        val scores = VideoForensicAnalyzer.evaluateWithKeyframeReuse(10, cosines) { idx ->
            towerCalls++
            0.5f
        }
        assertEquals("every frame must have a score", 10, scores.count { it.isFinite() })
        assertEquals("all frames share the keyframe's score", 1, scores.toSet().size)
        assertEquals(
            "the tower must only run on the keyframes",
            VideoForensicAnalyzer.MAX_PROBE_KEYFRAMES,
            towerCalls,
        )
    }

    @Test fun `a frame separated by a scene cut is scored directly`() {
        // The cut is between frames 3 and 4 (similarity 0.10), so the
        // frames after it are not connected to the keyframes before it.
        val cosines = floatArrayOf(0.99f, 0.99f, 0.99f, 0.10f, 0.99f, 0.99f, 0.99f, 0.99f, 0.99f)
        val scored = mutableSetOf<Int>()
        val scores = VideoForensicAnalyzer.evaluateWithKeyframeReuse(10, cosines) { idx ->
            scored += idx
            idx.toFloat()
        }
        assertEquals("every frame must have a score", 10, scores.count { it.isFinite() })
        // The frame right after the cut cannot inherit from before it, so it
        // must have been measured.
        assertTrue("the frame after the cut must be measured directly", 4 in scored)
        // And it must carry ITS OWN score, not an inherited one.
        assertEquals(4f, scores[4], 1e-5f)
    }

    @Test fun `an unavailable keyframe is not inherited`() {
        val cosines = FloatArray(9) { 0.99f }
        val scores = VideoForensicAnalyzer.evaluateWithKeyframeReuse(10, cosines) { idx ->
            // The probe fails on every keyframe.
            if (VideoForensicAnalyzer.selectProbeKeyframes(10, cosines).contains(idx)) null else 0.25f
        }
        // No keyframe has a score, so nothing can be inherited from one; the
        // non-keyframes are measured directly.
        assertTrue("no score may be NaN-free-and-wrong", scores.any { it == 0.25f })
    }

    @Test fun `no frames means no tower calls`() {
        var towerCalls = 0
        val scores = VideoForensicAnalyzer.evaluateWithKeyframeReuse(0, FloatArray(0)) {
            towerCalls++
            0.5f
        }
        assertEquals(0, towerCalls)
        assertEquals(0, scores.size)
    }
}
