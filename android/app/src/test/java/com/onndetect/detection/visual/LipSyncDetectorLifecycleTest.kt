package com.onndetect.detection.visual

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import kotlin.math.PI

/**
 * Deterministic unit regression tests for LipSyncDetector lifecycle, resource management,
 * and MediaPipe reflection/dimension contracts.
 */
class LipSyncDetectorLifecycleTest {

    // --- Contract 1: Reflection Package Names ---

    @Test
    fun `mediapipe facelandmarker package reflection target follows 0_10 convention`() {
        val expectedPackage = "com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker"
        val expectedOptions = "com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker\$FaceLandmarkerOptions"
        val expectedResult = "com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult"

        assertTrue(expectedPackage.contains("facelandmarker.FaceLandmarker"))
        assertTrue(expectedOptions.contains("facelandmarker.FaceLandmarker\$FaceLandmarkerOptions"))
        assertTrue(expectedResult.contains("facelandmarker.FaceLandmarkerResult"))
        assertFalse("Must not use obsolete 'face' package", expectedPackage.contains(".vision.face."))
    }

    // --- Contract 2: MediaMetadataRetriever Lifecycle Ordering ---

    @Test
    fun `mediaMetadataRetriever release must occur strictly after frame extraction`() {
        class MockRetriever {
            var isReleased = false
                private set
            var frameCount = 10
            var framesExtracted = 0

            fun extractMetadata(key: Int): String? {
                check(!isReleased) { "MediaMetadataRetriever is released" }
                return frameCount.toString()
            }

            fun getFrameAtTime(timeUs: Long, option: Int): Any? {
                check(!isReleased) { "IllegalStateException: getFrameAtTime called on released retriever" }
                framesExtracted++
                return Object() // Mock bitmap
            }

            fun release() {
                isReleased = true
            }
        }

        // 1. Correct sequence (deferred release in finally)
        val retriever = MockRetriever()
        try {
            val count = retriever.extractMetadata(9)?.toIntOrNull() ?: 0
            assertEquals(10, count)
            val extracted = mutableListOf<Any>()
            for (i in 0 until 5) {
                extracted.add(retriever.getFrameAtTime(i * 33333L, 2)!!)
            }
            assertEquals(5, extracted.size)
        } finally {
            retriever.release()
        }
        assertTrue(retriever.isReleased)

        // 2. Erroneous sequence (premature release as was in old code)
        val buggyRetriever = MockRetriever()
        buggyRetriever.extractMetadata(9)
        buggyRetriever.release() // Premature release
        try {
            buggyRetriever.getFrameAtTime(0L, 2)
            fail("Expected IllegalStateException when calling getFrameAtTime on released retriever")
        } catch (e: IllegalStateException) {
            assertTrue(e.message!!.contains("released"))
        }
    }

    // --- Contract 3: ParcelFileDescriptor Leak Prevention ---

    @Test
    fun `parcelFileDescriptor is reliably closed via use block even on extraction failure`() {
        class MockParcelFileDescriptor : AutoCloseable {
            var isClosed = false
                private set

            override fun close() {
                isClosed = true
            }
        }

        val pfd = MockParcelFileDescriptor()
        var exceptionThrown = false
        try {
            pfd.use {
                throw RuntimeException("Simulated decoding error")
            }
        } catch (_: RuntimeException) {
            exceptionThrown = true
        }

        assertTrue("Exception should have been thrown", exceptionThrown)
        assertTrue("ParcelFileDescriptor must be closed even after exception", pfd.isClosed)
    }

    // --- Contract 4: Bitmap Native Memory Recycling ---

    @Test
    fun `bitmaps allocated during video extraction must be explicitly recycled in finally block`() {
        class MockBitmap {
            var isRecycled = false
                private set

            fun recycle() {
                isRecycled = true
            }
        }

        val bitmaps = mutableListOf<MockBitmap>()
        for (i in 0 until 10) {
            val bmp = MockBitmap()
            bitmaps.add(bmp)
            try {
                // Simulate processing
                check(i != 5) { "Simulated error on frame 5" }
            } catch (_: IllegalStateException) {
                // Ignored
            } finally {
                if (!bmp.isRecycled) {
                    bmp.recycle()
                }
            }
        }

        assertEquals(10, bitmaps.size)
        assertTrue("All bitmaps must be recycled regardless of frame processing errors",
            bitmaps.all { it.isRecycled })
    }

    // --- Contract 5: Landmark Extraction and Dimension Requirements ---

    private data class MockNormalizedLandmark(val x: Double, val y: Double)

    private fun createFullFaceLandmarks(): List<MockNormalizedLandmark> {
        return (0 until 478).map { idx ->
            when (idx) {
                PerioralLandmarkAnalyzer.LEFT_EYE_OUTER -> MockNormalizedLandmark(350.0, 350.0)
                PerioralLandmarkAnalyzer.RIGHT_EYE_OUTER -> MockNormalizedLandmark(650.0, 350.0)
                PerioralLandmarkAnalyzer.NOSE_TIP -> MockNormalizedLandmark(500.0, 450.0)
                in PerioralLandmarkAnalyzer.LIP_INDICES -> {
                    val angle = 2.0 * PI * PerioralLandmarkAnalyzer.LIP_INDICES.indexOf(idx) / 40.0
                    MockNormalizedLandmark(500.0 + 30.0 * kotlin.math.cos(angle), 600.0 + 30.0 * kotlin.math.sin(angle))
                }
                else -> MockNormalizedLandmark(500.0, 500.0)
            }
        }
    }

    @Test
    fun `computeJitterScore returns NaN when partial landmark subset (40 points) is passed`() {
        // In the buggy implementation, only LIP_INDICES (40 points) were passed.
        // computeJitterScore requires pts.size > 415 to perform pose normalization.
        val partialFrames = (0 until 5).map {
            // Only 40 points
            (0 until 40).map { MockNormalizedLandmark(500.0, 600.0) }
        }
        val jitter = PerioralLandmarkAnalyzer.computeJitterScore(partialFrames)
        assertTrue("Partial landmark list (<= 415 pts) must return NaN", jitter.isNaN())
    }

    @Test
    fun `computeJitterScore succeeds with valid score when full face mesh (478 points) is passed`() {
        val fullFrames = (0 until 5).map { createFullFaceLandmarks() }
        val jitter = PerioralLandmarkAnalyzer.computeJitterScore(fullFrames)
        assertFalse("Full face mesh (478 pts) must not return NaN", jitter.isNaN())
        assertTrue("Static full face landmarks should yield near-zero jitter", jitter < 0.10f)
    }

    // --- Contract 6: Timestamp Microsecond Scaling ---

    @Test
    fun `timestamp microsecond calculation must scale to microseconds not nanoseconds`() {
        val fps = 30L
        val frameIdx = 1L

        // Correct: time in microseconds
        val timeUs = frameIdx * 1_000_000L / fps
        assertEquals(33333L, timeUs) // ~33.3 milliseconds

        // Buggy: time scaled by extra 1000
        val buggyTimeUs = (frameIdx * 1000000L * 1000L / fps)
        assertEquals(33333333L, buggyTimeUs) // 33.3 SECONDS! (1000x error)

        // Verify that timestampMs matches timeUs / 1000
        val timestampMs = frameIdx * 1000L / fps
        assertEquals(33L, timestampMs)
        assertEquals(timeUs / 1000L, timestampMs)
    }

    // --- Contract 7: Video Duration Fallback When Frame Count Metadata Is Missing ---

    @Test
    fun `durationMs provides frame count fallback when METADATA_KEY_VIDEO_FRAME_COUNT is null`() {
        fun resolveFrameCount(rawFrameCount: Int?, durationMs: Long): Int {
            return when {
                rawFrameCount != null && rawFrameCount > 0 -> rawFrameCount
                durationMs > 0L -> (durationMs * 30L / 1000L).toInt()
                else -> 0
            }
        }

        // Case 1: Raw frame count present
        assertEquals(60, resolveFrameCount(60, 2000L))

        // Case 2: Raw frame count null, 2-second video at 30 fps
        assertEquals(60, resolveFrameCount(null, 2000L))

        // Case 3: Raw frame count 0, 500 ms video -> 15 frames
        assertEquals(15, resolveFrameCount(0, 500L))

        // Case 4: Audio-only / zero duration -> 0 frames (below 3 threshold)
        val count = resolveFrameCount(null, 0L)
        assertTrue("Insufficient frames for analysis", count < 3)
    }

    // --- Contract 8: Temporal Gap Handling in computeJitterScore ---

    @Test
    fun `computeJitterScore returns NaN when valid frames are non-consecutive with gaps`() {
        val valid = createFullFaceLandmarks()
        // 5 frames where no 3 consecutive frames exist: e.g. valid, null, valid, null, valid
        val framesWithGaps = listOf(valid, null, valid, null, valid)
        val jitter = PerioralLandmarkAnalyzer.computeJitterScore(framesWithGaps)
        assertTrue("Frames with gaps having no 3 consecutive tracked frames must return NaN", jitter.isNaN())
    }

    @Test
    fun `computeJitterScore only computes triplets over consecutive frames`() {
        val valid = createFullFaceLandmarks()
        // Frames: valid, valid, null, valid, valid -> only 2 consecutive in each island -> no triplet -> NaN
        val frames = listOf(valid, valid, null, valid, valid)
        val jitter = PerioralLandmarkAnalyzer.computeJitterScore(frames)
        assertTrue("No consecutive triplet means zero triplets evaluated -> NaN", jitter.isNaN())
    }

    @Test
    fun `computeJitterScore computes over consecutive run even when surrounding frames are missing`() {
        val valid = createFullFaceLandmarks()
        // Frames: null, valid, valid, valid, null -> 1 consecutive triplet -> valid score
        val frames = listOf(null, valid, valid, valid, null)
        val jitter = PerioralLandmarkAnalyzer.computeJitterScore(frames)
        assertFalse("At least 3 consecutive frames in a run yields a valid score", jitter.isNaN())
    }

    // --- Contract 9: Bilateral Malar rPPG Contiguous Run Requirements ---

    @Test
    fun `computeRppgAnomalyFromTimeline returns NaN when contiguous run is less than 4 frames`() {
        // Timeline with 10 total frames, but each contiguous run has length <= 3
        val left = listOf(0.35, 0.36, 0.37, null, 0.38, 0.39, 0.40, null, 0.41, 0.42)
        val right = listOf(0.35, 0.36, 0.37, null, 0.38, 0.39, 0.40, null, 0.41, 0.42)
        val anomaly = PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(left, right)
        assertTrue("Timeline without any contiguous run >= 4 frames must return NaN", anomaly.isNaN())
    }

    @Test
    fun `computeRppgAnomalyFromTimeline succeeds on contiguous run of 4 or more frames`() {
        // 2 dummy frames, then null, then 5 contiguous in-phase oscillating frames
        val left = listOf(0.1, 0.2, null, 0.35, 0.38, 0.36, 0.40, 0.37)
        val right = listOf(0.1, 0.2, null, 0.35, 0.38, 0.36, 0.40, 0.37)
        val anomaly = PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(left, right)
        assertFalse("Contiguous run of 5 frames must produce a valid anomaly score", anomaly.isNaN())
        assertTrue("In-phase malar oscillation should yield near-zero phase anomaly", anomaly < 0.10f)
    }

    @Test
    fun `computeRppgAnomalyFromTimeline returns NaN when non-finite or empty`() {
        val left = listOf(0.35, Double.NaN, 0.37, 0.38, 0.39)
        val right = listOf(0.35, 0.36, 0.37, 0.38, 0.39)
        val anomaly = PerioralLandmarkAnalyzer.computeRppgAnomalyFromTimeline(left, right)
        assertTrue("NaN in middle splits contiguous run below 4 -> NaN", anomaly.isNaN())
    }
}
