package com.onndetect

import com.onndetect.detection.audio.WindowPooling
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * Pooling of per-window detector scores.
 *
 * Two properties matter and both were violated by the first draft:
 *
 *  1. `lcnn_audio_int8.tflite` ends in a sigmoid, so the model emits a
 *     PROBABILITY. Pooling probabilities with a log-sum-exp and returning
 *     the result would produce a value outside `[0, 1]`, which the
 *     `ScanResult` constructor rejects and which the calibrated threshold
 *     comparison cannot use.
 *  2. LSE pooling is bounded by the maximum. It interpolates between the
 *     mean (large temperature) and the max (small temperature), so it can
 *     never report a file as more synthetic than its most synthetic window.
 */
class WindowPoolingTest {

    private fun assertClose(label: String, expected: Double, actual: Float, tolerance: Double = 1e-4) {
        assertTrue(
            "$label: expected=$expected actual=$actual tolerance=$tolerance",
            abs(expected - actual) <= tolerance
        )
    }

    /**
     * Reference values for the standard LSE pooling, computed
     *
     *     logits = log(p / (1 - p))
     *     pooled = tau * log(mean(exp(logits / tau)))
     *     p_out  = sigmoid(pooled)
     *
     * against an independent implementation. They are hard-coded rather
     * than recomputed in Kotlin so the test cannot inherit a bug from a
     * second implementation of the same formula.
     */
    private val referenceCases = arrayOf(
        floatArrayOf(0.1f, 0.9f) to doubleArrayOf(0.8200000000, 0.6773362449, 0.5595835613),
        floatArrayOf(0.02f, 0.03f, 0.4f, 0.95f) to doubleArrayOf(0.8313517349, 0.5238152595, 0.3066522212),
        floatArrayOf(0.375f) to doubleArrayOf(0.3750000000, 0.3750000000, 0.3750000000),
    )
    private val referenceTaus = doubleArrayOf(1.0, 3.0, 10.0)

    @Test
    fun aSingleWindowPoolsToItself() {
        for (p in floatArrayOf(0.05f, 0.2f, 0.375f, 0.5f, 0.8f, 0.95f)) {
            assertClose("single $p", p.toDouble(), WindowPooling.poolProbabilities(floatArrayOf(p)), 1e-5)
        }
    }

    @Test
    fun identicalWindowsPoolToThatValue() {
        val pooled = WindowPooling.poolProbabilities(floatArrayOf(0.3f, 0.3f, 0.3f, 0.3f))
        assertClose("identical", 0.3, pooled, 1e-5)
    }

    @Test
    fun theResultIsAlwaysAProbability() {
        val cases = arrayOf(
            floatArrayOf(0.0f, 1.0f),
            floatArrayOf(0.01f, 0.99f),
            floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f),
            floatArrayOf(1f, 1f, 1f, 1f),
        )
        for (c in cases) {
            val pooled = WindowPooling.poolProbabilities(c)
            assertTrue("pooled=$pooled must be a probability in (0,1)", pooled > 0f && pooled < 1f)
        }
    }

    @Test
    fun theResultIsBetweenTheMeanAndTheMax() {
        val probs = floatArrayOf(0.1f, 0.9f, 0.2f, 0.8f)
        val pooled = WindowPooling.poolProbabilities(probs)
        val mean = probs.average()
        val max = probs.max()
        assertTrue("pooled=$pooled must be >= mean=$mean", pooled >= mean - 1e-6f)
        assertTrue("pooled=$pooled must be <= max=$max", pooled <= max + 1e-6f)
    }

    @Test
    fun aLargeTemperatureApproachesTheMean() {
        val probs = floatArrayOf(0.05f, 0.95f)
        val pooled = WindowPooling.poolProbabilities(probs, tau = 200f)
        assertClose("mean pooling", probs.average().toDouble(), pooled, 5e-2)
    }

    @Test
    fun aSmallTemperatureApproachesTheMax() {
        val probs = floatArrayOf(0.05f, 0.95f)
        val pooled = WindowPooling.poolProbabilities(probs, tau = 0.05f)
        assertClose("max pooling", probs.max().toDouble(), pooled, 5e-2)
    }

    @Test
    fun matchesTheReferenceLseValues() {
        for ((probs, expected) in referenceCases) {
            for (index in referenceTaus.indices) {
                assertClose(
                    "case ${probs.toList()} tau=${referenceTaus[index]}",
                    expected[index],
                    WindowPooling.poolProbabilities(probs, referenceTaus[index].toFloat()),
                    1e-5,
                )
            }
        }
    }

    @Test
    fun poolingIsMonotoneInEachWindow() {
        val base = floatArrayOf(0.2f, 0.3f, 0.4f)
        val raised = floatArrayOf(0.2f, 0.3f, 0.9f)
        assertTrue(
            "raising a window must not lower the pooled score",
            WindowPooling.poolProbabilities(raised) > WindowPooling.poolProbabilities(base)
        )
    }

    @Test
    fun anEmptyInputIsUnavailable() {
        assertTrue(WindowPooling.poolProbabilities(FloatArray(0)).isNaN())
    }

    @Test
    fun windowOffsetsAreSpreadAndInsideTheFile() {
        val offsets = WindowPooling.windowOffsets(durationMs = 60_000, windowMs = 4_000, count = 8)
        assertEquals(8, offsets.size)
        assertEquals(offsets.sorted(), offsets)
        assertEquals(offsets.distinct(), offsets)
        for (o in offsets) {
            assertTrue("offset $o must leave room for the window", o + 4_000 <= 60_000)
            assertTrue("offset $o must be inside the file", o >= 0)
        }
    }

    @Test
    fun aShortFileYieldsOneWindow() {
        val offsets = WindowPooling.windowOffsets(durationMs = 3_000, windowMs = 4_000, count = 8)
        assertEquals(1, offsets.size)
    }
}
