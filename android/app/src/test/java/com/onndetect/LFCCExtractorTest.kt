package com.onndetect

import com.onndetect.detection.audio.LFCCExtractor
import com.onndetect.detection.audio.PcmUtil
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Golden-vector test for the Kotlin LFCC front end.
 *
 * Goldens come from `scripts/gen_lfcc_golden.py --json scripts/lfcc_golden.json`,
 * generated from the NumPy reference (`ml/src/data/lfcc_reference.py`) and
 * cross-checked against the pinned PyTorch front end with `--torch`. CI runs
 * `--check` against that committed JSON, so these literals cannot drift from
 * their source unnoticed. Regenerate the JSON and these literals in the SAME
 * change whenever the front-end contract changes, and retrain: the 2e-3
 * tolerance is deliberately loose and will not flag a deliberate change.
 *
 * WHAT THIS TEST USED TO GET WRONG, AND WHY THAT MATTERED:
 * it asserted "each frame has zero mean across cepstral coefficients" - i.e.
 * per-frame z-scoring. That was the BUG, not the contract: training applies
 * CMVN per coefficient over TIME, so per-frame means are NOT zero. The test
 * therefore passed while the app disagreed with the model it was feeding.
 *
 * The invariants now asserted are:
 *   1. per-COEFFICIENT mean ~ 0 and std ~ 1 across TIME (the real contract)
 *   2. per-frame mean is NOT forced to 0 (regression guard for the old bug)
 *   3. c0 (log energy) retains real spread
 *   4. agreement with the training front end on golden frames
 */
class LFCCExtractorTest {

    private val sr = 16000
    private val n = 16000 // 1 s -> 101 frames at hop 160
    private val nLfcc = 40

    private fun signal(): FloatArray = FloatArray(n) { i ->
        val t = i.toDouble() / sr
        (0.5 * sin(2 * PI * 1000.0 * t)
            + 0.25 * sin(2 * PI * 2500.0 * t + 0.7)).toFloat()
    }

    private fun frame(out: Array<Array<Array<FloatArray>>>, t: Int) =
        DoubleArray(nLfcc) { out[0][t][it][0].toDouble() }

    @Test
    fun `shape matches the training front end`() {
        val out = LFCCExtractor().extract(signal())
        assertEquals(1, out.size)
        assertEquals(101, out[0].size)   // len/hop + 1 with center=true
        assertEquals(nLfcc, out[0][0].size)
        assertEquals(1, out[0][0][0].size)
    }

    @Test
    fun `matches training front-end golden frames`() {
        val out = LFCCExtractor().extract(signal())

        // FRAME50 = [-0.201757, -0.200907, 0.09098, -0.201051,
        //            -0.201809, -0.201672, -0.155396, 0.189619]
        val expected50 = doubleArrayOf(
            -0.201757, -0.200907, 0.09098, -0.201051,
            -0.201809, -0.201672, -0.155396, 0.189619,
        )
        val f50 = frame(out, 50)
        for (k in expected50.indices) {
            assertEquals(
                "coefficient $k of frame 50", expected50[k], f50[k], 2e-3,
            )
        }

        // FRAME0 = [5.148946, 4.709937, 1.004868, 5.588885]
        val expected0 = doubleArrayOf(5.148946, 4.709937, 1.004868, 5.588885)
        val f0 = frame(out, 0)
        for (k in expected0.indices) {
            assertEquals("coefficient $k of frame 0", expected0[k], f0[k], 2e-3)
        }

        // FRAME99 = [4.640401, 5.025739, -5.317201, 4.190491]
        // (literal frame index 99 - a previous golden mislabeled the LAST
        // frame as 99 and the JVM test caught the off-by-one)
        val expected99 = doubleArrayOf(4.640401, 5.025739, -5.317201, 4.190491)
        val f99 = frame(out, 99)
        for (k in expected99.indices) {
            assertEquals("coefficient $k of frame 99", expected99[k], f99[k], 2e-3)
        }

        // FRAME_LAST = [5.183328, 4.137931, -7.721352, 4.764157]
        // The generator emits this slice but the JVM test previously never
        // asserted it, so the exact index that once carried the off-by-one was
        // the one slice with no coverage.
        val expectedLast = doubleArrayOf(5.183328, 4.137931, -7.721352, 4.764157)
        val fLast = frame(out, out[0].size - 1)
        for (k in expectedLast.indices) {
            assertEquals("coefficient $k of last frame", expectedLast[k], fLast[k], 2e-3)
        }
    }

    @Test
    fun `CMVN is per-coefficient over time - the actual contract`() {
        val out = LFCCExtractor().extract(signal())
        val nFrames = out[0].size

        for (k in 0 until nLfcc) {
            var sum = 0.0
            for (t in 0 until nFrames) sum += frame(out, t)[k]
            val mean = sum / nFrames
            assertTrue(
                "coefficient $k mean over time = $mean (want ~0)",
                abs(mean) < 1e-3,
            )

            var sq = 0.0
            for (t in 0 until nFrames) {
                val d = frame(out, t)[k] - mean
                sq += d * d
            }
            val std = sqrt(sq / (nFrames - 1))
            assertEquals("coefficient $k std over time", 1.0, std, 0.02)
        }
    }

    @Test
    fun `per-frame mean is NOT zero - guards against the old per-frame bug`() {
        val out = LFCCExtractor().extract(signal())
        var worst = 0.0
        for (t in 0 until out[0].size) {
            var sum = 0.0
            for (k in 0 until nLfcc) sum += frame(out, t)[k]
            worst = maxOf(worst, abs(sum / nLfcc))
        }
        // If this ever approaches 0 the implementation has reverted to
        // per-frame z-scoring across coefficients, which does NOT match
        // training and collapses c0 (log energy).
        assertTrue(
            "per-frame mean is ~0 ($worst): per-frame normalization is back",
            worst > 1e-3,
        )
    }

    @Test
    fun `c0 log-energy retains spread`() {
        // Under per-frame z-scoring c0 is squeezed toward a constant; the
        // pinned CMVN leaves it with unit variance like every other coefficient.
        val out = LFCCExtractor().extract(signal())
        var sum = 0.0
        for (t in 0 until out[0].size) sum += frame(out, t)[0]
        val mean = sum / out[0].size
        var sq = 0.0
        for (t in 0 until out[0].size) {
            val d = frame(out, t)[0] - mean
            sq += d * d
        }
        val std = sqrt(sq / (out[0].size - 1))
        assertTrue("c0 std = $std (want >= 0.5)", std >= 0.5)
    }

    @Test
    fun `energy matches the CMVN theoretical value`() {
        // After per-coefficient CMVN, mean(x^2) == (T-1)/T for unit-variance
        // samples: 100/101 = 0.990099. The old per-frame version gave
        // 39/40 = 0.975, which is how the bug is distinguishable in aggregate.
        val out = LFCCExtractor().extract(signal())
        var sq = 0.0
        for (t in 0 until out[0].size) {
            for (k in 0 until nLfcc) {
                val v = frame(out, t)[k]
                sq += v * v
            }
        }
        assertEquals(100.0 / 101.0, sq / (out[0].size * nLfcc), 0.005)
    }

    @Test
    fun `four second window yields 401 frames`() {
        val out = LFCCExtractor().extract(FloatArray(64000))
        assertEquals(401, out[0].size)
    }

    @Test
    fun `digital silence stays finite`() {
        val out = LFCCExtractor().extract(FloatArray(16000))
        for (t in 0 until out[0].size) {
            for (k in 0 until nLfcc) {
                val v = frame(out, t)[k]
                assertTrue("silence frame $t coeff $k = $v", v.isFinite())
            }
        }
    }

    @Test
    fun `resampler identity and output length`() {
        val x = floatArrayOf(0f, 1f, 2f, 3f, 4f)
        assertTrue(x.contentEquals(PcmUtil.resampleLinear(x, 16000, 16000)))
        // 2 samples at 1 kHz -> 4001 Hz: floor(2*4001/1000) = 8
        assertEquals(8, PcmUtil.resampleLinear(floatArrayOf(0f, 4f), 1000, 4001).size)
    }

    @Test
    fun `center crop keeps middle window`() {
        val x = FloatArray(100) { it.toFloat() }
        val cropped = PcmUtil.centerCropOrPad(x, 10)
        assertEquals(listOf(45f, 46f, 47f, 48f, 49f, 50f, 51f, 52f, 53f, 54f), cropped.toList())
    }
}