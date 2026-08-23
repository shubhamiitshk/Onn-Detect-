package com.onndetect

import com.onndetect.detection.audio.LFCCExtractor
import com.onndetect.detection.audio.PcmUtil
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

/**
 * Golden-vector test: the same deterministic two-tone signal was run through
 * the REAL training front end (torchaudio.transforms.LFCC + per-frame
 * normalization, see ml/scripts/_gen_lfcc_golden.py). The Kotlin extractor
 * must reproduce those numbers.
 */
class LFCCExtractorTest {

    private val sr = 16000
    private val n = 16000 // 1 s -> 101 frames at hop 160

    private fun signal(): FloatArray = FloatArray(n) { i ->
        val t = i.toDouble() / sr
        (0.5 * sin(2 * PI * 1000.0 * t)
            + 0.25 * sin(2 * PI * 2500.0 * t + 0.7)).toFloat()
    }

    @Test
    fun `matches torchaudio golden frame coefficients`() {
        val out = LFCCExtractor().extract(signal())

        assertEquals(1, out.size)
        assertEquals(101, out[0].size)
        assertEquals(40, out[0][0].size)

        // FRAME50 = [-5.865654, 0.678512, 0.288967, -0.031839,
        //            -0.08776, 0.070386, 0.204051, 0.109862]
        // Tolerance rationale: torchaudio's internal float32 FFT leaves
        // different rounding noise than our float64 FFT in deep spectral
        // sidelobes; after DCT+normalization coefficients agree to <0.15 and
        // LCNN scores to <0.001 (see scripts/verify_audio_front_end.py).
        val expected50 = doubleArrayOf(
            -5.865654, 0.678512, 0.288967, -0.031839,
            -0.08776, 0.070386, 0.204051, 0.109862,
        )
        for (k in expected50.indices) {
            assertEquals(expected50[k], out[0][50][k][0].toDouble(), 0.2)
        }

        // FRAME0 = [-0.131986, 4.990418, 0.800025, -0.05293]
        val expected0 = doubleArrayOf(-0.131986, 4.990418, 0.800025, -0.05293)
        for (k in expected0.indices) {
            assertEquals(expected0[k], out[0][0][k][0].toDouble(), 0.2)
        }
    }

    @Test
    fun `normalized frames have zero mean and expected energy`() {
        val out = LFCCExtractor().extract(signal())
        var sqSum = 0.0
        for (t in 0 until 101) {
            var m = 0f
            for (k in 0 until 40) m += out[0][t][k][0]
            assertTrue(kotlin.math.abs(m / 40f) < 1e-3f)
            for (k in 0 until 40) sqSum += out[0][t][k][0].toDouble() * out[0][t][k][0].toDouble()
        }
        assertEquals(0.975, sqSum / (101 * 40), 0.02)
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
