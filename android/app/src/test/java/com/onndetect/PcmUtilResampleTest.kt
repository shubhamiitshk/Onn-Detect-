package com.onndetect

import com.onndetect.detection.audio.PcmUtil
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Anti-aliasing behaviour of the Kotlin resampler.
 *
 * The bug these lock down: `resampleLinear` has no anti-alias filter, so
 * downsampling a 48 kHz phone recording to 16 kHz folds the 8-24 kHz band into
 * 0-8 kHz - exactly where vocoder artifacts live, and content that was never
 * present in the training distribution. `PcmUtil.resample` is a windowed-sinc
 * low-pass that suppresses it.
 */
class PcmUtilResampleTest {

    private fun rms(x: FloatArray): Double {
        if (x.isEmpty()) return 0.0
        var s = 0.0
        for (v in x) s += v.toDouble() * v
        return sqrt(s / x.size)
    }

    private fun tone(freq: Double, sr: Int, seconds: Double): FloatArray {
        val n = (sr * seconds).toInt()
        return FloatArray(n) { i -> sin(2 * PI * freq * i / sr).toFloat() }
    }

    @Test
    fun `identity when rates match`() {
        val x = floatArrayOf(0f, 1f, 2f, 3f, 4f)
        assertTrue(x.contentEquals(PcmUtil.resample(x, 16000, 16000)))
    }

    @Test
    fun `output length is the expected duration`() {
        val x = FloatArray(48000) { (it % 100).toFloat() / 100f }
        val y = PcmUtil.resample(x, 48000, 16000)
        assertEquals(16000, y.size)
    }

    @Test
    fun `out-of-band tones are suppressed`() {
        val sr = 48000
        for (f in doubleArrayOf(10000.0, 15000.0, 20000.0)) {
            val x = tone(f, sr, 1.0)
            val y = PcmUtil.resample(x, sr, 16000)
            val r = rms(y)
            assertTrue(
                "$f Hz should be rejected: rms = $r (want < 0.1 of input)",
                r < 0.1,
            )
        }
    }

    @Test
    fun `linear resampler demonstrably fails the same test`() {
        // Documents the regression this replaced. If resampleLinear ever starts
        // passing, the comparison test above stops being meaningful.
        val sr = 48000
        val x = tone(15000.0, sr, 1.0)
        val lin = rms(PcmUtil.resampleLinear(x, sr, 16000))
        assertTrue(
            "linear interpolation should pass 15 kHz at ~0.707 rms, got $lin",
            lin > 0.5,
        )
    }

    @Test
    fun `in-band tone keeps its amplitude`() {
        val sr = 48000
        val y = PcmUtil.resample(tone(1000.0, sr, 1.0), sr, 16000)
        val r = rms(y)
        // A unit sine has rms 1/sqrt(2) = 0.7071.
        assertTrue("1 kHz rms = $r (want ~0.707)", abs(r - 0.7071) < 0.05)
    }

    @Test
    fun `near-Nyquist in-band tone survives`() {
        val sr = 48000
        // 7 kHz is inside the 8 kHz output Nyquist and must be preserved;
        // an over-aggressive filter would kill it.
        val y = PcmUtil.resample(tone(7000.0, sr, 1.0), sr, 16000)
        assertTrue("7 kHz rms = ${rms(y)} (want ~0.707)", rms(y) > 0.6)
    }

    @Test
    fun `upsampling works and preserves amplitude`() {
        val y = PcmUtil.resample(tone(1000.0, 16000, 0.5), 16000, 48000)
        assertEquals((16000 * 0.5 * 3).toInt(), y.size)
        assertTrue("upsampled rms = ${rms(y)}", abs(rms(y) - 0.7071) < 0.05)
    }

    @Test
    fun `non-integer ratio is exact in length and rate`() {
        // 44100 -> 16000 needs factor 2.756, not 3: integer-decimate
        // implementations get both the length and the rate wrong here.
        val sr = 44100
        val x = tone(1000.0, sr, 1.0)
        val y = PcmUtil.resample(x, sr, 16000)
        assertEquals(16000, y.size)
        assertTrue("44100->16000 rms = ${rms(y)}", abs(rms(y) - 0.7071) < 0.05)
    }

    @Test
    fun `voice-note upsampling preserves tone`() {
        // 8 kHz voice notes upsampled to the 16 kHz analysis rate.
        val y = PcmUtil.resample(tone(1000.0, 8000, 1.0), 8000, 16000)
        assertEquals(16000, y.size)
        assertTrue("8000->16000 rms = ${rms(y)}", abs(rms(y) - 0.7071) < 0.05)
    }

    @Test
    fun `silence stays silent`() {
        val y = PcmUtil.resample(FloatArray(4800), 48000, 16000)
        assertEquals(1600, y.size)
        assertEquals(0.0, rms(y), 1e-6)
    }

    @Test
    fun `signal shorter than kernel does not crash`() {
        val x = FloatArray(64) { (it % 7).toFloat() }
        val y = PcmUtil.resample(x, 48000, 16000)
        assertTrue("got ${y.size} samples", y.all { it.isFinite() })
    }
}