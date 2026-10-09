package com.onndetect

import com.onndetect.detection.audio.PcmUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class PcmEncodingTest {
    private fun bytes(vararg values: Int) = ByteBuffer.wrap(values.map { it.toByte() }.toByteArray()).order(ByteOrder.LITTLE_ENDIAN)
    @Test fun `unsigned8 and signed16 extremes decode correctly`() {
        assertArrayEquals(floatArrayOf(-1f, 0f, 127f / 128), PcmUtil.decodeInterleaved(bytes(0, 128, 255), 1, 3), 1e-7f)
        assertArrayEquals(floatArrayOf(-1f, 0f, 32767f / 32768), PcmUtil.decodeInterleaved(bytes(0, 128, 0, 0, 255, 127), 1, 2), 1e-7f)
    }
    @Test fun `signed24 sign extension is correct`() {
        assertArrayEquals(floatArrayOf(-1f, 0f, 8388607f / 8388608), PcmUtil.decodeInterleaved(bytes(0, 0, 128, 0, 0, 0, 255, 255, 127), 1, 21), 1e-7f)
    }
    @Test fun `signed32 and float32 stereo use encoding and channel count`() {
        val integer = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putInt(Int.MIN_VALUE).putInt(Int.MAX_VALUE).apply { flip() }
        assertEquals(0f, PcmUtil.decodeInterleaved(integer, 2, 22)[0], 1e-6f)
        val floats = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putFloat(.25f).putFloat(.75f).apply { flip() }
        assertEquals(.5f, PcmUtil.decodeInterleaved(floats, 2, 4)[0], 0f)
    }
    @Test(expected = IllegalArgumentException::class) fun `partial stereo frame rejected`() { PcmUtil.decodeInterleaved(bytes(0, 0), 2, 2) }
    @Test(expected = IllegalStateException::class) fun `unknown encoding rejected`() { PcmUtil.decodeInterleaved(bytes(0, 0), 1, 99) }
    @Test(expected = IllegalArgumentException::class) fun `nonfinite float PCM rejected`() {
        PcmUtil.decodeInterleaved(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(Float.NaN).apply { flip() }, 1, 4)
    }
    @Test(expected = IllegalArgumentException::class) fun `invalid rate rejected`() { PcmUtil.resample(floatArrayOf(1f), 0, 16000) }
}
