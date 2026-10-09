package com.onndetect

import com.onndetect.detection.temporal.Yuv420Converter
import java.nio.ByteBuffer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class Yuv420ConverterTest {
    private fun plane(bytes: IntArray, row: Int, pixel: Int, position: Int = 0) =
        Yuv420Converter.Plane(ByteBuffer.wrap(bytes.map { it.toByte() }.toByteArray()).apply { position(position) }, row, pixel)
    @Test fun `interleaved chroma padding matches planar values`() {
        val y = plane(IntArray(8) { 100 }, 4, 1)
        val planar = listOf(y, plane(intArrayOf(80, 180), 2, 1), plane(intArrayOf(160, 90), 2, 1))
        val interleaved = listOf(y, plane(intArrayOf(80, 99, 180, 99), 4, 2), plane(intArrayOf(160, 99, 90, 99), 4, 2))
        assertArrayEquals(Yuv420Converter.convert(planar, 0, 0, 4, 2), Yuv420Converter.convert(interleaved, 0, 0, 4, 2))
    }
    @Test fun `crop and plane buffer positions are obeyed`() {
        val y = plane(intArrayOf(55, 16, 16, 235, 235, 16, 16, 235, 235), 4, 1, 1)
        val u = plane(intArrayOf(55, 128, 128), 2, 1, 1)
        val v = plane(intArrayOf(55, 128, 128), 2, 1, 1)
        assertArrayEquals(intArrayOf(-1, -1, -1, -1), Yuv420Converter.convert(listOf(y, u, v), 2, 0, 2, 2))
    }
    @Test fun `limited black and full white decode correctly`() {
        val black = listOf(plane(intArrayOf(16), 1, 1), plane(intArrayOf(128), 1, 1), plane(intArrayOf(128), 1, 1))
        assertEquals(0xff000000.toInt(), Yuv420Converter.convert(black, 0, 0, 1, 1)[0])
        val white = listOf(plane(intArrayOf(255), 1, 1), black[1], black[2])
        assertEquals(-1, Yuv420Converter.convert(white, 0, 0, 1, 1, range = 1)[0])
    }
    @Test(expected = IllegalArgumentException::class) fun `truncated chroma rejected rather than clamped`() {
        val planes = listOf(plane(IntArray(8) { 100 }, 4, 1), plane(intArrayOf(128), 2, 1), plane(intArrayOf(128), 2, 1))
        Yuv420Converter.convert(planes, 0, 0, 4, 2)
    }
}
