package com.onndetect

import com.onndetect.detection.visual.BicubicResizer
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * Parity of the Android bicubic resampler with the Python training frontend.
 *
 * The Engine C frontend contract is
 * `clip-appearance-bicubic224-clipnorm-meanstd-float32`. The Python side
 * resizes with Pillow's `Image.Resampling.BICUBIC`
 * (`ml/src/data/image_frontend.py:53`). The Android side previously used
 * `Bitmap.createScaledBitmap(..., filter = true)`, which is BILINEAR, so
 * the device and the calibration pipeline disagreed about the input
 * tensor inside a hash-bound contract.
 *
 * The expected values below were produced by Pillow itself:
 *
 * ```
 * from PIL import Image
 * import numpy as np
 * h, w = 24, 32
 * arr = np.zeros((h, w, 3), dtype=np.uint8)
 * for y in range(h):
 *     for x in range(w):
 *         arr[y, x] = ((x*7 + y*3) % 256, (x*13 + y*5) % 256, (x*3 + y*11) % 256)
 * Image.fromarray(arr, "RGB").resize((8, 6), Image.Resampling.BICUBIC)
 * ```
 *
 * This is a JVM test because the JVM unit test source set has no
 * Robolectric, so `android.graphics.Bitmap` cannot be constructed there.
 * The pure pixel-array path is therefore the one under test.
 */
class BicubicResizerTest {

    private val srcW = 32
    private val srcH = 24

    private fun source(): IntArray {
        val pixels = IntArray(srcW * srcH)
        for (y in 0 until srcH) {
            for (x in 0 until srcW) {
                val r = (x * 7 + y * 3) % 256
                val g = (x * 13 + y * 5) % 256
                val b = (x * 3 + y * 11) % 256
                pixels[y * srcW + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return pixels
    }

    /** Pillow BICUBIC 32x24 -> 8x6, flattened row-major as RGB triples. */
    private val pillowReference8x6 = intArrayOf(
        16, 29, 23, 43, 78, 34, 72, 132, 47, 100, 189, 59, 128, 194, 71, 156, 37, 83, 184, 83, 95, 211, 134, 106,
        27, 48, 65, 54, 97, 76, 83, 151, 89, 111, 213, 101, 139, 128, 113, 167, 41, 125, 195, 104, 137, 222, 153, 148,
        40, 69, 110, 67, 118, 121, 96, 173, 134, 124, 219, 146, 152, 60, 157, 180, 65, 170, 207, 125, 183, 236, 174, 197,
        52, 89, 154, 79, 137, 164, 108, 199, 178, 136, 175, 193, 164, 36, 210, 191, 90, 224, 224, 145, 225, 207, 194, 207,
        64, 109, 202, 91, 159, 217, 120, 219, 225, 148, 98, 210, 176, 50, 178, 204, 112, 118, 235, 165, 66, 107, 215, 39,
        75, 128, 201, 102, 182, 153, 131, 204, 92, 159, 42, 42, 187, 75, 28, 219, 131, 33, 213, 186, 49, 30, 210, 65,
    )

    /** Pillow BICUBIC 32x24 -> 224x224, first row, first 8 pixels as RGB triples. */
    private val pillowFirstRow224 = intArrayOf(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 2, 0, 2, 3, 0, 3, 5, 0,
    )

    /** Pillow BICUBIC 32x24 -> 224x224, last row, first 8 pixels as RGB triples. */
    private val pillowLastRow224 = intArrayOf(
        68, 114, 255, 68, 115, 255, 69, 115, 255, 69, 116, 254,
        70, 117, 227, 70, 119, 190, 71, 120, 146, 72, 122, 95,
    )

    private fun assertChannel(label: String, expected: Int, actual: Int, tolerance: Int = 1) {
        assertTrue(
            "$label: expected=$expected actual=$actual tolerance=$tolerance",
            abs(expected - actual) <= tolerance
        )
    }

    private fun assertPixel(label: String, expected: IntArray, offset: Int, actual: Int, tolerance: Int = 1) {
        assertChannel("$label.r", expected[offset], (actual shr 16) and 0xFF, tolerance)
        assertChannel("$label.g", expected[offset + 1], (actual shr 8) and 0xFF, tolerance)
        assertChannel("$label.b", expected[offset + 2], actual and 0xFF, tolerance)
    }

    /**
     * Whole-image checksum against Pillow.
     *
     * The weighted sum is far stricter than a handful of sampled pixels:
     * a single wrong channel anywhere shifts it. Pillow values:
     *   32x24 -> 8x6      sum 18659,     checksum 1442299
     *   32x24 -> 224x224  sum 19368043,  checksum 1534380191550
     */
    private fun assertChecksum(label: String, out: IntArray, expectedSum: Long, expectedChecksum: Long) {
        var sum = 0L
        var checksum = 0L
        for (i in out.indices) {
            val r = (out[i] shr 16) and 0xFF
            val g = (out[i] shr 8) and 0xFF
            val b = out[i] and 0xFF
            sum += (r + g + b).toLong()
            checksum += (r.toLong() * (3 * i + 1)) + (g.toLong() * (3 * i + 2)) + (b.toLong() * (3 * i + 3))
        }
        assertTrue("$label sum: expected=$expectedSum actual=$sum", sum == expectedSum)
        assertTrue(
            "$label checksum: expected=$expectedChecksum actual=$checksum",
            checksum == expectedChecksum
        )
    }

    @Test
    fun matchesPillowExactlyOnADownscale() {
        val out = BicubicResizer.resizePixels(source(), srcW, srcH, 8, 6)
        for (y in 0 until 6) {
            for (x in 0 until 8) {
                assertPixel("downscale($x,$y)", pillowReference8x6, (y * 8 + x) * 3, out[y * 8 + x])
            }
        }
        assertChecksum("downscale", out, 18659L, 1442299L)
    }

    @Test
    fun matchesPillowExactlyOnTheFirstRowOfAnUpscale() {
        val out = BicubicResizer.resizePixels(source(), srcW, srcH, 224, 224)
        for (x in 0 until 8) {
            assertPixel("firstRow($x)", pillowFirstRow224, x * 3, out[x], 0)
        }
    }

    @Test
    fun matchesPillowExactlyOnTheLastRowOfAnUpscale() {
        val out = BicubicResizer.resizePixels(source(), srcW, srcH, 224, 224)
        val lastRow = 223 * 224
        for (x in 0 until 8) {
            assertPixel("lastRow($x)", pillowLastRow224, x * 3, out[lastRow + x], 0)
        }
    }

    @Test
    fun matchesPillowOnTheWholeUpscaledImage() {
        // The strongest parity statement available without shipping a
        // 224x224x3 fixture: the weighted checksum of every channel.
        val out = BicubicResizer.resizePixels(source(), srcW, srcH, 224, 224)
        assertChecksum("upscale", out, 19368043L, 1534380191550L)
    }

    @Test
    fun aFlatImageStaysFlat() {
        val flat = IntArray(srcW * srcH) { (0xFF shl 24) or (0x20 shl 16) or (0x40 shl 8) or 0x60 }
        val out = BicubicResizer.resizePixels(flat, srcW, srcH, 8, 6)
        for (pixel in out) {
            assertChannel("flat.r", 0x20, (pixel shr 16) and 0xFF, 0)
            assertChannel("flat.g", 0x40, (pixel shr 8) and 0xFF, 0)
            assertChannel("flat.b", 0x60, pixel and 0xFF, 0)
            assertChannel("flat.a", 0xFF, (pixel ushr 24) and 0xFF, 0)
        }
    }

    @Test
    fun theResultIsNotTheSameAsABoxFilter() {
        // Guards against a regression to nearest-neighbour: a box filter on
        // this image would be flat per source cell, bicubic is not.
        val out = BicubicResizer.resizePixels(source(), srcW, srcH, 8, 6)
        val distinct = out.toHashSet().size
        assertTrue("Expected a non-degenerate resample, got $distinct unique pixels", distinct > 12)
    }
}
