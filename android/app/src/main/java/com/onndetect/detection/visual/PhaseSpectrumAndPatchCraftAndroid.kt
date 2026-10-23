package com.onndetect.detection.visual

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import kotlin.math.abs

/**
 * Native-resolution PatchCraft (arXiv:2311.12397) tile mosaic extraction on Android.
 *
 * Extracts non-overlapping [tileSize]x[tileSize] patches from the native [Bitmap]
 * before any spatial downsampling, ranks them by integer BT.601 luma L1 gradient energy,
 * and assembles [mosaicSize]x[mosaicSize] mosaics (rich and poor) with zero interpolation.
 *
 * Bit-for-bit parity with `extract_patchcraft_mosaic` in `ml/scripts/train_srm_dcpt_patchcraft.py`.
 */
object PhaseSpectrumAndPatchCraftAndroid {

    data class MosaicPair(
        val richMosaic: Bitmap,
        val poorMosaic: Bitmap,
    )

    data class TileEnergy(
        val energy: Long,
        val y0: Int,
        val x0: Int,
    )

    fun extractPatchCraftMosaics(
        bitmap: Bitmap,
        mosaicSize: Int = 128,
        tileSize: Int = 32,
    ): MosaicPair {
        require(mosaicSize % tileSize == 0) { "mosaicSize ($mosaicSize) must be divisible by tileSize ($tileSize)" }
        val gridN = mosaicSize / tileSize
        val needed = gridN * gridN

        val w = bitmap.width
        val h = bitmap.height

        // If source is smaller than mosaicSize, scale with nearest neighbor (zero interpolation)
        val src = if (w < mosaicSize || h < mosaicSize) {
            Bitmap.createScaledBitmap(bitmap, maxOf(w, mosaicSize), maxOf(h, mosaicSize), false)
        } else {
            bitmap
        }

        val sw = src.width
        val sh = src.height
        val pixels = IntArray(sw * sh)
        src.getPixels(pixels, 0, sw, 0, 0, sw, sh)

        // Integer BT.601 luma: (299*R + 587*G + 114*B) / 1000
        val luma = IntArray(sw * sh)
        for (i in pixels.indices) {
            val c = pixels[i]
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            luma[i] = (299 * r + 587 * g + 114 * b) / 1000
        }

        val nRows = sh / tileSize
        val nCols = sw / tileSize
        val tiles = ArrayList<TileEnergy>(nRows * nCols)

        for (ry in 0 until nRows) {
            val y0 = ry * tileSize
            for (cx in 0 until nCols) {
                val x0 = cx * tileSize
                var gx = 0L
                var gy = 0L
                for (ty in 0 until tileSize) {
                    val rowOffset = (y0 + ty) * sw + x0
                    for (tx in 0 until tileSize - 1) {
                        gx += abs(luma[rowOffset + tx + 1] - luma[rowOffset + tx])
                    }
                }
                for (ty in 0 until tileSize - 1) {
                    val rowOffset = (y0 + ty) * sw + x0
                    val nextRowOffset = (y0 + ty + 1) * sw + x0
                    for (tx in 0 until tileSize) {
                        gy += abs(luma[nextRowOffset + tx] - luma[rowOffset + tx])
                    }
                }
                tiles.add(TileEnergy(gx + gy, y0, x0))
            }
        }

        val richOrdered = tiles.sortedWith(
            compareByDescending<TileEnergy> { it.energy }
                .thenBy { it.y0 }
                .thenBy { it.x0 }
        )
        val poorOrdered = tiles.sortedWith(
            compareBy<TileEnergy> { it.energy }
                .thenBy { it.y0 }
                .thenBy { it.x0 }
        )

        fun assemble(ordered: List<TileEnergy>): Bitmap {
            val out = Bitmap.createBitmap(mosaicSize, mosaicSize, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(out)
            for (idx in 0 until needed) {
                val tile = ordered[idx % ordered.size]
                val dstX = (idx % gridN) * tileSize
                val dstY = (idx / gridN) * tileSize
                val srcRect = Rect(tile.x0, tile.y0, tile.x0 + tileSize, tile.y0 + tileSize)
                val dstRect = Rect(dstX, dstY, dstX + tileSize, dstY + tileSize)
                canvas.drawBitmap(src, srcRect, dstRect, null)
            }
            return out
        }

        val richMosaic = assemble(richOrdered)
        val poorMosaic = assemble(poorOrdered)

        if (src !== bitmap) {
            src.recycle()
        }

        return MosaicPair(richMosaic, poorMosaic)
    }
}
