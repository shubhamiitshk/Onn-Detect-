package com.onndetect.detection.temporal

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import kotlin.math.max
import kotlin.math.min

/**
 * Samples up to K evenly-spaced frames from a video using SEEK-based
 * decoding: for every target timestamp the extractor seeks to the nearest
 * preceding sync frame and only enough packets are decoded to produce that
 * one frame. Cost is O(K segments) instead of decoding the whole stream.
 */
class VideoDecoder(private val context: Context) {

    data class SampledVideo(
        val frames: List<Bitmap>,
        val width: Int,
        val height: Int,
        val durationMs: Long,
        val hasAudioTrack: Boolean,
    )

    fun sampleFrames(uri: Uri, maxFrames: Int = 10, outSize: Int = 512): SampledVideo {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)

            var videoTrack = -1
            var audioTrack = -1
            var format: MediaFormat? = null
            var durationUs = 0L
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                when (f.getString(MediaFormat.KEY_MIME)?.takeIf { it.isNotEmpty() }) {
                    null -> Unit
                    else -> if (f.getString(MediaFormat.KEY_MIME)!!.startsWith("video/") && videoTrack < 0) {
                        videoTrack = i
                        format = f
                        durationUs = if (f.containsKey(MediaFormat.KEY_DURATION))
                            f.getLong(MediaFormat.KEY_DURATION) else 0L
                    } else if (f.getString(MediaFormat.KEY_MIME)!!.startsWith("audio/") && audioTrack < 0) {
                        audioTrack = i
                    }
                }
            }
            require(videoTrack >= 0) { "No video track found" }

            val width = format!!.getInteger(MediaFormat.KEY_WIDTH)
            val height = format.getInteger(MediaFormat.KEY_HEIGHT)
            val mime = format.getString(MediaFormat.KEY_MIME)!!
            extractor.selectTrack(videoTrack)

            codec = MediaCodec.createDecoderByType(mime).apply {
                configure(format, null, null, 0)
                start()
            }

            // Evenly spaced targets across [2%, 98%] of the timeline.
            val lo = durationUs / 50
            val hi = durationUs - lo
            val targets = LongArray(maxFrames) { k ->
                if (durationUs > 0) lo + (hi - lo).toLong() * k / max(1, maxFrames - 1)
                else k * 500_000L // fallback: 0.5 s spacing when untimed
            }

            val picked = ArrayList<Bitmap>(maxFrames)
            val info = MediaCodec.BufferInfo()

            fun drainUntil(targetUs: Long): Boolean {
                // Returns true when a frame at/after the target was captured.
                var inputDone = false
                var guard = 0
                while (guard++ < 5_000) {
                    if (!inputDone) {
                        val inIdx = codec.dequeueInputBuffer(10_000)
                        if (inIdx >= 0) {
                            val buf = codec.getInputBuffer(inIdx)!!
                            val size = extractor.readSampleData(buf, 0)
                            if (size < 0) {
                                codec.queueInputBuffer(inIdx, 0, 0, 0,
                                    MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputDone = true
                            } else {
                                codec.queueInputBuffer(inIdx, 0, size,
                                    extractor.sampleTime, 0)
                                extractor.advance()
                            }
                        }
                    }
                    when (val outIdx = codec.dequeueOutputBuffer(info, 10_000)) {
                        MediaCodec.INFO_TRY_AGAIN_LATER,
                        MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> Unit
                        else -> if (outIdx >= 0) {
                            val hit = info.size > 0 &&
                                info.presentationTimeUs >= targetUs - 66_000
                            if (hit) {
                                codec.getOutputImage(outIdx)?.let { img ->
                                    picked.add(scale(yuvToArgb(img), outSize))
                                    img.close()
                                }
                            }
                            codec.releaseOutputBuffer(outIdx, false)
                            if (hit) return true
                            if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                                return picked.isNotEmpty() // stream exhausted
                            }
                        }
                    }
                }
                return false
            }

            for ((k, t) in targets.withIndex()) {
                if (picked.size > k) continue // already captured
                if (k > 0) {
                    extractor.seekTo(max(0L, t - 500_000L),
                        MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
                    codec.flush()
                }
                if (!drainUntil(t)) break
            }

            require(picked.isNotEmpty()) { "No frames could be decoded" }

            return SampledVideo(picked, width, height, durationUs / 1000, audioTrack >= 0)
        } finally {
            runCatching { codec?.stop(); codec?.release() }
            runCatching { extractor.release() }
        }
    }

    private fun scale(bmp: Bitmap, outSize: Int): Bitmap {
        val s = outSize.toFloat() / max(bmp.width, bmp.height)
        val scaled = Bitmap.createScaledBitmap(
            bmp, max(1, (bmp.width * s).toInt()), max(1, (bmp.height * s).toInt()), true)
        if (scaled !== bmp) bmp.recycle()
        return scaled
    }

    /** BT.601 limited-range YUV420 planar/semi-planar -> ARGB. */
    private fun yuvToArgb(image: Image): Bitmap {
        val w = image.width
        val h = image.height
        val yP = image.planes[0]
        val uP = image.planes[1]
        val vP = image.planes[2]
        val yB = yP.buffer.duplicate()
        val uB = uP.buffer.duplicate()
        val vB = vP.buffer.duplicate()
        val yRow = max(yP.rowStride, w)
        val uRow = max(uP.rowStride, w / 2)
        val vRow = max(vP.rowStride, w / 2)

        val argb = IntArray(w * h)
        var p = 0
        val yLim = yB.limit() - 1
        val uLim = uB.limit() - 1
        val vLim = vB.limit() - 1
        for (y in 0 until h) {
            val yLine = min(y * yRow, yLim - w + 1)
            val uvY = min((y / 2) * uRow, uLim)
            val vvY = min((y / 2) * vRow, vLim)
            for (x in 0 until w) {
                val yi = (yB.get(min(yLine + x, yLim)).toInt() and 0xFF) - 16
                val ci = min(x / 2, min(uRow, vRow) - 1)
                val uv = (uB.get(min(uvY + ci, uLim)).toInt() and 0xFF) - 128
                val vv = (vB.get(min(vvY + ci, vLim)).toInt() and 0xFF) - 128

                fun clamp(v: Int) = when {
                    v < 0 -> 0
                    v > 255 -> 255
                    else -> v
                }
                val r = clamp((1192 * yi + 1634 * vv) shr 10)
                val g = clamp((1192 * yi - 833 * vv - 400 * uv) shr 10)
                val b = clamp((1192 * yi + 2066 * uv) shr 10)
                argb[p++] = 0xFF shl 24 or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(argb, w, h, Bitmap.Config.ARGB_8888)
    }
}
