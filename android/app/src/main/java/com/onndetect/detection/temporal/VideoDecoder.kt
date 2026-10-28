package com.onndetect.detection.temporal

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.media.Image
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.SystemClock
import java.nio.ByteBuffer
import java.util.concurrent.CancellationException
import kotlin.math.max

/** Bounded sampled-frame decoding. Report actual captured timestamps, not whole-file coverage. */
class VideoDecoder(private val context: Context) {
    data class SampledVideo(val frames: List<Bitmap>, val width: Int, val height: Int,
        val durationMs: Long, val hasAudioTrack: Boolean, val timestampsMs: List<Long> = emptyList())
    fun sampleFrames(uri: Uri, maxFrames: Int = 10, outSize: Int = 512, checkStop: () -> Unit = {}): SampledVideo {
        require(maxFrames in 1..30 && outSize in 32..1024)
        val deadline = SystemClock.elapsedRealtime() + 30_000
        fun checkProgress() {
            if (Thread.currentThread().isInterrupted) throw CancellationException("Video decode cancelled")
            checkStop()
            check(SystemClock.elapsedRealtime() <= deadline) { "Video decoding deadline exceeded" }
        }
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        val picked = ArrayList<Bitmap>()
        val timestamps = ArrayList<Long>()
        var transferred = false
        try {
            extractor.setDataSource(context, uri, null)
            val track = (0 until extractor.trackCount).firstOrNull { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true }
                ?: throw IllegalArgumentException("No video track found")
            val hasAudio = (0 until extractor.trackCount).any { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true }
            val format = extractor.getTrackFormat(track)
            fun rejectHdr(f: MediaFormat) {
                val transfer = if (f.containsKey(MediaFormat.KEY_COLOR_TRANSFER)) f.getInteger(MediaFormat.KEY_COLOR_TRANSFER) else 0
                require(transfer != 6 && transfer != 7) { "HDR video requires a validated tone-map path" }
            }
            rejectHdr(format)
            val rotation = if (format.containsKey(MediaFormat.KEY_ROTATION)) format.getInteger(MediaFormat.KEY_ROTATION) else 0
            require(rotation % 90 == 0) { "Unsupported video rotation" }
            val width = format.getInteger(MediaFormat.KEY_WIDTH)
            val height = format.getInteger(MediaFormat.KEY_HEIGHT)
            val duration = if (format.containsKey(MediaFormat.KEY_DURATION)) format.getLong(MediaFormat.KEY_DURATION) else 0L
            require(width > 0 && height > 0 && width.toLong() * height <= 16_777_216)
            extractor.selectTrack(track)
            codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME) ?: error("Missing video MIME"))
            val decoder = codec!!
            decoder.configure(format, null, null, 0); decoder.start()
            var outputFormat = format
            val targets = LongArray(maxFrames) { k ->
                if (duration > 0) { val lo = duration / 50; lo + (duration - 2 * lo) * k / max(1, maxFrames - 1) }
                else k * 500_000L
            }
            val info = MediaCodec.BufferInfo()
            for ((k, target) in targets.withIndex()) {
                checkProgress()
                // Seek for the first target too: a long clip's first target can
                // already be far beyond the decoder's per-frame deadline.
                extractor.seekTo(target, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
                if (k > 0) decoder.flush()
                var inputDone = false
                var outputDone = false
                var captured = false
                val stageDeadline = SystemClock.elapsedRealtime() + 5000
                while (!captured && !outputDone && SystemClock.elapsedRealtime() < stageDeadline) {
                    checkProgress()
                    if (!inputDone) {
                        val index = decoder.dequeueInputBuffer(10_000)
                        if (index >= 0) {
                            val buf = decoder.getInputBuffer(index) ?: error("Missing codec input")
                            val size = extractor.readSampleData(buf, 0)
                            if (size < 0) { decoder.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM); inputDone = true }
                            else { decoder.queueInputBuffer(index, 0, size, extractor.sampleTime, 0); extractor.advance() }
                        }
                    }
                    when (val index = decoder.dequeueOutputBuffer(info, 10_000)) {
                        MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> { outputFormat = decoder.outputFormat; rejectHdr(outputFormat) }
                        MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                        else -> if (index >= 0) {
                            try {
                                if (info.size > 0 && info.presentationTimeUs >= target) {
                                    val image = decoder.getOutputImage(index)
                                    if (image != null) {
                                        val bitmap = try { yuvToArgb(image, outputFormat) } finally { image.close() }
                                        val final = try { orientAndScale(bitmap, rotation, outSize) } catch (e: Throwable) { bitmap.recycle(); throw e }
                                        picked.add(final); timestamps.add(info.presentationTimeUs / 1000); captured = true
                                    }
                                }
                                outputDone = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                            } finally { decoder.releaseOutputBuffer(index, false) }
                        }
                    }
                }
                if (!captured) break
            }
            require(picked.isNotEmpty()) { "No supported frames could be decoded" }
            // Deduplicate frames that landed on the same presentation timestamp
            // (common when seekTo snaps to the same sync sample repeatedly).
            val seen = mutableSetOf<Long>()
            val dedupPicked = ArrayList<Bitmap>()
            val dedupTimestamps = ArrayList<Long>()
            for (i in picked.indices) {
                if (seen.add(timestamps[i])) {
                    dedupPicked.add(picked[i])
                    dedupTimestamps.add(timestamps[i])
                } else {
                    if (!picked[i].isRecycled) picked[i].recycle()
                }
            }
            transferred = true
            val rotated = rotation % 180 != 0
            return SampledVideo(dedupPicked, if (rotated) height else width, if (rotated) width else height, duration / 1000, hasAudio, dedupTimestamps)
        } finally {
            if (!transferred) picked.forEach { if (!it.isRecycled) it.recycle() }
            runCatching { codec?.stop() }; runCatching { codec?.release() }; extractor.release()
        }
    }
    private fun orientAndScale(bitmap: Bitmap, rotation: Int, outSize: Int): Bitmap {
        val oriented = if (rotation % 360 == 0) bitmap else Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height,
            Matrix().apply { postRotate(rotation.toFloat()) }, true).also { if (it !== bitmap) bitmap.recycle() }
        val factor = minOf(1f, outSize.toFloat() / max(oriented.width, oriented.height))
        val scaled = Bitmap.createScaledBitmap(oriented, max(1, (oriented.width * factor).toInt()), max(1, (oriented.height * factor).toInt()), true)
        if (scaled !== oriented) oriented.recycle()
        return scaled
    }
    private fun yuvToArgb(image: Image, format: MediaFormat): Bitmap {
        require(image.format == ImageFormat.YUV_420_888 && image.planes.size == 3) { "Unsupported video pixel format" }
        val crop = image.cropRect
        require(crop.left >= 0 && crop.top >= 0 && crop.right <= image.width && crop.bottom <= image.height && crop.width() > 0 && crop.height() > 0)
        require(crop.width().toLong() * crop.height() <= 16_777_216)
        val planes = image.planes.map { Yuv420Converter.Plane(it.buffer.duplicate(), it.rowStride, it.pixelStride) }
        val standard = if (format.containsKey(MediaFormat.KEY_COLOR_STANDARD)) format.getInteger(MediaFormat.KEY_COLOR_STANDARD) else 2
        val range = if (format.containsKey(MediaFormat.KEY_COLOR_RANGE)) format.getInteger(MediaFormat.KEY_COLOR_RANGE) else 2
        return Bitmap.createBitmap(Yuv420Converter.convert(planes, crop.left, crop.top, crop.width(), crop.height(), standard, range), crop.width(), crop.height(), Bitmap.Config.ARGB_8888)
    }
}

/** JVM-testable YUV plane addressing including buffer origin, crop and chroma pixel stride. */
internal object Yuv420Converter {
    data class Plane(val buffer: ByteBuffer, val rowStride: Int, val pixelStride: Int)
    fun convert(planes: List<Plane>, left: Int, top: Int, width: Int, height: Int, standard: Int = 2, range: Int = 2): IntArray {
        require(planes.size == 3 && width > 0 && height > 0 && left >= 0 && top >= 0 && width.toLong() * height <= 16_777_216)
        require(planes.all { it.rowStride > 0 && it.pixelStride > 0 })
        require(standard in setOf(1, 2, 4, 6) && range in 1..2) { "Unsupported color metadata" }
        val kr = when (standard) { 1 -> .2126; 6 -> .2627; else -> .299 }
        val kb = when (standard) { 1 -> .0722; 6 -> .0593; else -> .114 }
        val kg = 1.0 - kr - kb
        fun read(plane: Plane, x: Int, y: Int): Int {
            val index = plane.buffer.position().toLong() + y.toLong() * plane.rowStride + x.toLong() * plane.pixelStride
            require(index >= plane.buffer.position() && index < plane.buffer.limit()) { "Truncated video plane" }
            return plane.buffer.get(index.toInt()).toInt() and 255
        }
        fun clamp(value: Double) = value.toInt().coerceIn(0, 255)
        return IntArray(width * height) { i ->
            val x = left + i % width; val y = top + i / width
            val rawY = read(planes[0], x, y)
            val rawU = read(planes[1], x / 2, y / 2) - 128
            val rawV = read(planes[2], x / 2, y / 2) - 128
            val yy = if (range == 1) rawY.toDouble() else (rawY - 16) * (255.0 / 219)
            val uu = rawU * if (range == 1) 1.0 else 255.0 / 224
            val vv = rawV * if (range == 1) 1.0 else 255.0 / 224
            val red = clamp(yy + 2 * (1 - kr) * vv)
            val blue = clamp(yy + 2 * (1 - kb) * uu)
            val green = clamp(yy - 2 * kb * (1 - kb) / kg * uu - 2 * kr * (1 - kr) / kg * vv)
            (255 shl 24) or (red shl 16) or (green shl 8) or blue
        }
    }
}
