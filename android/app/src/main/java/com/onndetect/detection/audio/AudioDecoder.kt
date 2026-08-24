package com.onndetect.detection.audio

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * Streams a bounded window of audio out of any Android-supported media file.
 *
 * Only the model's 4-second analysis window is ever materialised: the
 * extractor seeks near the middle of the stream (or start when duration is
 * unknown) and decoding stops as soon as enough PCM is collected. Memory use
 * is O(window) regardless of file length - the naive whole-track buffer this
 * replaces OOM'd on long videos (160MB allocation against a 200MB heap).
 */
class AudioDecoder(private val context: Context) {

    fun decodeMono16k(uri: Uri, targetRate: Int = 16000, targetSamples: Int = 64000): FloatArray {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)

            var trackIndex = -1
            var format: MediaFormat? = null
            var durationUs = 0L
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                if (f.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                    trackIndex = i
                    format = f
                    if (f.containsKey(MediaFormat.KEY_DURATION)) {
                        durationUs = f.getLong(MediaFormat.KEY_DURATION)
                    }
                    break
                }
            }
            require(trackIndex >= 0) { "No audio track found" }
            extractor.selectTrack(trackIndex)

            var rate = format!!.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            val mime = format.getString(MediaFormat.KEY_MIME)!!

            // How many interleaved-native-rate samples equal one 16k window.
            fun neededNative() =
                ceil(targetSamples.toLong() * rate.toDouble() / targetRate).toInt()

            // Start near the centre of the track so the analysed snippet is
            // representative; fall back to the beginning when untimed.
            if (durationUs > 4_200_000L) {
                extractor.seekTo(durationUs / 2 - 2_000_000L,
                    MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
            }

            // ---- RAW PCM passthrough (WAV etc.): no MediaCodec exists for
            // ---- audio/raw; read decoded samples straight from the source.
            if (mime == MediaFormat.MIMETYPE_AUDIO_RAW) {
                val chunks = ArrayList<FloatArray>(16)
                var filled = 0
                val needed = neededNative()
                val scratch = java.nio.ByteBuffer.allocateDirect(1 shl 16)
                    .order(java.nio.ByteOrder.nativeOrder())
                var guard = 0
                while (filled < needed && guard++ < 10_000) {
                    scratch.clear()
                    val n = extractor.readSampleData(scratch, 0)
                    if (n <= 0) break
                    scratch.position(0)
                    scratch.limit(n)
                    val mono = toMonoFloat(scratch, channels)
                    chunks.add(mono)
                    filled += mono.size
                    extractor.advance()
                }
                val pcm = FloatArray(min(filled, needed))
                var pos = 0
                for (c in chunks) {
                    if (pos >= pcm.size) break
                    val k = min(c.size, pcm.size - pos)
                    System.arraycopy(c, 0, pcm, pos, k)
                    pos += k
                }
                return PcmUtil.centerCropOrPad(
                    PcmUtil.resampleLinear(pcm, rate, targetRate), targetSamples)
            }

            codec = MediaCodec.createDecoderByType(mime).apply {
                configure(format, null, null, 0)
                start()
            }

            val collected = ArrayList<FloatArray>(16)
            var filled = 0
            val info = MediaCodec.BufferInfo()
            var inputDone = false
            var outputDone = false
            var guard = 0
            var needed = neededNative()

            while (!outputDone && filled < needed && guard++ < 50_000) {
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
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val of = codec.outputFormat
                        rate = of.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        channels = of.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        needed = neededNative()
                    }
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    else -> if (outIdx >= 0 && info.size > 0) {
                        val buf = codec.getOutputBuffer(outIdx)!!
                        buf.position(info.offset)
                        buf.limit(info.offset + info.size)
                        val mono = toMonoFloat(buf, channels)
                        collected.add(mono)
                        filled += mono.size
                        codec.releaseOutputBuffer(outIdx, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            outputDone = true
                        }
                    } else if (outIdx >= 0) {
                        codec.releaseOutputBuffer(outIdx, false)
                    }
                }
            }

            // Merge only what we kept (a few hundred KB at most).
            val pcm = FloatArray(min(filled, needed))
            var pos = 0
            for (c in collected) {
                if (pos >= pcm.size) break
                val n = min(c.size, pcm.size - pos)
                System.arraycopy(c, 0, pcm, pos, n)
                pos += n
            }

            val at16k = PcmUtil.resampleLinear(pcm, rate, targetRate)
            return PcmUtil.centerCropOrPad(at16k, targetSamples)
        } finally {
            runCatching { codec?.stop(); codec?.release() }
            runCatching { extractor.release() }
        }
    }

    private fun toMonoFloat(pcmBuf: java.nio.ByteBuffer, channels: Int): FloatArray {
        pcmBuf.order(java.nio.ByteOrder.LITTLE_ENDIAN)
        val shortCount = pcmBuf.remaining() / 2
        val shorts = ShortArray(shortCount)
        pcmBuf.asShortBuffer().get(shorts)

        val frames = shortCount / channels
        val out = FloatArray(frames)
        var s = 0
        for (f in 0 until frames) {
            var acc = 0f
            for (c in 0 until channels) {
                acc += shorts[s]; s++
            }
            out[f] = acc / channels / 32768f
        }
        return out
    }
}

/** Pure PCM math, JVM-testable without Android types. */
internal object PcmUtil {

    fun resampleLinear(x: FloatArray, srcRate: Int, dstRate: Int): FloatArray {
        if (srcRate == dstRate || x.isEmpty()) return x
        val outLen = x.size.toLong() * dstRate / srcRate
        val out = FloatArray(outLen.toInt())
        val step = srcRate.toDouble() / dstRate
        for (i in out.indices) {
            val p = i * step
            val i0 = kotlin.math.floor(p).toInt().coerceIn(0, x.size - 1)
            val i1 = (i0 + 1).coerceAtMost(x.size - 1)
            val frac = (p - kotlin.math.floor(p)).toFloat()
            out[i] = x[i0] * (1f - frac) + x[i1] * frac
        }
        return out
    }

    fun centerCropOrPad(x: FloatArray, target: Int): FloatArray {
        if (x.size == target) return x
        val out = FloatArray(target)
        if (x.size > target) {
            val start = (x.size - target) / 2
            System.arraycopy(x, start, out, 0, target)
        } else {
            System.arraycopy(x, 0, out, 0, x.size)
        }
        return out
    }
}
