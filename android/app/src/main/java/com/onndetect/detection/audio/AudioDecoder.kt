package com.onndetect.detection.audio

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.SystemClock
import android.os.Build
import java.nio.ByteBuffer
import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/** Decode a timestamped, bounded window. No empty media is silently padded into success. */
class AudioDecoder(private val context: Context) {
    data class DecodedAudio(val pcm: FloatArray, val validSamples: Int, val sampleRate: Int,
        val startMs: Long, val analyzedDurationMs: Long, val totalDurationMs: Long,
        val rms: Float, val peak: Float, val clippedFraction: Float)
    fun decodeMono16k(uri: Uri, targetRate: Int = 16000, targetSamples: Int = 64000): FloatArray =
        decodeWindow(uri, targetRate, targetSamples).pcm

    fun decodeWindow(uri: Uri, targetRate: Int = 16000, targetSamples: Int = 64000,
        requestedStartMs: Long? = null, checkStop: () -> Unit = {}): DecodedAudio {
        // F4 raised the ceiling to 60 s @ 16 kHz for enrollment; the 20 s
        // wall-clock deadline below still bounds every decode.
        require(targetRate in 8000..48000 && targetSamples in 1..AudioClassifier.MAX_DECODE_SAMPLES)
        val deadline = SystemClock.elapsedRealtime() + 20_000
        fun check() {
            if (Thread.currentThread().isInterrupted) throw java.util.concurrent.CancellationException("Audio decode cancelled")
            checkStop()
            check(SystemClock.elapsedRealtime() <= deadline) { "Audio decoder deadline exceeded" }
        }
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)
            val track = (0 until extractor.trackCount).firstOrNull { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true }
                ?: throw IllegalArgumentException("No audio track found")
            val format = extractor.getTrackFormat(track)
            val durationUs = if (format.containsKey(MediaFormat.KEY_DURATION)) format.getLong(MediaFormat.KEY_DURATION) else 0L
            val windowUs = targetSamples.toLong() * 1_000_000 / targetRate
            val startUs = ((requestedStartMs?.times(1000) ?: max(0L, (durationUs - windowUs) / 2))).coerceIn(0L, max(0L, durationUs))
            extractor.selectTrack(track)
            if (startUs > 0) extractor.seekTo(startUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
            var rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            var encoding = if (format.containsKey(MediaFormat.KEY_PCM_ENCODING)) format.getInteger(MediaFormat.KEY_PCM_ENCODING) else 2
            val mime = format.getString(MediaFormat.KEY_MIME) ?: error("Audio MIME missing")
            require(rate in 1000..384000 && channels in 1..32)
            fun needed() = ceil(targetSamples.toDouble() * rate / targetRate).toInt()
            val chunks = ArrayList<FloatArray>()
            var filled = 0
            var firstUs: Long? = null
            fun append(buf: ByteBuffer, timestampUs: Long) {
                val decoded = PcmUtil.decodeInterleaved(buf, channels, encoding)
                val skip = ceil(max(0L, startUs - timestampUs).toDouble() * rate / 1_000_000).toInt().coerceAtMost(decoded.size)
                val take = min(decoded.size - skip, needed() - filled)
                if (take > 0) {
                    if (firstUs == null) firstUs = timestampUs + skip.toLong() * 1_000_000 / rate
                    chunks += decoded.copyOfRange(skip, skip + take)
                    filled += take
                }
            }
            if (mime == MediaFormat.MIMETYPE_AUDIO_RAW) {
                val rawBuffer = ByteBuffer.allocateDirect(4_194_304)
                while (filled < needed()) {
                    check()
                    // getSampleSize is API28+. On26/27 use a bounded packet
                    // buffer; oversized packets fail explicitly rather than
                    // raising the application's minimum supported SDK.
                    val size = if (Build.VERSION.SDK_INT >= 28) extractor.sampleSize else -2L
                    if (size == -1L) break
                    if (size >= 0) require(size in 1..4_194_304) { "PCM packet exceeds decode budget" }
                    val buffer = rawBuffer.apply { clear() }
                    val n = extractor.readSampleData(buffer, 0)
                    if (n <= 0) break
                    require(n <= buffer.capacity()) { "PCM packet exceeds decode budget" }
                    buffer.position(0); buffer.limit(n)
                    append(buffer, extractor.sampleTime)
                    if (!extractor.advance()) break
                }
            } else {
                codec = MediaCodec.createDecoderByType(mime)
                val decoder = codec!!
                decoder.configure(format, null, null, 0); decoder.start()
                val info = MediaCodec.BufferInfo()
                var inputDone = false
                var outputDone = false
                var lastProgress = SystemClock.elapsedRealtime()
                while (!outputDone && filled < needed()) {
                    check()
                    check(SystemClock.elapsedRealtime() - lastProgress <= 5000) { "Audio decoder stalled" }
                    if (!inputDone) {
                        val index = decoder.dequeueInputBuffer(10_000)
                        if (index >= 0) {
                            val buf = decoder.getInputBuffer(index) ?: error("Missing codec input")
                            val size = extractor.readSampleData(buf, 0)
                            if (size < 0) { decoder.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM); inputDone = true }
                            else { decoder.queueInputBuffer(index, 0, size, extractor.sampleTime, 0); extractor.advance() }
                            lastProgress = SystemClock.elapsedRealtime()
                        }
                    }
                    when (val index = decoder.dequeueOutputBuffer(info, 10_000)) {
                        MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            val output = decoder.outputFormat
                            val nextRate = output.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                            val nextChannels = output.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                            require(filled == 0 || (rate == nextRate && channels == nextChannels)) { "Audio format changed mid-window" }
                            rate = nextRate; channels = nextChannels
                            require(rate in 1000..384000 && channels in 1..32)
                            encoding = if (output.containsKey(MediaFormat.KEY_PCM_ENCODING)) output.getInteger(MediaFormat.KEY_PCM_ENCODING) else 2
                        }
                        MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                        else -> if (index >= 0) {
                            try {
                                if (info.size > 0) {
                                    val buffer = decoder.getOutputBuffer(index) ?: error("Missing codec output")
                                    buffer.position(info.offset); buffer.limit(info.offset + info.size)
                                    append(buffer, info.presentationTimeUs)
                                }
                                outputDone = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                                lastProgress = SystemClock.elapsedRealtime()
                            } finally { decoder.releaseOutputBuffer(index, false) }
                        }
                    }
                }
            }
            require(filled > 0) { "No decoded audio samples" }
            val native = FloatArray(filled)
            var offset = 0
            for (chunk in chunks) { chunk.copyInto(native, offset); offset += chunk.size }
            check()
            val resampled = PcmUtil.resample(native, rate, targetRate)
            val valid = min(resampled.size, targetSamples)
            val observed = resampled.copyOf(valid)
            val rms = sqrt(observed.fold(0.0) { sum, x -> sum + x.toDouble() * x } / valid).toFloat()
            val peak = observed.maxOf { kotlin.math.abs(it) }
            val clipped = observed.count { kotlin.math.abs(it) >= .999f }.toFloat() / valid
            check()
            return DecodedAudio(PcmUtil.centerCropOrPad(observed, targetSamples), valid, targetRate,
                (firstUs ?: startUs) / 1000, valid.toLong() * 1000 / targetRate, durationUs / 1000, rms, peak, clipped)
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            extractor.release()
        }
    }
}

/** Pure PCM math, JVM-testable without Android types. */
internal object PcmUtil {
    /** Android PCM encoding values, kept literal so this core runs on the JVM. */
    fun decodeInterleaved(buffer: java.nio.ByteBuffer, channels: Int, encoding: Int): FloatArray {
        require(channels in 1..32)
        val bytes = when (encoding) { 3 -> 1; 2 -> 2; 21 -> 3; 4, 22 -> 4; else -> error("Unsupported PCM encoding: $encoding") }
        require(buffer.remaining() % (bytes * channels) == 0) { "Truncated PCM frame" }
        buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(buffer.remaining() / bytes / channels)
        for (frame in out.indices) {
            var sum = 0.0
            repeat(channels) {
                val sample = when (encoding) {
                    3 -> ((buffer.get().toInt() and 255) - 128) / 128f
                    2 -> buffer.short / 32768f
                    4 -> buffer.float
                    21 -> {
                        val word = (buffer.get().toInt() and 255) or ((buffer.get().toInt() and 255) shl 8) or ((buffer.get().toInt() and 255) shl 16)
                        (word shl 8 shr 8) / 8388608f
                    }
                    else -> buffer.int.toDouble().div(2147483648.0).toFloat()
                }
                require(sample.isFinite()) { "Non-finite PCM sample" }
                sum += sample.coerceIn(-1f, 1f)
            }
            out[frame] = (sum / channels).toFloat()
        }
        return out
    }


    /** Shared quadratic-window sinc v1; deliberately not torchaudio Hann. */
    fun resample(
        x: FloatArray,
        srcRate: Int,
        dstRate: Int,
        lowpassFilterWidth: Int = 6,
        rolloff: Double = 0.99,
    ): FloatArray {
        require(srcRate in 1000..384000 && dstRate in 1000..384000 && x.all { it.isFinite() })
        require(lowpassFilterWidth in 1..64 && rolloff > 0.0 && rolloff <= 1.0)
        if (srcRate == dstRate || x.isEmpty()) return x

        val ratio = dstRate.toDouble() / srcRate
        // Ceiling division: the output must cover the full input duration.
        val outSize = ceil(x.size * ratio)
        require(outSize <= 10_000_000) { "Resampling budget exceeded" }
        val outLen = outSize.toInt()

        // Cutoff as a fraction of the source Nyquist rate: roll off at
        // rolloff*ratio when downsampling so nothing above the new Nyquist
        // survives; full bandwidth when upsampling.
        val cutoff = rolloff * minOf(1.0, ratio)
        // Kernel support widens when downsampling (more source samples per
        // output sample); upsampling needs only the base width.
        val half = ceil(lowpassFilterWidth * maxOf(1.0, 1.0 / ratio)).toInt()

        // Reflect-padded input so edges do not ring; constant fallback for
        // signals shorter than the kernel.
        val useReflect = x.size > half
        val padded = FloatArray(x.size + 2 * half)
        for (i in padded.indices) {
            val src = i - half
            padded[i] = when {
                src in 0 until x.size -> x[src]
                !useReflect -> 0f
                src < 0 -> x[-src]
                else -> x[2 * (x.size - 1) - src]
            }
        }

        val out = FloatArray(outLen)
        for (i in 0 until outLen) {
            if (i % 1024 == 0 && Thread.currentThread().isInterrupted) throw java.util.concurrent.CancellationException("Audio resampling cancelled")
            val p = i / ratio                       // exact source position
            val i0 = floor(p).toInt()
            var acc = 0.0
            var wsum = 0.0
            for (k in -half..half) {
                val t = (i0 + k) - p                // fractional offset
                val hk = k.toDouble() / (half + 1.0)
                val quadraticWindow = maxOf(0.0, 1.0 - hk * hk)
                val w = cutoff * sinc(cutoff * t) * quadraticWindow
                acc += w * padded[i0 + k + half]
                wsum += w
            }
            out[i] = if (wsum != 0.0) (acc / wsum).toFloat() else 0f
        }
        return out
    }

    private fun sinc(v: Double): Double {
        val d = PI * v
        return if (kotlin.math.abs(d) < 1e-12) 1.0 else sin(d) / d
    }

    /**
     * Linear interpolation. RETAINED FOR COMPARISON ONLY - the parity script
     * uses it to demonstrate the aliasing this replaced. Do not call from the
     * scan path; use [resample].
     */
    fun resampleLinear(x: FloatArray, srcRate: Int, dstRate: Int): FloatArray {
        require(srcRate > 0 && dstRate > 0 && x.all { it.isFinite() })
        if (srcRate == dstRate || x.isEmpty()) return x
        val outLen = x.size.toLong() * dstRate / srcRate
        val out = FloatArray(outLen.toInt())
        val step = srcRate.toDouble() / dstRate
        for (i in out.indices) {
            val p = i * step
            val i0 = floor(p).toInt().coerceIn(0, x.size - 1)
            val i1 = (i0 + 1).coerceAtMost(x.size - 1)
            val frac = (p - floor(p)).toFloat()
            out[i] = x[i0] * (1f - frac) + x[i1] * frac
        }
        return out
    }

    fun centerCropOrPad(x: FloatArray, target: Int): FloatArray {
        require(target > 0 && x.all { it.isFinite() })
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
