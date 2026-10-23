package com.onndetect.detection.audio

/**
 * F4 enrollment gate: measures how much of the clip actually carries
 * voiced speech. Enrolling a family voiceprint requires >= 8 s of voiced
 * audio; silence, background stretches and music do not count.
 *
 * Frame model: 20 ms frames, a frame is VOICED when its RMS exceeds
 * 0.15x the clip's 90th-percentile frame energy AND its zero-crossing
 * rate stays below 0.35 (voiced glottal excitation is low-ZCR). This is
 * deliberately conservative: a frame that is merely loud (claps, taps)
 * is not voiced speech.
 */
object VoicedDurationGate {

    const val MIN_VOICED_MS = 8_000L
    private const val FRAME_MS = 20

    /** Total voiced duration in ms. */
    fun voicedMs(pcm: FloatArray, sampleRate: Int = 16_000): Long {
        if (pcm.isEmpty()) return 0L
        val frameLen = (sampleRate * FRAME_MS / 1000).coerceAtLeast(64)
        val frames = pcm.size / frameLen
        if (frames < 4) return 0L

        val energies = FloatArray(frames)
        val zcrs = FloatArray(frames)
        for (f in 0 until frames) {
            val base = f * frameLen
            var eSq = 0.0
            var zc = 0
            for (i in 0 until frameLen) {
                val s = pcm[base + i]
                if (!s.isFinite()) return 0L
                eSq += s.toDouble() * s.toDouble()
                if (i > 0 && (s >= 0f) != (pcm[base + i - 1] >= 0f)) zc++
            }
            energies[f] = kotlin.math.sqrt(eSq / frameLen).toFloat()
            zcrs[f] = zc.toFloat() / (frameLen - 1)
        }

        // 90th percentile energy (voiced frames sit in the loud decile).
        val sorted = energies.sortedArray()
        val p90 = sorted[(sorted.size * 9 / 10).coerceAtMost(sorted.size - 1)]
        if (p90 <= 1e-5f) return 0L

        var voicedFrames = 0
        for (f in 0 until frames) {
            if (energies[f] >= 0.15f * p90 && zcrs[f] < 0.35f) voicedFrames++
        }
        return voicedFrames.toLong() * FRAME_MS
    }

    fun passes(pcm: FloatArray, sampleRate: Int = 16_000): Boolean =
        voicedMs(pcm, sampleRate) >= MIN_VOICED_MS
}
