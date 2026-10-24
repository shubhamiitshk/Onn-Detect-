package com.onndetect.detection.audio

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class AudioGatingTest {

    private fun generateSineWave(freqHz: Double, durationSeconds: Double, sampleRate: Int = 16000): FloatArray {
        val totalSamples = (durationSeconds * sampleRate).toInt()
        val buffer = FloatArray(totalSamples)
        for (i in 0 until totalSamples) {
            val t = i.toDouble() / sampleRate
            buffer[i] = (sin(2.0 * PI * freqHz * t) * 0.7).toFloat()
        }
        return buffer
    }

    private fun generateRealisticSpeech(durationSeconds: Double = 4.0, sampleRate: Int = 16000): FloatArray {
        val totalSamples = (durationSeconds * sampleRate).toInt()
        val buffer = FloatArray(totalSamples)
        val random = java.util.Random(42)
        val frameSamples = sampleRate / 50 // 20ms frames
        val numFrames = totalSamples / frameSamples
        for (f in 0 until numFrames) {
            if (f % 10 in 0..1) {
                // Natural speech inter-word breath/pause (silence)
                continue
            }
            val isVoiced = (f % 5) < 3 // Voiced glottal frames vs unvoiced fricatives
            val base = f * frameSamples
            val pitch = 150.0 + (f % 7) * 20.0
            for (i in 0 until frameSamples) {
                val t = (base + i).toDouble() / sampleRate
                if (isVoiced) {
                    val glottal = (sin(2.0 * PI * pitch * t) * 0.5 + sin(2.0 * PI * 2 * pitch * t) * 0.25).toFloat()
                    buffer[base + i] = glottal
                } else {
                    buffer[base + i] = (random.nextGaussian() * 0.15).toFloat()
                }
            }
        }
        return buffer
    }

    @Test
    fun stationaryToneIsRejectedBySpeechGate() {
        val tone = generateSineWave(1000.0, 4.0)
        assertFalse("Stationary 1kHz tone must not be classified as speech-like",
            SpeechContentGate.isSpeechLike(tone, tone.size, 16000))
    }

    @Test
    fun pureSilenceIsRejectedBySpeechGateAndVoicedGate() {
        val silence = FloatArray(64000)
        assertFalse("Pure silence must not be classified as speech-like",
            SpeechContentGate.isSpeechLike(silence, silence.size, 16000))
        assertEquals(0L, VoicedDurationGate.voicedMs(silence, 16000))
    }

    @Test
    fun modulatedSpeechPassesSpeechGateAndHasVoicedDuration() {
        val speech = generateRealisticSpeech(4.0, 16000)
        assertTrue("Speech signal with phonemic modulation must pass speech content gate",
            SpeechContentGate.isSpeechLike(speech, speech.size, 16000))
        val voicedMs = VoicedDurationGate.voicedMs(speech, 16000)
        assertTrue("Speech signal should have at least 1000ms voiced frames, got $voicedMs",
            voicedMs >= 1000L)
    }
}
