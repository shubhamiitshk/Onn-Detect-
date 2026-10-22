package com.onndetect.detection.audio

import com.onndetect.data.local.VoiceProfileEntity
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class SpeakerVerifierTest {

    private val verifier = SpeakerVerifier()

    private fun generateSineWave(freqHz: Double, durationSeconds: Double, sampleRate: Int = 16000): FloatArray {
        val totalSamples = (durationSeconds * sampleRate).toInt()
        val buffer = FloatArray(totalSamples)
        for (i in 0 until totalSamples) {
            val t = i.toDouble() / sampleRate
            buffer[i] = (sin(2.0 * PI * freqHz * t) * 0.7).toFloat()
        }
        return buffer
    }

    private fun generateModulatedVoice(baseFreq: Double, durationSeconds: Double = 1.0): FloatArray {
        val totalSamples = (durationSeconds * 16000).toInt()
        val buffer = FloatArray(totalSamples)
        for (i in 0 until totalSamples) {
            val t = i.toDouble() / 16000
            val f = baseFreq + 25.0 * sin(2.0 * PI * 4.0 * t)
            buffer[i] = (sin(2.0 * PI * f * t) * 0.5 + sin(2.0 * PI * 2 * f * t) * 0.25).toFloat()
        }
        return buffer
    }

    @Test
    fun `extractVoiceprint produces 160-dimensional unit normalized vector`() {
        val pcm = generateModulatedVoice(220.0, 1.0)
        val embedding = verifier.extractVoiceprint(pcm)

        assertEquals(160, embedding.size)
        var normSq = 0.0
        for (v in embedding) normSq += v * v
        assertEquals("Voiceprint should be unit L2 normalized", 1.0, kotlin.math.sqrt(normSq), 1e-3)
    }

    @Test
    fun `identical audio produces exact cosine match of 1_0`() {
        val pcm = generateModulatedVoice(300.0, 1.0)
        val vec1 = verifier.extractVoiceprint(pcm)
        val vec2 = verifier.extractVoiceprint(pcm)

        val sim = verifier.computeCosineSimilarity(vec1, vec2)
        assertEquals(1.0f, sim, 1e-4f)
    }

    @Test
    fun `distinct acoustic voiceprints have lower cosine similarity`() {
        val pcmA = generateModulatedVoice(130.0, 1.0) // Deep male voice pitch
        val pcmB = generateModulatedVoice(450.0, 1.0) // Child/high female pitch

        val vecA = verifier.extractVoiceprint(pcmA)
        val vecB = verifier.extractVoiceprint(pcmB)

        val sim = verifier.computeCosineSimilarity(vecA, vecB)
        assertTrue("Distinct pitches should have similarity well below 0.80", sim < 0.70f)
    }

    @Test
    fun `matchAgainstVault succeeds when incoming speech matches enrolled family profile`() {
        val momAudio = generateModulatedVoice(260.0, 1.2)
        val momEmbedding = verifier.extractVoiceprint(momAudio)

        val momProfile = VoiceProfileEntity.fromEmbeddingArray(
            id = "mom-uuid",
            contactName = "Mom",
            relationship = "Mother",
            enrolledAtMs = System.currentTimeMillis(),
            sampleDurationMs = 1200L,
            embedding = momEmbedding,
        )

        // Incoming audio from Mom with slight noise
        val incomingMomAudio = momAudio.copyOf()
        for (i in incomingMomAudio.indices step 10) {
            incomingMomAudio[i] += 0.01f
        }

        val report = verifier.matchAgainstVault(incomingMomAudio, listOf(momProfile))

        assertTrue("Should confirm enrollment exists", report.hasEnrolledProfiles)
        assertTrue("Should match Mom's voiceprint", report.isMatch)
        assertEquals("Mom", report.matchedContactName)
        assertEquals("Mother", report.matchedRelationship)
        assertTrue("Similarity should exceed threshold", report.maxSimilarity >= 0.80f)
        assertEquals(VoiceMatchConfidence.HIGH_MATCH, report.confidence)
    }

    @Test
    fun `matchAgainstVault flags mismatch when incoming speech does not match enrolled profile`() {
        val dadAudio = generateModulatedVoice(120.0, 1.0)
        val dadEmbedding = verifier.extractVoiceprint(dadAudio)

        val dadProfile = VoiceProfileEntity.fromEmbeddingArray(
            id = "dad-uuid",
            contactName = "Dad",
            relationship = "Father",
            enrolledAtMs = System.currentTimeMillis(),
            sampleDurationMs = 1000L,
            embedding = dadEmbedding,
        )

        // Imposter speaking at different pitch
        val imposterAudio = generateModulatedVoice(400.0, 1.0)

        val report = verifier.matchAgainstVault(imposterAudio, listOf(dadProfile))

        assertTrue(report.hasEnrolledProfiles)
        assertFalse("Imposter should NOT match Dad", report.isMatch)
        assertNull(report.matchedContactName)
        assertEquals(VoiceMatchConfidence.NO_MATCH, report.confidence)
        assertTrue(report.flags.any { it.contains("does not match") })
    }

    @Test
    fun `matchAgainstVault reports NO_PROFILES_ENROLLED when vault is empty`() {
        val audio = generateModulatedVoice(200.0, 1.0)
        val report = verifier.matchAgainstVault(audio, emptyList())

        assertFalse(report.hasEnrolledProfiles)
        assertFalse(report.isMatch)
        assertEquals(VoiceMatchConfidence.NO_PROFILES_ENROLLED, report.confidence)
    }

    @Test
    fun `VoiceProfileEntity converts to and from embedding float array accurately`() {
        val originalVec = floatArrayOf(0.123456f, -0.654321f, 0.987654f, 0.0f)
        val entity = VoiceProfileEntity.fromEmbeddingArray(
            id = "test-id",
            contactName = "Alice",
            relationship = "Sister",
            enrolledAtMs = 1000L,
            sampleDurationMs = 5000L,
            embedding = originalVec,
        )

        val recovered = entity.toEmbeddingArray()
        assertEquals(originalVec.size, recovered.size)
        for (i in originalVec.indices) {
            assertEquals(originalVec[i], recovered[i], 1e-5f)
        }
    }
}
