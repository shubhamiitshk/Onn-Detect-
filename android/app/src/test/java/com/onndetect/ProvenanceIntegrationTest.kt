package com.onndetect

import com.onndetect.detection.provenance.AudioSealDetector
import com.onndetect.detection.provenance.C2PAManifestChecker
import com.onndetect.detection.provenance.C2PAVerifier
import com.onndetect.detection.provenance.ProvenanceEvidence
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream

class ProvenanceIntegrationTest {

    private fun jpegWithApp11(payload: ByteArray): ByteArray {
        val out = ByteArrayOutputStream()
        out.write(byteArrayOf(0xFF.toByte(), 0xD8.toByte())) // SOI
        out.write(byteArrayOf(0xFF.toByte(), 0xEB.toByte())) // APP11
        val length = payload.size + 2
        out.write((length shr 8) and 0xFF)
        out.write(length and 0xFF)
        out.write(payload)
        out.write(byteArrayOf(0xFF.toByte(), 0xD9.toByte())) // EOI
        return out.toByteArray()
    }

    @Test
    fun c2paVerifierDetectsAndValidatesAiClaim() {
        val payload = "c2patrainedAlgorithmicMedia".toByteArray(Charsets.ISO_8859_1)
        val jpegBytes = jpegWithApp11(payload)
        val verifier = C2PAVerifier(C2PAManifestChecker())
        val result = verifier.verifyStream(ByteArrayInputStream(jpegBytes))

        assertTrue(result.evidence is ProvenanceEvidence.C2PAClaim)
        val claim = result.evidence as ProvenanceEvidence.C2PAClaim
        assertTrue("Manifest should be present", claim.manifestPresent)
        assertTrue("Signature should be verified offline", claim.signed)
        assertEquals("trainedAlgorithmicMedia", claim.digitalSourceType)
        assertTrue("Should confirm AI involvement", claim.confirmsAiInvolvement)
    }

    @Test
    fun c2paVerifierIgnoresPlainJpeg() {
        val plainJpeg = byteArrayOf(
            0xFF.toByte(), 0xD8.toByte(),
            0xFF.toByte(), 0xE0.toByte(), 0x00, 0x10,
            0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
            0xFF.toByte(), 0xD9.toByte(),
        )
        val verifier = C2PAVerifier(C2PAManifestChecker())
        val result = verifier.verifyStream(ByteArrayInputStream(plainJpeg))
        assertEquals(ProvenanceEvidence.Absent, result.evidence)
    }

    @Test
    fun audioSealDetectorRejectsSilenceAndRandomNoise() {
        val detector = AudioSealDetector()
        val silence = FloatArray(16000) { 0f }
        assertEquals(ProvenanceEvidence.Absent, detector.analyze(silence))

        val randomNoise = FloatArray(16000) { ((it % 10) - 5) * 0.01f }
        val noiseEvidence = detector.analyze(randomNoise)
        assertEquals(ProvenanceEvidence.Absent, noiseEvidence)
    }

    @Test
    fun audioSealDetectorErfcAccuracy() {
        val detector = AudioSealDetector()
        assertEquals(1.0, detector.approximateErfc(0.0), 1e-5)
        // erfc(3.0) ≈ 2.209e-5
        val erfc3 = detector.approximateErfc(3.0)
        assertTrue("erfc(3.0) should be small (~2.2e-5)", erfc3 in 1e-5..5e-5)
    }
}
