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
    fun audioSealBinomialTailIsCorrect() {
        // The real AudioSeal graph is NOT in the release yet (its weights
        // are behind a gated Hugging Face repository), so the detector's
        // device-level behaviour is asserted by the instrumented test. What
        // is asserted here is the STATISTIC, which is where the previous
        // implementation was wrong: it took the maximum correlation over
        // ~100 frames and treated it as a single Gaussian tail.
        //
        // The null hypothesis is a RANDOM 32-bit message, so a random
        // message agrees with the expected pattern on about 16 bits. Under
        // the two-sided sign test:
        //   16 agreeing bits -> p = 1.0     (a coin flip, not evidence)
        //   24 agreeing bits -> p = 0.007
        //   32 agreeing bits -> p = 4.7e-10
        assertEquals(1.0, AudioSealDetector.binomialTwoTailed(16, 32), 1e-9)
        assertTrue(AudioSealDetector.binomialTwoTailed(32, 32) < 1e-9)
        // More agreeing bits must be strictly more significant.
        assertTrue(
            AudioSealDetector.binomialTwoTailed(24, 32) < AudioSealDetector.binomialTwoTailed(16, 32)
        )
        assertTrue(
            AudioSealDetector.binomialTwoTailed(28, 32) < AudioSealDetector.binomialTwoTailed(24, 32)
        )
        // A degenerate input is not significant.
        assertEquals(1.0, AudioSealDetector.binomialTwoTailed(0, 0), 1e-9)
    }
}
