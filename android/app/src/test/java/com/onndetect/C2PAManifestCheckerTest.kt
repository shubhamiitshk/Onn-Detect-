package com.onndetect

import com.onndetect.detection.provenance.C2PAManifestChecker
import com.onndetect.detection.provenance.ProvenanceEvidence
import com.onndetect.detection.provenance.VerdictConfidence
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream

/**
 * The C2PA detector and the provenance verdict discipline.
 *
 * The critical property is what the detector must NOT do: a manifest that
 * is merely PRESENT is not a claim. The C2PA specification's "exclusion
 * range" allows parts of an asset to sit outside the signed region, and an
 * unverified manifest can be forged, so only a VERIFIED signature chain
 * may raise a confirmed verdict.
 */
class C2PAManifestCheckerTest {

    /** Build a JPEG with an APP11 segment carrying the given payload. */
    private fun jpegWithApp11(payload: ByteArray): ByteArray {
        val out = java.io.ByteArrayOutputStream()
        out.write(byteArrayOf(0xFF.toByte(), 0xD8.toByte()))          // SOI
        out.write(byteArrayOf(0xFF.toByte(), 0xEB.toByte()))          // APP11
        val length = payload.size + 2
        out.write((length shr 8) and 0xFF)
        out.write(length and 0xFF)
        out.write(payload)
        out.write(byteArrayOf(0xFF.toByte(), 0xD9.toByte()))          // EOI
        return out.toByteArray()
    }

    private fun payload(vararg parts: String): ByteArray {
        val text = parts.joinToString("")
        return text.toByteArray(Charsets.ISO_8859_1)
    }

    @Test
    fun findsAManifestInAJpegApp11Segment() {
        val bytes = jpegWithApp11(payload("c2pa", "trainedAlgorithmicMedia"))
        val finding = C2PAManifestChecker().inspect(ByteArrayInputStream(bytes))
        assertTrue("manifest should be detected", finding.present)
        assertEquals("jpeg", finding.container)
        assertEquals("trainedAlgorithmicMedia", finding.digitalSourceType)
    }

    @Test
    fun reportsNoManifestForAPlainJpeg() {
        // SOI + a JFIF APP0 segment + EOI.
        val bytes = byteArrayOf(
            0xFF.toByte(), 0xD8.toByte(),
            0xFF.toByte(), 0xE0.toByte(), 0x00, 0x10,
            0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
            0xFF.toByte(), 0xD9.toByte(),
        )
        val finding = C2PAManifestChecker().inspect(ByteArrayInputStream(bytes))
        assertFalse("no manifest should be found", finding.present)
    }

    @Test
    fun reportsNoManifestForGarbage() {
        val finding = C2PAManifestChecker().inspect(ByteArrayInputStream(ByteArray(64) { 7 }))
        assertFalse(finding.present)
    }

    @Test
    fun handlesAnEmptyStream() {
        val finding = C2PAManifestChecker().inspect(ByteArrayInputStream(ByteArray(0)))
        assertFalse(finding.present)
    }

    @Test
    fun aPresentButUnverifiedManifestDoesNotConfirmAiInvolvement() {
        // This is the property the whole design turns on.
        val claim = ProvenanceEvidence.C2PAClaim(
            manifestPresent = true,
            signed = false,
            digitalSourceType = "trainedAlgorithmicMedia",
        )
        assertFalse(
            "an unverified manifest must not confirm AI involvement",
            claim.confirmsAiInvolvement,
        )
    }

    @Test
    fun aVerifiedManifestDoesConfirmAiInvolvement() {
        val claim = ProvenanceEvidence.C2PAClaim(
            manifestPresent = true,
            signed = true,
            digitalSourceType = "trainedAlgorithmicMedia",
        )
        assertTrue(claim.confirmsAiInvolvement)
    }

    @Test
    fun aVerifiedManifestWithANonGenerativeSourceTypeDoesNotConfirm() {
        val claim = ProvenanceEvidence.C2PAClaim(
            manifestPresent = true,
            signed = true,
            digitalSourceType = "digitalCapture",
        )
        assertFalse(claim.confirmsAiInvolvement)
    }

    @Test
    fun aSignificantWatermarkDetectionConfirms() {
        val mark = ProvenanceEvidence.WatermarkMark(detected = true, kind = "audioseal", pValue = 1e-9)
        assertTrue(mark.confirmsAiInvolvement)
    }

    @Test
    fun anInsignificantWatermarkDetectionDoesNotConfirm() {
        val mark = ProvenanceEvidence.WatermarkMark(detected = true, kind = "audioseal", pValue = 0.4)
        assertFalse(mark.confirmsAiInvolvement)
    }

    @Test
    fun absenceIsNotEvidenceOfAuthenticity() {
        // Absent carries no claim in either direction.
        val absent = ProvenanceEvidence.Absent
        assertFalse(absent is ProvenanceEvidence.C2PAClaim)
        assertFalse(absent is ProvenanceEvidence.WatermarkMark)
    }

    @Test
    fun theConfidenceStatesAreDistinct() {
        assertEquals(3, VerdictConfidence.entries.size)
        assertTrue(VerdictConfidence.PROVENANCE_CONFIRMED != VerdictConfidence.STATISTICAL)
    }
}
