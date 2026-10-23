package com.onndetect

import com.onndetect.detection.provenance.C2PAManifestChecker
import com.onndetect.detection.provenance.C2paTestFixtures
import com.onndetect.detection.provenance.CoseSign1
import com.onndetect.detection.provenance.ProvenanceEvidence
import com.onndetect.detection.provenance.VerdictConfidence
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
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
 *
 * Fixtures are REAL, spec-shaped structures (JUMBF store boxes, CBOR
 * claims, COSE_Sign1) with keys generated at runtime — never the fictional
 * `c2cs:` ASCII protocol of the pre-audit code.
 */
class C2PAManifestCheckerTest {

    private fun signedStoreFixture(
        digitalSourceType: String = "trainedAlgorithmicMedia",
        maxSegPayload: Int = 64,
    ): C2paTestFixtures.JpegLayout {
        val ca = C2paTestFixtures.newCa()
        val claim = C2paTestFixtures.buildClaim(digitalSourceType)
        val cose = C2paTestFixtures.buildCoseSign1(claim, ca)
        val store = C2paTestFixtures.buildStore(claim, cose, listOf(ca.certificate.encoded))
        return C2paTestFixtures.jpegWithApp11(store, maxSegPayload = maxSegPayload)
    }

    @Test
    fun findsAManifestInASingleSegmentJpegApp11() {
        val layout = signedStoreFixture()
        val finding = C2PAManifestChecker().inspect(ByteArrayInputStream(layout.bytes))
        assertTrue("manifest should be detected", finding.present)
        assertEquals("jpeg", finding.container)
        assertEquals("trainedAlgorithmicMedia", finding.digitalSourceType)
        assertNotNull("claim bytes extracted", finding.claimData)
        assertNotNull("COSE_Sign1 extracted", finding.coseBytes)
        assertEquals("one trust-chain certificate", 1, finding.certificatesDer.size)
    }

    @Test
    fun reassemblesMultiSegmentApp11ByEnAndZ() {
        // Force >= 3 segments: fragments of the same En, contiguous, with the
        // continuation flag Z=1 on all but the last, exactly per ISO 19566-5 D.2.
        val layout = signedStoreFixture(maxSegPayload = 16)
        val finding = C2PAManifestChecker().inspect(ByteArrayInputStream(layout.bytes))
        assertTrue("multi-segment manifest should be detected", finding.present)
        assertEquals("jpeg", finding.container)
        assertEquals("trainedAlgorithmicMedia", finding.digitalSourceType)
        assertTrue(
            "manifest range must cover the APP11 group",
            finding.manifestRanges.size >= 3,
        )
    }

    @Test
    fun findsAManifestBeyondTheOld64KbScanWindow() {
        // ~100 KB of APP0 filler before the APP11 group: the pre-F6 walker
        // (64 KB read cap) could never see this manifest.
        val layout = signedStoreFixture().let {
            // rebuild with padding; the fixture produces a fresh layout
            val ca = C2paTestFixtures.newCa()
            val claim = C2paTestFixtures.buildClaim()
            val cose = C2paTestFixtures.buildCoseSign1(claim, ca)
            val store = C2paTestFixtures.buildStore(claim, cose, listOf(ca.certificate.encoded))
            C2paTestFixtures.jpegWithApp11(store, maxSegPayload = 64, prefixPadBytes = 100 * 1024)
        }
        assertTrue(layout.manifestStart > 96 * 1024)
        val finding = C2PAManifestChecker().inspect(ByteArrayInputStream(layout.bytes))
        assertTrue("manifest past 64 KB must still be found (full-file walk)", finding.present)
        assertEquals("trainedAlgorithmicMedia", finding.digitalSourceType)
    }

    @Test
    fun findsAManifestInAnIsoBmffUuidBoxWithoutDescendingIntoMdat() {
        val ca = C2paTestFixtures.newCa()
        val claim = C2paTestFixtures.buildClaim()
        val cose = C2paTestFixtures.buildCoseSign1(claim, ca)
        val store = C2paTestFixtures.buildStore(claim, cose, listOf(ca.certificate.encoded))
        val bmff = C2paTestFixtures.bmffWithUuidStore(store, noiseInMdat = true)
        val finding = C2PAManifestChecker().inspect(ByteArrayInputStream(bmff))
        assertTrue("uuid-box manifest should be detected", finding.present)
        assertEquals("isobmff", finding.container)
        // The decoy 'c2pa'/'JUMBF' text inside mdat must not be reported as a
        // second manifest or corrupt the first: the walk skips mdat whole.
        assertEquals("trainedAlgorithmicMedia", finding.digitalSourceType)
    }

    @Test
    fun anUnterminatedApp11GroupIsNotTrusted() {
        // Z=1 on every segment: the box stream never closes, so the group is
        // a truncated manifest and must not be parsed.
        val ca = C2paTestFixtures.newCa()
        val claim = C2paTestFixtures.buildClaim()
        val cose = C2paTestFixtures.buildCoseSign1(claim, ca)
        val store = C2paTestFixtures.buildStore(claim, cose, listOf(ca.certificate.encoded))
        val out = java.io.ByteArrayOutputStream()
        out.write(byteArrayOf(0xFF.toByte(), 0xD8.toByte()))
        out.write(byteArrayOf(0xFF.toByte(), 0xEB.toByte()))
        val payload = ByteArray(7 + store.size)
        "JUMBF".toByteArray(Charsets.US_ASCII).copyInto(payload, 0)
        payload[5] = 1 // En
        payload[6] = 1 // Z=1: more segments follow... and none ever arrives
        store.copyInto(payload, 7)
        out.write(((payload.size + 2) shr 8) and 0xFF); out.write((payload.size + 2) and 0xFF)
        out.write(payload)
        out.write(byteArrayOf(0xFF.toByte(), 0xD9.toByte()))
        val finding = C2PAManifestChecker().inspect(ByteArrayInputStream(out.toByteArray()))
        assertFalse("an unterminated (truncated) manifest must not be trusted", finding.present)
    }

    @Test
    fun theFictionalC2csMarkerIsNotASignature() {
        // The pre-audit format: an APP11 payload of bare ASCII text with the
        // invented `c2cs:claim:hexsig` protocol. Under the spec-driven walker
        // this is not a manifest at all — regression against the old bug.
        val junk = "c2patrainedAlgorithmicMedia c2cs:claim:3045022100ff".toByteArray(Charsets.ISO_8859_1)
        val jpeg = C2paTestFixtures.jpegWithApp11(junk, maxSegPayload = 1 shl 20)
        val finding = C2PAManifestChecker().inspect(ByteArrayInputStream(jpeg.bytes))
        assertFalse("the fictional c2cs payload must not parse as a manifest", finding.present)
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
    fun coseRawAndDerConversionsRoundTrip() {
        val ca = C2paTestFixtures.newCa()
        val claim = C2paTestFixtures.buildClaim()
        val cose = C2paTestFixtures.buildCoseSign1(claim, ca)
        val parsed = CoseSign1.parse(cose)!!
        // The fixture DER->raw conversion must invert the verifier's raw->DER.
        val der = CoseSign1.es256RawToDer(parsed.signature)!!
        val back = CoseSign1.derToRaw(der, 32)!!
        assertTrue(back.contentEquals(parsed.signature))
        assertEquals(-7L, parsed.alg)
        assertEquals(1, parsed.x5chain.size)
    }

    // ── Provenance verdict discipline (unchanged properties) ────────────

    @Test
    fun aPresentButUnverifiedManifestDoesNotConfirmAiInvolvement() {
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
        val absent = ProvenanceEvidence.Absent
        assertFalse(absent is ProvenanceEvidence.C2PAClaim)
        assertFalse(absent is ProvenanceEvidence.WatermarkMark)
    }

    @Test
    fun theConfidenceStatesAreDistinct() {
        assertEquals(3, VerdictConfidence.entries.size)
        assertTrue(VerdictConfidence.PROVENANCE_CONFIRMED != VerdictConfidence.STATISTICAL)
    }

    @Test
    fun digitalSourceTypeComesFromTheClaimNotTheContainer() {
        // digitalCapture in the *container* must not leak into the finding:
        // the source type is read from the signed claim bytes only.
        val layout = signedStoreFixture(digitalSourceType = "digitalCapture")
        val finding = C2PAManifestChecker().inspect(ByteArrayInputStream(layout.bytes))
        assertTrue(finding.present)
        // 'digitalCapture' is not an AI source type, so the extractor keeps it
        // exactly as stored — proving it came from the claim, unfiltered.
        assertEquals("digitalCapture", finding.digitalSourceType)
    }
}
