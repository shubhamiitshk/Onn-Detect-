package com.onndetect

import com.onndetect.detection.provenance.AudioSealDetector
import com.onndetect.detection.provenance.C2PAManifestChecker
import com.onndetect.detection.provenance.C2PATrustStore
import com.onndetect.detection.provenance.C2PAVerifier
import com.onndetect.detection.provenance.C2paTestFixtures
import com.onndetect.detection.provenance.ProvenanceEvidence
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream

/**
 * End-to-end C2PA verification over REAL spec-shaped fixtures, against the
 * F2 fail-closed rule:
 *
 *   signed = hashVerified && coseSignatureValid && certChainValid
 *
 * Keys are generated at runtime; the trust list is the runtime CA's PEM
 * (mirroring how DI loads the production list from assets).
 */
class ProvenanceIntegrationTest {

    /** A converged layout: the data-hash inside the store matches THIS file. */
    private class SignedJpeg(val layout: C2paTestFixtures.JpegLayout, val ca: C2paTestFixtures.TestCa)

    /**
     * Builds a JPEG whose manifest store is fully consistent: the
     * c2pa.hash.data exclusion range and digest match the final file layout.
     * The store/segment layout is rebuilt until it converges (the JSON inside
     * the store carries byte offsets that depend on the store's own size).
     */
    private fun buildSignedJpeg(
        maxSegPayload: Int = 24,
        prefixPadBytes: Int = 2048,
        signer: (C2paTestFixtures.TestCa) -> C2paTestFixtures.TestCa = { it },
    ): SignedJpeg {
        val ca = C2paTestFixtures.newCa()
        val signingCa = signer(ca)
        var prev: C2paTestFixtures.JpegLayout? = null
        var stable: C2paTestFixtures.JpegLayout? = null

        for (pass in 0 until 8) {
            val exclusions = prev?.let { listOf(it.manifestStart to it.manifestEnd) }
                ?: listOf(0L to 4096L) // first pass guess; converges below
            val nonManifest = prev?.nonManifestBytes()
                ?: ByteArray(0) // placeholder; converges below
            val digest = C2paTestFixtures.sha256(nonManifest)
            val json = C2paTestFixtures.dataHashJson(exclusions, digest)

            val claim = C2paTestFixtures.buildClaim()
            val cose = C2paTestFixtures.buildCoseSign1(claim, signingCa)
            val store = C2paTestFixtures.buildStore(
                claim, cose, listOf(ca.certificate.encoded),
                extraBoxes = listOf("jumd" to ("c2pa.hash.data" + json).toByteArray(Charsets.UTF_8)),
            )
            val layout = C2paTestFixtures.jpegWithApp11(
                store, maxSegPayload = maxSegPayload, prefixPadBytes = prefixPadBytes
            )
            if (prev != null && prev.bytes.size == layout.bytes.size &&
                prev.manifestStart == layout.manifestStart && prev.manifestEnd == layout.manifestEnd
            ) {
                stable = layout
                break
            }
            prev = layout
        }
        assertNotNull("fixture layout did not converge", stable)
        return SignedJpeg(stable!!, ca)
    }

    private fun verifierFor(ca: C2paTestFixtures.TestCa): C2PAVerifier =
        C2PAVerifier(C2PAManifestChecker(), C2PATrustStore(ca.pem()))

    @Test
    fun verifiesChainCoseSignatureAndDataHashTogether() {
        val fixture = buildSignedJpeg()
        val result = verifierFor(fixture.ca).verifyStream(ByteArrayInputStream(fixture.layout.bytes))

        assertTrue(result.evidence is ProvenanceEvidence.C2PAClaim)
        val claim = result.evidence as ProvenanceEvidence.C2PAClaim
        assertTrue("manifest should be present", claim.manifestPresent)
        assertTrue(
            "signed requires chain + COSE + data-hash (fail-closed conjunction)",
            claim.signed,
        )
        assertEquals("trainedAlgorithmicMedia", claim.digitalSourceType)
        assertTrue("a signed AI-source manifest confirms AI involvement", claim.confirmsAiInvolvement)
        assertTrue(claim.ingredientCount >= 1)
        assertNotNull(result.issuer)
        assertTrue(
            "notes must name the conjunction",
            result.validationNotes?.contains("fail-closed conjunction") == true,
        )
    }

    @Test
    fun aChainAloneIsExplicitlyUnverified() {
        // F2 core: a valid chain, no COSE signature, no data-hash. The
        // pre-audit code called this "verified"; it must be UNVERIFIED.
        val ca = C2paTestFixtures.newCa()
        val claim = C2paTestFixtures.buildClaim()
        // Empty COSE box content -> parse() returns null -> "no COSE found".
        val store = C2paTestFixtures.buildStore(
            claim, ByteArray(0), listOf(ca.certificate.encoded),
        )
        val layout = C2paTestFixtures.jpegWithApp11(store)
        val result = verifierFor(ca).verifyStream(ByteArrayInputStream(layout.bytes))

        assertTrue(result.evidence is ProvenanceEvidence.C2PAClaim)
        val c2pa = result.evidence as ProvenanceEvidence.C2PAClaim
        assertTrue(c2pa.manifestPresent)
        assertFalse("chain-only must NOT be signed", c2pa.signed)
        assertFalse(c2pa.confirmsAiInvolvement)
        assertTrue(
            "chain-only must be reported as an explicit unverified state",
            result.validationNotes?.contains("stays UNVERIFIED") == true,
        )
        assertTrue(result.validationNotes?.contains("No COSE_Sign1") == true)
        assertTrue(result.validationNotes?.contains("no data-hash binding") == true)
    }

    @Test
    fun aTransplantedManifestFailsTheDataHashBinding() {
        // The hard-binding attack: take a fully signed file and edit ONE byte
        // inside the covered (non-excluded) region. The chain and the COSE
        // signature still verify — only the data-hash exposes the edit.
        val fixture = buildSignedJpeg()
        val tampered = fixture.layout.bytes.copyOfMutable()
        // A padding byte near the end of the filler, before the APP11 group.
        val editAt = (fixture.layout.manifestStart - 16).toInt()
        tampered[editAt] = (tampered[editAt].toInt() xor 0x5A).toByte()
        val result = verifierFor(fixture.ca).verifyStream(ByteArrayInputStream(tampered))

        val claim = result.evidence as ProvenanceEvidence.C2PAClaim
        assertTrue(claim.manifestPresent)
        assertFalse("a transplanted/edited asset must not verify", claim.signed)
        assertFalse(claim.confirmsAiInvolvement)
        assertTrue(
            "the failure must be attributed to the data-hash binding",
            result.validationNotes?.contains("Data-hash binding failed") == true,
        )
    }

    @Test
    fun aSignatureByAnotherKeyFailsTheCoseVerification() {
        // Chain is genuine, but the signature was made by a DIFFERENT key:
        // the x5chain claims the trusted CA's certificate while a rogue
        // key actually signed. Only the COSE check catches this.
        val fixture = buildSignedJpeg(signer = { legitCa ->
            C2paTestFixtures.TestCa(
                C2paTestFixtures.newCa("Rogue Key").keyPair,
                legitCa.certificate, // x5chain still claims the trusted cert
            )
        })
        val result = verifierFor(fixture.ca).verifyStream(ByteArrayInputStream(fixture.layout.bytes))

        val claim = result.evidence as ProvenanceEvidence.C2PAClaim
        assertFalse("a forged signature must not verify", claim.signed)
        assertTrue(
            "the failure must be attributed to the COSE signature",
            result.validationNotes?.contains("COSE_Sign1 signature invalid") == true,
        )
    }

    @Test
    fun anUntrustedRootFailsTheChain() {
        // Signed by a rogue CA end to end: its own cert in the x5chain.
        val fixture = buildSignedJpeg(signer = { C2paTestFixtures.newCa("Rogue Authority") })
        // The verifier trusts the FIRST CA only (fixture.ca); the signer is rogue.
        val result = verifierFor(fixture.ca).verifyStream(ByteArrayInputStream(fixture.layout.bytes))

        val claim = result.evidence as ProvenanceEvidence.C2PAClaim
        assertTrue(claim.manifestPresent)
        assertFalse("a rogue root must not verify", claim.signed)
        assertTrue(
            "the failure must be attributed to the untrusted chain",
            result.validationNotes?.contains("does not chain to a production C2PA trust anchor") == true,
        )
    }

    @Test
    fun anEmptyTrustStoreFailsClosed() {
        val fixture = buildSignedJpeg()
        // Default store: no anchors loaded (the DI path on a device with a
        // missing asset, or any empty list) — nothing may verify as signed.
        val verifier = C2PAVerifier(C2PAManifestChecker())
        val result = verifier.verifyStream(ByteArrayInputStream(fixture.layout.bytes))

        val claim = result.evidence as ProvenanceEvidence.C2PAClaim
        assertTrue(claim.manifestPresent)
        assertFalse("empty trust list must fail closed", claim.signed)
        assertTrue(
            result.validationNotes?.contains("trust list not loaded") == true,
        )
    }

    @Test
    fun theProductionAssetTrustListParsesIntoAnchors() {
        // The shipped asset list is the OFFICIAL C2PA trust list; parsing it
        // must yield a non-empty anchor set (Adobe, Google, DigiCert, ...).
        val pem = java.io.File(
            /* src/test/resources mirror of assets */
            "src/main/assets/${C2PATrustStore.TRUST_LIST_ASSET}"
        ).let { file ->
            // Tests run from android/app; fall back to the repo-relative path.
            val f = if (file.exists()) file
            else java.io.File("android/app/src/main/assets/${C2PATrustStore.TRUST_LIST_ASSET}")
            assertTrue("trust list asset must exist at ${f.path}", f.exists())
            f.readText()
        }
        val store = C2PATrustStore(pem)
        assertFalse("the production list must load anchors", store.isEmpty())
        assertTrue("expected dozens of anchors, got a suspiciously small set", store.size() >= 20)
    }

    @Test
    fun verifierIgnoresPlainJpeg() {
        val plainJpeg = byteArrayOf(
            0xFF.toByte(), 0xD8.toByte(),
            0xFF.toByte(), 0xE0.toByte(), 0x00, 0x10,
            0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
            0xFF.toByte(), 0xD9.toByte(),
        )
        val ca = C2paTestFixtures.newCa()
        val result = verifierFor(ca).verifyStream(ByteArrayInputStream(plainJpeg))
        assertEquals(ProvenanceEvidence.Absent, result.evidence)
    }

    @Test
    fun audioSealBinomialTailIsCorrect() {
        assertEquals(1.0, AudioSealDetector.binomialTwoTailed(16, 32), 1e-9)
        assertTrue(AudioSealDetector.binomialTwoTailed(32, 32) < 1e-9)
        assertTrue(
            AudioSealDetector.binomialTwoTailed(24, 32) < AudioSealDetector.binomialTwoTailed(16, 32)
        )
        assertTrue(
            AudioSealDetector.binomialTwoTailed(28, 32) < AudioSealDetector.binomialTwoTailed(24, 32)
        )
        assertEquals(1.0, AudioSealDetector.binomialTwoTailed(0, 0), 1e-9)
    }

    @Test
    fun anOverbroadExclusionRangeOutsideManifestHullIsRejected() {
        // IACR ePrint 2026/804 exclusion-range attack: the manifest is validly
        // signed by a trusted CA and its c2pa.hash.data digest matches the
        // excluded byte subset, but the exclusion range covers the entire file
        // [0, fileSize) outside the JUMBF manifest container hull.
        val ca = C2paTestFixtures.newCa()
        val emptyDigest = C2paTestFixtures.sha256(ByteArray(0))
        var prev: C2paTestFixtures.JpegLayout? = null
        for (pass in 0 until 6) {
            val totalLen = prev?.bytes?.size?.toLong() ?: 4096L
            val json = C2paTestFixtures.dataHashJson(listOf(0L to totalLen), emptyDigest)
            val claim = C2paTestFixtures.buildClaim()
            val cose = C2paTestFixtures.buildCoseSign1(claim, ca)
            val store = C2paTestFixtures.buildStore(
                claim, cose, listOf(ca.certificate.encoded),
                extraBoxes = listOf("jumd" to ("c2pa.hash.data" + json).toByteArray(Charsets.UTF_8)),
            )
            prev = C2paTestFixtures.jpegWithApp11(store, maxSegPayload = 64, prefixPadBytes = 2048)
        }
        val result = verifierFor(ca).verifyStream(ByteArrayInputStream(prev!!.bytes))
        val c2pa = result.evidence as ProvenanceEvidence.C2PAClaim
        assertTrue(c2pa.manifestPresent)
        assertFalse("exclusion range outside manifest hull must be rejected", c2pa.signed)
        assertTrue(result.validationNotes?.contains("Data-hash binding failed") == true)
    }

    @Test
    fun cborDecodesOneByteLengthsBetween128And255WithoutSignExtension() {
        // Regression test for CBOR additional info == 24 with lengths 128..255
        // (including 255 = 0xFF, which previously sign-extended to -1L).
        for (len in listOf(128, 200, 255)) {
            val payload = ByteArray(len) { (it and 0xFF).toByte() }
            val encoded = com.onndetect.detection.provenance.Cbor.encodeByteString(payload)
            val decoded = com.onndetect.detection.provenance.Cbor.decode(encoded)
            assertTrue("expected BStr for length $len", decoded is com.onndetect.detection.provenance.Cbor.Value.BStr)
            val bstr = decoded as com.onndetect.detection.provenance.Cbor.Value.BStr
            assertEquals(len, bstr.bytes.size)
            assertTrue(bstr.bytes.contentEquals(payload))
        }
    }

    private fun ByteArray.copyOfMutable(): ByteArray {
        val out = ByteArray(size)
        System.arraycopy(this, 0, out, 0, size)
        return out
    }
}
