package com.onndetect

import com.onndetect.detection.provenance.AudioSealDetector
import com.onndetect.detection.provenance.C2PAManifestChecker
import com.onndetect.detection.provenance.C2PATrustStore
import com.onndetect.detection.provenance.C2PAVerifier
import com.onndetect.detection.provenance.Cbor
import com.onndetect.detection.provenance.ProvenanceEvidence
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.security.KeyFactory
import java.security.Signature
import java.security.spec.PKCS8EncodedKeySpec
import java.util.Base64

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

    private fun bytesToHex(bytes: ByteArray): String =
        bytes.joinToString("") { "%02x".format(it) }

    @Test
    fun c2paVerifierRejectsManifestWithTrustedCertMissingSignature() {
        // Manifest carries a valid trusted OpenAI root certificate but no signature block.
        // Fail-closed conjunction requires: signed = hashVerified && coseSignatureValid && certChainValid.
        val certPem = C2PATrustStore.OPENAI_ROOT_PEM.trim()
        val payload = ("c2patrainedAlgorithmicMedia\n" + certPem).toByteArray(Charsets.ISO_8859_1)
        val jpegBytes = jpegWithApp11(payload)
        val verifier = C2PAVerifier(C2PAManifestChecker())
        val result = verifier.verifyStream(ByteArrayInputStream(jpegBytes))

        assertTrue(result.evidence is ProvenanceEvidence.C2PAClaim)
        val claim = result.evidence as ProvenanceEvidence.C2PAClaim
        assertTrue("Manifest should be present", claim.manifestPresent)
        assertFalse("Manifest with valid cert but missing signature must NOT be signed (fail-closed)", claim.signed)
        assertFalse("Unsigned manifest must not confirm AI involvement", claim.confirmsAiInvolvement)
        assertTrue(result.validationNotes?.contains("lacks cryptographic claim signature") == true)
    }

    @Test
    fun c2paVerifierRejectsManifestWithoutCertificate() {
        // Unverified manifest missing certificate chain (exclusion-range discipline)
        val payload = "c2patrainedAlgorithmicMedia".toByteArray(Charsets.ISO_8859_1)
        val jpegBytes = jpegWithApp11(payload)
        val verifier = C2PAVerifier(C2PAManifestChecker())
        val result = verifier.verifyStream(ByteArrayInputStream(jpegBytes))

        assertTrue(result.evidence is ProvenanceEvidence.C2PAClaim)
        val claim = result.evidence as ProvenanceEvidence.C2PAClaim
        assertTrue("Manifest should be present", claim.manifestPresent)
        assertFalse("Unverified manifest without certificate must not be signed", claim.signed)
        assertFalse("Unsigned manifest must not confirm AI involvement", claim.confirmsAiInvolvement)
        assertTrue(result.validationNotes?.contains("missing X.509 certificate chain") == true)
    }

    @Test
    fun c2paVerifierRejectsUntrustedCertificate() {
        // Genuine X.509 certificate signed by an untrusted rogue authority
        val untrustedCertPem = """
-----BEGIN CERTIFICATE-----
MIIBmTCCAT+gAwIBAgIUQiayhFRXYMTiOFPg389pqgAaMyIwCgYIKoZIzj0EAwIw
QjELMAkGA1UEBhMCWFgxGzAZBgNVBAoMElVudHJ1c3RlZCBSb2d1ZSBDQTEWMBQG
A1UEAwwNUm9ndWUgUm9vdCBDQTAeFw0yNDAxMDEwMDAwMDBaFw0zNTAxMDEwMDAw
MDBaMEIxCzAJBgNVBAYTAlhYMRswGQYDVQQKDBJVbnRydXN0ZWQgUm9ndWUgQ0Ex
FjAUBgNVBAMMDVJvZ3VlIFJvb3QgQ0EwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNC
AAT0ojISNKrkeLDOqnaEbKpncsdzEkrR0Dl06DJyXFz4PmlNc7d0RHCSho+hxhr4
4+BZds57cdelURQGVz5Kic2moxMwETAPBgNVHRMBAf8EBTADAQH/MAoGCCqGSM49
BAMCA0gAMEUCIA1RJSt5AUzhWN8+DCC0gSVkRmkxCXM6MK8SkYDdJvO9AiEAv/RQ
QBxVum7HATnh3Jw5d65hi8pPudk7TkkxJa3lPdo=
-----END CERTIFICATE-----
""".trimIndent()

        val payload = ("c2patrainedAlgorithmicMedia\n" + untrustedCertPem).toByteArray(Charsets.ISO_8859_1)
        val jpegBytes = jpegWithApp11(payload)
        val verifier = C2PAVerifier(C2PAManifestChecker())
        val result = verifier.verifyStream(ByteArrayInputStream(jpegBytes))

        assertTrue(result.evidence is ProvenanceEvidence.C2PAClaim)
        val claim = result.evidence as ProvenanceEvidence.C2PAClaim
        assertTrue(claim.manifestPresent)
        assertFalse("Manifest with untrusted root must not be signed", claim.signed)
        assertFalse(claim.confirmsAiInvolvement)
        assertTrue(result.validationNotes?.contains("Untrusted certificate authority") == true)
    }

    @Test
    fun c2paVerifierValidatesCryptographicSignatureOverClaim() {
        val trustStore = C2PATrustStore()
        val testCert = trustStore.parseCertificate(TEST_ROOT_PEM.trim().toByteArray(Charsets.UTF_8))!!
        trustStore.addTrustAnchor(testCert)

        // Sign claim using TEST_ROOT_KEY_PKCS8
        val keyDer = Base64.getMimeDecoder().decode(
            TEST_ROOT_KEY_PKCS8
                .replace("-----BEGIN PRIVATE KEY-----", "")
                .replace("-----END PRIVATE KEY-----", "")
                .trim()
        )
        val keySpec = PKCS8EncodedKeySpec(keyDer)
        val privKey = KeyFactory.getInstance("EC").generatePrivate(keySpec)

        val claimText = "c2pa_claim_sha256_hash_test_asset"
        val dsa = Signature.getInstance("SHA256withECDSA")
        dsa.initSign(privKey)
        dsa.update(claimText.toByteArray(Charsets.UTF_8))
        val signatureBytes = dsa.sign()
        val signatureHex = bytesToHex(signatureBytes)

        val certPem = TEST_ROOT_PEM.trim()
        val payload = ("c2patrainedAlgorithmicMedia\n" +
            certPem + "\n" +
            "c2cs:" + claimText + ":" + signatureHex).toByteArray(Charsets.ISO_8859_1)

        val jpegBytes = jpegWithApp11(payload)
        val verifier = C2PAVerifier(C2PAManifestChecker(), trustStore)
        val result = verifier.verifyStream(ByteArrayInputStream(jpegBytes))

        assertTrue(result.evidence is ProvenanceEvidence.C2PAClaim)
        val claim = result.evidence as ProvenanceEvidence.C2PAClaim
        assertTrue("Manifest should be present", claim.manifestPresent)
        assertTrue("Signature over claim should be verified", claim.signed)
        assertEquals("trainedAlgorithmicMedia", claim.digitalSourceType)
        assertTrue(claim.confirmsAiInvolvement)
        assertTrue(result.validationNotes?.contains("Cryptographically verified against trusted C2PA root") == true)
    }

    @Test
    fun c2paVerifierRejectsTamperedCryptographicSignature() {
        val trustStore = C2PATrustStore()
        val testCert = trustStore.parseCertificate(TEST_ROOT_PEM.trim().toByteArray(Charsets.UTF_8))!!
        trustStore.addTrustAnchor(testCert)

        val claimText = "c2pa_claim_sha256_hash_test_asset"
        val tamperedSignatureHex = "3045022100" + "ff".repeat(31) + "0220" + "ee".repeat(32)

        val certPem = TEST_ROOT_PEM.trim()
        val payload = ("c2patrainedAlgorithmicMedia\n" +
            certPem + "\n" +
            "c2cs:" + claimText + ":" + tamperedSignatureHex).toByteArray(Charsets.ISO_8859_1)

        val jpegBytes = jpegWithApp11(payload)
        val verifier = C2PAVerifier(C2PAManifestChecker(), trustStore)
        val result = verifier.verifyStream(ByteArrayInputStream(jpegBytes))

        assertTrue(result.evidence is ProvenanceEvidence.C2PAClaim)
        val claim = result.evidence as ProvenanceEvidence.C2PAClaim
        assertTrue(claim.manifestPresent)
        assertFalse("Tampered signature must fail verification", claim.signed)
        assertFalse(claim.confirmsAiInvolvement)
        assertTrue(result.validationNotes?.contains("C2PA signature mismatch") == true)
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
    fun cborDecodes128To255ByteLengthsCorrectlyWithoutSignExtension() {
        // ByteString with major type 2, 1-byte length info 24 (0x58), length 200 (0xC8)
        val len = 200
        val payload = ByteArray(2 + len)
        payload[0] = 0x58.toByte()
        payload[1] = 0xC8.toByte() // 200 in unsigned byte, -56 in signed byte
        for (i in 0 until len) {
            payload[2 + i] = (i and 0xFF).toByte()
        }

        val decoded = Cbor.decode(payload)
        assertTrue(decoded is Cbor.Value.ByteString)
        val byteStr = decoded as Cbor.Value.ByteString
        assertEquals(200, byteStr.bytes.size)
        assertEquals(0.toByte(), byteStr.bytes[0])
        assertEquals(199.toByte(), byteStr.bytes[199])
    }

    @Test
    fun c2paVerifierEnforcesIacr2026804ExclusionHullInvariant() {
        val verifier = C2PAVerifier(C2PAManifestChecker())
        val dummyFile = ByteArray(1024) { (it % 256).toByte() }
        val manifestStart = 200L
        val manifestEnd = 400L
        val manifestRanges = listOf(Pair(manifestStart, manifestEnd))

        // Legitimate exclusion inside manifest hull (200..400)
        val legitDh = C2PAManifestChecker.DataHash(
            exclusions = listOf(C2PAManifestChecker.Exclusion(210L, 390L))
        )
        val legitDigest = verifier.recomputeDataHash(dummyFile, legitDh, manifestRanges)
        assertNotNull("Legitimate exclusion within container hull must compute hash", legitDigest)

        // Forged exclusion outside manifest hull: excluding entire file 0..1024 (exploit vector)
        val exploitDh = C2PAManifestChecker.DataHash(
            exclusions = listOf(C2PAManifestChecker.Exclusion(0L, 1024L))
        )
        val exploitDigest = verifier.recomputeDataHash(dummyFile, exploitDh, manifestRanges)
        assertNull("Exclusion covering entire asset outside manifest hull must be rejected", exploitDigest)

        // Forged exclusion outside hull bounds (e.g., 0..100)
        val outsideDh = C2PAManifestChecker.DataHash(
            exclusions = listOf(C2PAManifestChecker.Exclusion(0L, 100L))
        )
        val outsideDigest = verifier.recomputeDataHash(dummyFile, outsideDh, manifestRanges)
        assertNull("Exclusion outside JUMBF container hull must fail IACR 2026/804 invariant", outsideDigest)
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

    companion object {
        // Ephemeral EC key and test certificate for ProvenanceIntegrationTest only.
        // Never shipped or compiled into production release artifacts.
        private const val TEST_ROOT_PEM = """
-----BEGIN CERTIFICATE-----
MIIBnzCCAUWgAwIBAgIUWiyNi980f5j+oj+EgQYFGagJJPkwCgYIKoZIzj0EAwIw
RTELMAkGA1UEBhMCVVMxFzAVBgNVBAoMDlRlc3QgQXV0aG9yaXR5MR0wGwYDVQQD
DBRUZXN0IEMyUEEgVHJ1c3QgUm9vdDAeFw0yNDAxMDEwMDAwMDBaFw0zNTAxMDEw
MDAwMDBaMEUxCzAJBgNVBAYTAlVTMRcwFQYDVQQKDA5UZXN0IEF1dGhvcml0eTEd
MBsGA1UEAwwUVGVzdCBDMlBBIFRydXN0IFJvb3QwWTATBgcqhkjOPQIBBggqhkjO
PQMBBwNCAASt6BSIVFQRYt94m18Ce5o+9Mae9VoOtbRiSX1SelzulUX43By/sWPl
UrQTUDWjDrNKjdY7VDeXb6mZ9c02ZoJXoxMwETAPBgNVHRMBAf8EBTADAQH/MAoG
CCqGSM49BAMCA0gAMEUCIDsLPOF0RKFx6sBvdsbc5fcKuVJvRNv3xqSrcJIaKtoa
AiEA6NZcKmMj6mqMLPepnoOUy2/Yg6ajSQk++qDmYkHhsAY=
-----END CERTIFICATE-----
"""

        private const val TEST_ROOT_KEY_PKCS8 = """
-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgAtT/+8yrPVKxmtrz
YY6i5OiwGQ8mNHK9i1XBMy4N1OmhRANCAASt6BSIVFQRYt94m18Ce5o+9Mae9VoO
tbRiSX1SelzulUX43By/sWPlUrQTUDWjDrNKjdY7VDeXb6mZ9c02ZoJX
-----END PRIVATE KEY-----
"""
    }
}
