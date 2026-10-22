package com.onndetect.detection.provenance

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.security.KeyPairGenerator
import java.security.Signature
import java.security.spec.ECGenParameterSpec
import java.util.Base64

class CanonicalReceiptPayloadTest {

    @Test
    fun `canonical serializer produces sorted lexicographical keys`() {
        val data = CanonicalReceiptPayload.ReceiptData(
            timestampEpochMs = 1729612800000L,
            challengeNonce = "nonce-test-1234",
            mediaSha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            modelRelease = "onn-20261021-v7",
            modelHashes = mapOf(
                "srm_forensic_cnn" to "e90b7a42",
                "siglip_probe" to "9c11bd18",
                "lcnn_audio" to "2a76f231"
            ),
            verdict = "SYNTHETIC",
            overallScore = 0.8941f,
            confidence = 0.9200f,
            signals = mapOf(
                "srm_score" to 0.6871f,
                "semantic_probe_score" to 0.8910f,
                "c2pa_present" to 0.0f
            ),
            deviceSecurityTier = "STRONGBOX"
        )

        val json = CanonicalReceiptPayload.serialize(data)

        // Verify key ordering
        val expectedPrefix = "{\"challenge_nonce\":\"nonce-test-1234\",\"confidence\":0.9200,\"device_security_tier\":\"STRONGBOX\""
        assertTrue("JSON must start with sorted keys", json.startsWith(expectedPrefix))

        // Verify inner map ordering: lcnn_audio must appear before siglip_probe
        val lcnnIdx = json.indexOf("\"lcnn_audio\"")
        val siglipIdx = json.indexOf("\"siglip_probe\"")
        val srmIdx = json.indexOf("\"srm_forensic_cnn\"")
        assertTrue("model_hashes must be sorted", lcnnIdx in 0 until siglipIdx && siglipIdx < srmIdx)

        // Verify inner signals ordering: c2pa_present must appear before semantic_probe_score
        val c2paIdx = json.indexOf("\"c2pa_present\"")
        val probeIdx = json.indexOf("\"semantic_probe_score\"")
        assertTrue("signals must be sorted", c2paIdx in 0 until probeIdx)
    }

    @Test
    fun `receipt signature verification round-trip succeeds`() {
        val kpg = KeyPairGenerator.getInstance("EC")
        kpg.initialize(ECGenParameterSpec("secp256r1"))
        val keyPair = kpg.generateKeyPair()

        val data = CanonicalReceiptPayload.ReceiptData(
            timestampEpochMs = 1729612800000L,
            challengeNonce = "nonce-live-verify",
            mediaSha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            modelRelease = "onn-20261021-v7",
            modelHashes = mapOf("model1" to "hash1"),
            verdict = "AUTHENTIC",
            overallScore = 0.1200f,
            confidence = 0.8500f,
            signals = mapOf("srm_score" to 0.2000f),
            deviceSecurityTier = "TEE"
        )

        val canonicalJson = CanonicalReceiptPayload.serialize(data)
        val signer = Signature.getInstance("SHA256withECDSA")
        signer.initSign(keyPair.private)
        signer.update(canonicalJson.toByteArray(Charsets.UTF_8))
        val signatureBytes = signer.sign()
        val signatureB64 = Base64.getEncoder().encodeToString(signatureBytes)

        val pubPem = ReceiptVerifier.formatPublicKeyToPem(keyPair.public.encoded)

        val receipt = ForensicReceipt(
            canonicalPayloadJson = canonicalJson,
            signatureBase64 = signatureB64,
            publicKeyPem = pubPem,
            securityTier = "TEE"
        )

        // Verify roundtrip
        val verified = ReceiptVerifier.verify(receipt)
        assertTrue("Valid signature must verify", verified)

        // Tamper test: Altering a single character in the payload fails verification
        val tamperedJson = canonicalJson.replace("\"AUTHENTIC\"", "\"SYNTHETIC\"")
        val tamperedVerified = ReceiptVerifier.verify(
            canonicalPayloadJson = tamperedJson,
            signatureBase64 = signatureB64,
            publicKeyPem = pubPem
        )
        assertFalse("Tampered payload must fail signature verification", tamperedVerified)
    }

    @Test
    fun `computeSha256 produces exact 64-char hex digest`() {
        val bytes = "ONN-DETECT-MEDIA-TEST".toByteArray(Charsets.UTF_8)
        val sha256 = CanonicalReceiptPayload.computeSha256(bytes)
        assertEquals(64, sha256.length)
        assertTrue(sha256.matches(Regex("^[0-9a-f]{64}$")))
    }
}
