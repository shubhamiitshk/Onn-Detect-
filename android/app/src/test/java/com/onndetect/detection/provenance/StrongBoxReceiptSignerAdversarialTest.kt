package com.onndetect.detection.provenance

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.security.KeyPairGenerator
import java.security.spec.ECGenParameterSpec
import java.util.Base64
import java.util.concurrent.Callable
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Adversarial stress and challenge test suite for StrongBoxReceiptSigner and ReceiptVerifier.
 *
 * Challenges covered:
 * 1. Extreme thread contention (32+ and 64 concurrent threads racing to sign and query securityTier).
 * 2. Key stability: ensures no key overwrite or race-induced key regeneration occurs under high concurrency.
 * 3. Signature tamper resistance: bit flips, byte mutations, truncated signatures, and corrupt ASN.1 DER.
 * 4. Payload tampering: subtle modifications (whitespace, case, float precision, verdict swap).
 * 5. Corrupt / invalid Base64 and public keys.
 * 6. Non-finite values and invalid media digests in canonical payload generator.
 */
class StrongBoxReceiptSignerAdversarialTest {

    private fun samplePayload(index: Int = 0): String {
        val data = CanonicalReceiptPayload.ReceiptData(
            timestampEpochMs = 1729612800000L + index * 500L,
            challengeNonce = "adv-challenge-nonce-$index",
            mediaSha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            modelRelease = "onn-prod-v10",
            modelHashes = mapOf(
                "srm_forensic_cnn_int8.tflite" to "08397e56b53e506ef89b290f114f167babb2ca530a52320ec770b519bf307ca1",
                "siglip_probe_int8.tflite" to "1b8c4f3de7822196c3c0ebf59f683d86f24d5fa2476fb1d9312a1e14b1519c7b"
            ),
            verdict = if (index % 2 == 0) "AUTHENTIC" else "LIKELY_AI",
            overallScore = ((index % 100) / 100.0f),
            confidence = 0.9500f,
            signals = mapOf(
                "srm_score" to 0.4200f,
                "semantic_probe_score" to 0.7800f
            ),
            deviceSecurityTier = "SOFTWARE_TEST"
        )
        return CanonicalReceiptPayload.serialize(data)
    }

    // ── Challenge 1: Extreme Concurrency (64 threads with CyclicBarrier) ─

    @Test
    fun `test extreme concurrency with 64 racing threads on uninitialized signer`() {
        val signer = StrongBoxReceiptSigner()
        val threadCount = 64
        val executor = Executors.newFixedThreadPool(threadCount)
        val barrier = CyclicBarrier(threadCount)
        val receipts = ConcurrentLinkedQueue<ForensicReceipt>()
        val exceptions = ConcurrentLinkedQueue<Throwable>()
        val tiersSeen = ConcurrentHashMap.newKeySet<String>()

        val tasks = (0 until threadCount).map { i ->
            Callable {
                try {
                    // Force all threads to align at the exact same execution instant
                    barrier.await(10, TimeUnit.SECONDS)
                    // Interleave securityTier and sign calls
                    if (i % 3 == 0) {
                        tiersSeen.add(signer.securityTier)
                    }
                    val receipt = signer.sign(samplePayload(i))
                    receipts.add(receipt)
                    tiersSeen.add(signer.securityTier)
                } catch (t: Throwable) {
                    exceptions.add(t)
                }
            }
        }

        val futures = tasks.map { executor.submit(it) }
        for (f in futures) {
            f.get(15, TimeUnit.SECONDS)
        }
        executor.shutdown()

        assertTrue("No exceptions allowed during 64-thread concurrent execution", exceptions.isEmpty())
        assertEquals("All 64 threads must produce receipts", threadCount, receipts.size)
        assertEquals("Security tier must consistently remain SOFTWARE_TEST", setOf("SOFTWARE_TEST"), tiersSeen)

        // Key stability: ALL 64 receipts MUST have identical public key PEM
        val distinctKeys = receipts.map { it.publicKeyPem }.toSet()
        assertEquals(
            "Exactly ONE key pair must be generated and shared across all 64 concurrent threads (no key rotation race)",
            1,
            distinctKeys.size
        )

        // Cryptographic verification: every signature must be valid for its respective payload
        for (receipt in receipts) {
            assertTrue(
                "Every receipt signature must verify with the shared public key",
                ReceiptVerifier.verifySignature(receipt)
            )
        }
    }

    // ── Challenge 2: Multi-instance Parallel Stress ───────────────────────

    @Test
    fun `test multi-instance independent key generation under concurrency`() {
        val instanceCount = 16
        val executor = Executors.newFixedThreadPool(instanceCount)
        val latch = CountDownLatch(instanceCount)
        val distinctInstanceKeys = ConcurrentHashMap.newKeySet<String>()
        val errorCounter = AtomicInteger(0)

        for (i in 0 until instanceCount) {
            executor.submit {
                try {
                    val independentSigner = StrongBoxReceiptSigner()
                    val receipt = independentSigner.sign(samplePayload(i))
                    assertTrue(ReceiptVerifier.verifySignature(receipt))
                    distinctInstanceKeys.add(receipt.publicKeyPem)
                } catch (t: Throwable) {
                    errorCounter.incrementAndGet()
                } finally {
                    latch.countDown()
                }
            }
        }

        assertTrue("Instances must complete within 10s", latch.await(10, TimeUnit.SECONDS))
        executor.shutdown()

        assertEquals("Zero errors during multi-instance execution", 0, errorCounter.get())
        assertEquals(
            "Each independent StrongBoxReceiptSigner instance must possess a distinct keypair",
            instanceCount,
            distinctInstanceKeys.size
        )
    }

    // ── Challenge 3: Signature Bit-Flip and Mutation Tampering ────────────

    @Test
    fun `adversarially mutate signature bytes across all positions`() {
        val signer = StrongBoxReceiptSigner()
        val payload = samplePayload(1)
        val receipt = signer.sign(payload)

        val originalSigBytes = Base64.getDecoder().decode(receipt.signatureBase64)
        assertTrue("Signature bytes must not be empty", originalSigBytes.isNotEmpty())

        // Mutate single bits/bytes at various positions: first, quarter, mid, three-quarter, last
        val testPositions = listOf(0, originalSigBytes.size / 4, originalSigBytes.size / 2, (originalSigBytes.size * 3) / 4, originalSigBytes.size - 1)
        for (pos in testPositions) {
            val mutated = originalSigBytes.clone()
            mutated[pos] = (mutated[pos].toInt() xor 0x01).toByte()
            val mutatedB64 = Base64.getEncoder().encodeToString(mutated)

            val verified = ReceiptVerifier.verifySignature(
                canonicalPayloadJson = receipt.canonicalPayloadJson,
                signatureBase64 = mutatedB64,
                publicKeyPem = receipt.publicKeyPem
            )
            assertFalse("Mutated signature byte at position $pos must FAIL verification", verified)
        }
    }

    // ── Challenge 4: Truncated and Corrupt Signature Byte Sequences ───────

    @Test
    fun `adversarially verify truncated and corrupted signature encodings fail safely`() {
        val signer = StrongBoxReceiptSigner()
        val payload = samplePayload(2)
        val receipt = signer.sign(payload)
        val rawSigBytes = Base64.getDecoder().decode(receipt.signatureBase64)

        val corruptSignatures = listOf(
            "", // empty string
            "AAAA", // valid base64 but invalid DER
            "not-valid-base64!@#$%", // invalid base64 chars
            Base64.getEncoder().encodeToString(byteArrayOf(0x30, 0x02, 0x01, 0x00)), // invalid DER sequence
            Base64.getEncoder().encodeToString(rawSigBytes.copyOf(rawSigBytes.size / 2)), // truncated DER
            Base64.getEncoder().encodeToString(rawSigBytes.copyOf(1)), // single byte DER
            Base64.getEncoder().encodeToString(ByteArray(rawSigBytes.size) { 0xFF.toByte() }) // all 0xFF bytes
        )

        for (corruptSig in corruptSignatures) {
            val verified = ReceiptVerifier.verifySignature(
                canonicalPayloadJson = receipt.canonicalPayloadJson,
                signatureBase64 = corruptSig,
                publicKeyPem = receipt.publicKeyPem
            )
            assertFalse("Corrupt signature '$corruptSig' must safely return false without crashing", verified)
        }
    }

    // ── Challenge 5: Cross-Key Forgery Attack ────────────────────────────

    @Test
    fun `signature must fail when verified against a foreign public key`() {
        val signerA = StrongBoxReceiptSigner()
        val signerB = StrongBoxReceiptSigner()

        val payload = samplePayload(3)
        val receiptA = signerA.sign(payload)
        val receiptB = signerB.sign(payload)

        // Attempt to verify A's signature with B's public key
        val crossVerified = ReceiptVerifier.verifySignature(
            canonicalPayloadJson = receiptA.canonicalPayloadJson,
            signatureBase64 = receiptA.signatureBase64,
            publicKeyPem = receiptB.publicKeyPem
        )
        assertFalse("Signature verified against foreign public key must FAIL", crossVerified)
    }

    // ── Challenge 6: Canonical Payload Tampering ──────────────────────────

    @Test
    fun `subtle payload alterations must fail signature verification`() {
        val signer = StrongBoxReceiptSigner()
        val payload = samplePayload(4)
        val receipt = signer.sign(payload)

        val variations = listOf(
            payload + " ", // trailing space
            " " + payload, // leading space
            payload.replace("0.9500", "0.9501"), // minor float drift
            payload.replace("AUTHENTIC", "authentic"), // case alteration
            payload.replace("onn-prod-v10", "onn-prod-v10.1"), // release tag tampering
            payload.replace("0123456789abcdef", "0123456789abcdee") // media hash byte modification
        )

        for (tampered in variations) {
            val verified = ReceiptVerifier.verifySignature(
                canonicalPayloadJson = tampered,
                signatureBase64 = receipt.signatureBase64,
                publicKeyPem = receipt.publicKeyPem
            )
            assertFalse("Tampered payload variation must fail signature verification: $tampered", verified)
        }
    }

    // ── Challenge 7: Non-Finite Values and Malformed Media Digest Rejection

    @Test
    fun `canonical serializer strictly rejects non-finite values and invalid digests`() {
        // Test NaN rejection
        try {
            CanonicalReceiptPayload.serialize(
                CanonicalReceiptPayload.ReceiptData(
                    timestampEpochMs = 1000L,
                    challengeNonce = "nonce",
                    mediaSha256 = "a".repeat(64),
                    modelRelease = "onn-prod-v10",
                    modelHashes = emptyMap(),
                    verdict = "AUTHENTIC",
                    overallScore = Float.NaN,
                    confidence = 0.5f,
                    signals = emptyMap(),
                    deviceSecurityTier = "SOFTWARE_TEST"
                )
            )
            fail("Serialization must throw IllegalArgumentException for NaN overallScore")
        } catch (expected: IllegalArgumentException) {
            assertTrue(expected.message!!.contains("non-finite"))
        }

        // Test Infinity rejection
        try {
            CanonicalReceiptPayload.serialize(
                CanonicalReceiptPayload.ReceiptData(
                    timestampEpochMs = 1000L,
                    challengeNonce = "nonce",
                    mediaSha256 = "a".repeat(64),
                    modelRelease = "onn-prod-v10",
                    modelHashes = emptyMap(),
                    verdict = "AUTHENTIC",
                    overallScore = 0.5f,
                    confidence = Float.POSITIVE_INFINITY,
                    signals = emptyMap(),
                    deviceSecurityTier = "SOFTWARE_TEST"
                )
            )
            fail("Serialization must throw IllegalArgumentException for Infinite confidence")
        } catch (expected: IllegalArgumentException) {
            assertTrue(expected.message!!.contains("non-finite"))
        }

        // Test non-hex mediaSha256 rejection
        try {
            CanonicalReceiptPayload.serialize(
                CanonicalReceiptPayload.ReceiptData(
                    timestampEpochMs = 1000L,
                    challengeNonce = "nonce",
                    mediaSha256 = "g".repeat(64), // 'g' is non-hex
                    modelRelease = "onn-prod-v10",
                    modelHashes = emptyMap(),
                    verdict = "AUTHENTIC",
                    overallScore = 0.5f,
                    confidence = 0.9f,
                    signals = emptyMap(),
                    deviceSecurityTier = "SOFTWARE_TEST"
                )
            )
            fail("Serialization must throw IllegalArgumentException for non-hex mediaSha256")
        } catch (expected: IllegalArgumentException) {
            assertTrue(expected.message!!.contains("Invalid media SHA-256"))
        }

        // Test short mediaSha256 rejection
        try {
            CanonicalReceiptPayload.serialize(
                CanonicalReceiptPayload.ReceiptData(
                    timestampEpochMs = 1000L,
                    challengeNonce = "nonce",
                    mediaSha256 = "abc123",
                    modelRelease = "onn-prod-v10",
                    modelHashes = emptyMap(),
                    verdict = "AUTHENTIC",
                    overallScore = 0.5f,
                    confidence = 0.9f,
                    signals = emptyMap(),
                    deviceSecurityTier = "SOFTWARE_TEST"
                )
            )
            fail("Serialization must throw IllegalArgumentException for short mediaSha256")
        } catch (expected: IllegalArgumentException) {
            assertTrue(expected.message!!.contains("Invalid media SHA-256"))
        }
    }
}
