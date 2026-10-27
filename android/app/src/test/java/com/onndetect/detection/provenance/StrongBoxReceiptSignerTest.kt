package com.onndetect.detection.provenance

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class StrongBoxReceiptSignerTest {

    private fun samplePayload(index: Int = 0): String {
        val data = CanonicalReceiptPayload.ReceiptData(
            timestampEpochMs = 1729612800000L + index * 1000L,
            challengeNonce = "nonce-test-$index",
            mediaSha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            modelRelease = "onn-20261021-v7",
            modelHashes = mapOf("model1" to "hash1"),
            verdict = "AUTHENTIC",
            overallScore = 0.1200f,
            confidence = 0.8500f,
            signals = mapOf("srm_score" to 0.2000f),
            deviceSecurityTier = "SOFTWARE_TEST"
        )
        return CanonicalReceiptPayload.serialize(data)
    }

    @Test
    fun testSingleThreadSigningRoundtrip() {
        val signer = StrongBoxReceiptSigner()
        assertEquals("SOFTWARE_TEST", signer.securityTier)

        val payload = samplePayload(1)
        val receipt = signer.sign(payload)

        assertNotNull(receipt)
        assertEquals("SOFTWARE_TEST", receipt.securityTier)
        assertTrue(receipt.certificateChainBase64.isEmpty())
        assertTrue(receipt.publicKeyPem.contains("BEGIN PUBLIC KEY"))
        assertTrue(receipt.signatureBase64.isNotBlank())

        val verified = ReceiptVerifier.verifySignature(receipt)
        assertTrue("Signature roundtrip must succeed for software-signed receipt", verified)

        // Tamper test
        val tamperedPayload = payload.replace("\"AUTHENTIC\"", "\"SYNTHETIC\"")
        val tamperedVerified = ReceiptVerifier.verifySignature(
            canonicalPayloadJson = tamperedPayload,
            signatureBase64 = receipt.signatureBase64,
            publicKeyPem = receipt.publicKeyPem
        )
        assertFalse("Tampered payload must fail signature verification", tamperedVerified)
    }

    @Test
    fun testConcurrentSigningKeyStabilityAndThreadSafety() {
        val signer = StrongBoxReceiptSigner()
        val threadCount = 16
        val executor = Executors.newFixedThreadPool(threadCount)
        val startLatch = CountDownLatch(1)
        val endLatch = CountDownLatch(threadCount)

        val receipts = ConcurrentLinkedQueue<ForensicReceipt>()
        val exceptions = ConcurrentLinkedQueue<Throwable>()

        for (i in 0 until threadCount) {
            val payload = samplePayload(i)
            executor.submit {
                try {
                    // Block until all threads are ready, then execute concurrently
                    startLatch.await()
                    val receipt = signer.sign(payload)
                    receipts.add(receipt)
                } catch (t: Throwable) {
                    exceptions.add(t)
                } finally {
                    endLatch.countDown()
                }
            }
        }

        // Release all threads simultaneously to maximize contention
        startLatch.countDown()

        val completed = endLatch.await(10, TimeUnit.SECONDS)
        executor.shutdown()
        assertTrue("All concurrent signing threads must finish within timeout", completed)
        assertTrue("No exceptions should be thrown under concurrency, but got: ${exceptions.map { it.message }}", exceptions.isEmpty())
        assertEquals(threadCount, receipts.size)

        // All receipts must share the EXACT SAME public key (key identity stability, no race-condition overwrite)
        val distinctPublicKeys = receipts.map { it.publicKeyPem }.toSet()
        assertEquals(
            "Concurrent signing must reuse the same keypair and not overwrite/rotate keys across threads",
            1,
            distinctPublicKeys.size
        )

        // Every receipt's signature must individually verify against its own payload and the shared public key
        for (receipt in receipts) {
            assertEquals("SOFTWARE_TEST", receipt.securityTier)
            val isValid = ReceiptVerifier.verifySignature(receipt)
            assertTrue("Every concurrent receipt must carry a valid cryptographic signature", isValid)
        }
    }

    @Test
    fun testSecurityTierConsistentBeforeAndAfterSign() {
        val signer = StrongBoxReceiptSigner()
        val tierBefore = signer.securityTier
        assertEquals("SOFTWARE_TEST", tierBefore)

        val payload = samplePayload(42)
        val receipt = signer.sign(payload)
        val tierAfter = signer.securityTier

        assertEquals("SOFTWARE_TEST", receipt.securityTier)
        assertEquals("SOFTWARE_TEST", tierAfter)
    }
}
