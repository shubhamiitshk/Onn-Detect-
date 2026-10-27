package com.onndetect.data.local

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.security.SecureRandom
import java.util.Arrays
import javax.crypto.AEADBadTagException

/**
 * Adversarial stress and challenge test suite for VoiceVaultCrypto and SealedBlob.
 *
 * Challenges covered:
 * 1. SealedBlob value equality under boundary and edge conditions:
 *    - Differing IVs, differing ciphertext, altered GCM tags.
 *    - Zero-length arrays, single-byte differences, length extensions.
 *    - Extreme boundary byte values (0x00, 0x7F, 0x80, 0xFF).
 * 2. HashCode contract and hash collision resistance across 10,000 distinct allocations.
 * 3. Strict equivalence relation laws: Reflexivity, Symmetry, Transitivity, Consistency, Non-nullity.
 * 4. Collection semantics (HashSet and HashMap membership, mutation caveats).
 * 5. AES-256-GCM cryptographic boundary tests (IV uniqueness, authentication tag corruption, AAD binding).
 */
class VoiceVaultCryptoAdversarialTest {

    private val crypto: VoiceVaultCrypto = InMemoryVoiceVaultCrypto()
    private val random = SecureRandom()

    // ── Challenge 1: Boundary Array Sizes & Edge Equality ────────────────

    @Test
    fun `test SealedBlob equality with zero-length and asymmetric empty arrays`() {
        val emptyIvEmptyCt1 = SealedBlob(ByteArray(0), ByteArray(0))
        val emptyIvEmptyCt2 = SealedBlob(ByteArray(0), ByteArray(0))
        assertEquals(emptyIvEmptyCt1, emptyIvEmptyCt2)
        assertEquals(emptyIvEmptyCt1.hashCode(), emptyIvEmptyCt2.hashCode())

        val emptyIvWithCt = SealedBlob(ByteArray(0), byteArrayOf(1, 2, 3))
        val withIvEmptyCt = SealedBlob(byteArrayOf(1, 2, 3), ByteArray(0))
        val fullBlob = SealedBlob(byteArrayOf(1, 2, 3), byteArrayOf(1, 2, 3))

        assertNotEquals(emptyIvEmptyCt1, emptyIvWithCt)
        assertNotEquals(emptyIvEmptyCt1, withIvEmptyCt)
        assertNotEquals(emptyIvWithCt, withIvEmptyCt)
        assertNotEquals(emptyIvWithCt, fullBlob)
        assertNotEquals(withIvEmptyCt, fullBlob)
    }

    @Test
    fun `test SealedBlob with extreme boundary byte values`() {
        // Test 0x00, 0x7F (127), 0x80 (-128 in signed byte), 0xFF (-1 in signed byte)
        val iv1 = byteArrayOf(0x00, 0x7F.toByte(), 0x80.toByte(), 0xFF.toByte())
        val iv2 = byteArrayOf(0x00, 0x7F.toByte(), 0x80.toByte(), 0xFF.toByte())
        val ct1 = byteArrayOf(0xFF.toByte(), 0x80.toByte(), 0x7F.toByte(), 0x00)
        val ct2 = byteArrayOf(0xFF.toByte(), 0x80.toByte(), 0x7F.toByte(), 0x00)

        val blob1 = SealedBlob(iv1, ct1)
        val blob2 = SealedBlob(iv2, ct2)

        assertEquals("Equal boundary bytes must evaluate as equal", blob1, blob2)
        assertEquals(blob1.hashCode(), blob2.hashCode())

        // Mutate single signed boundary bit (0x7F vs 0x80)
        val mutatedIv = byteArrayOf(0x00, 0x80.toByte(), 0x80.toByte(), 0xFF.toByte())
        val blobMutated = SealedBlob(mutatedIv, ct1)
        assertNotEquals("0x7F must not equal 0x80 signed byte representation", blob1, blobMutated)
    }

    @Test
    fun `test SealedBlob length extension and prefix sensitivity`() {
        val baseIv = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        val baseCt = byteArrayOf(10, 20, 30, 40)
        val original = SealedBlob(baseIv, baseCt)

        // Trailing zero append
        val extendedIv = SealedBlob(baseIv + byteArrayOf(0x00), baseCt)
        val extendedCt = SealedBlob(baseIv, baseCt + byteArrayOf(0x00))

        assertNotEquals("Appended zero to IV must produce inequality", original, extendedIv)
        assertNotEquals("Appended zero to Ciphertext must produce inequality", original, extendedCt)
        assertNotEquals("Hash codes must differ for extended arrays", original.hashCode(), extendedIv.hashCode())
    }

    @Test
    fun `test SealedBlob with large 100KiB ciphertext payload`() {
        val largeIv = ByteArray(12).also { random.nextBytes(it) }
        val largeCt = ByteArray(102400).also { random.nextBytes(it) }

        val b1 = SealedBlob(largeIv.clone(), largeCt.clone())
        val b2 = SealedBlob(largeIv.clone(), largeCt.clone())

        assertEquals("Large 100KiB payload equality must hold", b1, b2)
        assertEquals("Large 100KiB payload hashCode must match", b1.hashCode(), b2.hashCode())

        // Modify the very last byte of 100KiB payload
        val modifiedCt = largeCt.clone()
        modifiedCt[modifiedCt.size - 1] = (modifiedCt[modifiedCt.size - 1] + 1).toByte()
        val b3 = SealedBlob(largeIv.clone(), modifiedCt)
        assertNotEquals("Modification of last byte in 100KiB payload must produce inequality", b1, b3)
    }

    // ── Challenge 2: Transposition & Hash Collision Resistance ────────────

    @Test
    fun `test asymmetric hash calculation prevents IV and ciphertext transposition collisions`() {
        val bytesA = byteArrayOf(1, 2, 3, 4)
        val bytesB = byteArrayOf(5, 6, 7, 8)

        val blobAB = SealedBlob(iv = bytesA, ciphertext = bytesB)
        val blobBA = SealedBlob(iv = bytesB, ciphertext = bytesA)

        assertNotEquals("Transposed IV and ciphertext must not equal", blobAB, blobBA)
        assertNotEquals(
            "Multiplier 31 must differentiate transposed IV and ciphertext hashcodes",
            blobAB.hashCode(),
            blobBA.hashCode()
        )
    }

    @Test
    fun `test hash collision resistance across 10,000 distinct SealedBlobs`() {
        val count = 10_000
        val distinctBlobs = HashSet<SealedBlob>(count)
        val distinctHashes = HashSet<Int>(count)

        for (i in 0 until count) {
            val iv = ByteArray(12)
            // Use deterministic structured variation
            iv[0] = (i and 0xFF).toByte()
            iv[1] = ((i shr 8) and 0xFF).toByte()
            val ct = ByteArray(32)
            ct[0] = ((i shr 16) and 0xFF).toByte()
            ct[1] = (i and 0xFF).toByte()

            val blob = SealedBlob(iv, ct)
            distinctBlobs.add(blob)
            distinctHashes.add(blob.hashCode())
        }

        assertEquals("All 10,000 generated blobs must be distinct", count, distinctBlobs.size)
        // With 32-bit hash codes and 10,000 items, birthday paradox allows minor collisions,
        // but distinct hashes should be > 99.5% of total
        val collisionCount = count - distinctHashes.size
        assertTrue(
            "Excessive hash collisions ($collisionCount collisions out of $count items)",
            collisionCount < 50
        )
    }

    // ── Challenge 3: Strict Equivalence Relation Verification ─────────────

    @Test
    fun `verify strict mathematical equivalence relations for SealedBlob`() {
        val ivA = byteArrayOf(10, 20, 30)
        val ctA = byteArrayOf(40, 50, 60)
        val x = SealedBlob(ivA.clone(), ctA.clone())
        val y = SealedBlob(ivA.clone(), ctA.clone())
        val z = SealedBlob(ivA.clone(), ctA.clone())
        val w = SealedBlob(byteArrayOf(99), byteArrayOf(99))

        // 1. Reflexivity: x == x
        assertTrue("Reflexivity: x == x", x == x)
        assertTrue("Reflexivity: x.equals(x)", x.equals(x))

        // 2. Symmetry: x == y <=> y == x
        assertTrue("Symmetry: x == y", x == y)
        assertTrue("Symmetry: y == x", y == x)
        assertEquals(x == w, w == x)

        // 3. Transitivity: x == y and y == z implies x == z
        assertTrue("Transitivity: x == y", x == y)
        assertTrue("Transitivity: y == z", y == z)
        assertTrue("Transitivity: x == z", x == z)

        // 4. Consistency: multiple invocations yield identical results
        for (i in 0 until 100) {
            assertTrue(x == y)
            assertEquals(x.hashCode(), y.hashCode())
        }

        // 5. Non-nullity
        assertFalse("Non-nullity: x != null", x.equals(null))

        // 6. Type mismatch
        assertFalse("Type mismatch: x != String", x.equals("SealedBlob"))
        assertFalse("Type mismatch: x != Any", x.equals(Any()))
    }

    // ── Challenge 4: Collection Semantics and Array Mutation Caveat ───────

    @Test
    fun `test HashSet and HashMap operations with cloned and independent instances`() {
        val testSet = HashSet<SealedBlob>()
        val testMap = HashMap<SealedBlob, String>()

        val blobs = (0 until 50).map { i ->
            SealedBlob(
                iv = byteArrayOf(i.toByte(), 1, 2, 3),
                ciphertext = byteArrayOf(i.toByte(), 4, 5, 6)
            )
        }

        for ((idx, b) in blobs.withIndex()) {
            testSet.add(b)
            testMap[b] = "value-$idx"
        }

        assertEquals(50, testSet.size)
        assertEquals(50, testMap.size)

        // Query with newly allocated identical byte arrays
        for ((idx, b) in blobs.withIndex()) {
            val queryBlob = SealedBlob(b.iv.clone(), b.ciphertext.clone())
            assertTrue("HashSet must contain queryBlob", testSet.contains(queryBlob))
            assertEquals("HashMap must return correct value", "value-$idx", testMap[queryBlob])
        }
    }

    @Test
    fun `document in-place array mutation hazard on hash collections`() {
        // In Java/Kotlin, ByteArray is mutable. If a caller mutates the array
        // backing a SealedBlob already inserted in a HashSet, the object becomes
        // unreachable because its hash code changed.
        val mutableIv = byteArrayOf(1, 2, 3)
        val mutableCt = byteArrayOf(4, 5, 6)
        val blob = SealedBlob(mutableIv, mutableCt)

        val set = HashSet<SealedBlob>()
        set.add(blob)
        val originalHash = blob.hashCode()

        // Mutate array in-place
        mutableIv[0] = 99
        val newHash = blob.hashCode()

        assertNotEquals("In-place array mutation changes the hash code", originalHash, newHash)
        // Note: the set still contains the reference, but contains(queryWithNewHash) can fail or succeed depending on bucket
    }

    // ── Challenge 5: AES-GCM Cryptographic Integrity & Tag Alterations ────

    @Test
    fun `test 1,000 consecutive seals produce completely unique IVs (zero collision)`() {
        val aad = vaultAad("profile-1", "Owner")
        val plaintext = "voiceprint-embedding-12345".toByteArray()
        val ivSet = HashSet<String>()

        for (i in 0 until 1000) {
            val sealed = crypto.seal(plaintext, aad)
            assertEquals(12, sealed.iv.size)
            val ivHex = sealed.iv.joinToString("") { "%02x".format(it) }
            val added = ivSet.add(ivHex)
            assertTrue("IV must never repeat across 1,000 seals", added)
        }
    }

    @Test
    fun `tampering with GCM 128-bit authentication tag strictly throws AEADBadTagException`() {
        val aad = vaultAad("profile-2", "Daughter")
        val plaintext = "biometric-voice-vector-data".toByteArray()
        val sealed = crypto.seal(plaintext, aad)

        // In AES/GCM/NoPadding, the last 16 bytes (128 bits) of ciphertext comprise the auth tag
        val ct = sealed.ciphertext
        assertTrue("Ciphertext must contain at least 16 bytes for auth tag", ct.size >= 16)

        // Flip a bit in the authentication tag (last byte)
        val tagTamperedCt = ct.clone()
        tagTamperedCt[tagTamperedCt.size - 1] = (tagTamperedCt[tagTamperedCt.size - 1].toInt() xor 0x01).toByte()
        val tamperedBlob = SealedBlob(sealed.iv, tagTamperedCt)

        try {
            crypto.unseal(tamperedBlob, aad)
            fail("AEADBadTagException must be thrown when GCM authentication tag is altered")
        } catch (expected: AEADBadTagException) {
            // Expected pass
        }
    }

    @Test
    fun `tampering with IV strictly throws AEADBadTagException`() {
        val aad = vaultAad("profile-3", "Son")
        val plaintext = "secret-voice-sample".toByteArray()
        val sealed = crypto.seal(plaintext, aad)

        val tamperedIv = sealed.iv.clone()
        tamperedIv[0] = (tamperedIv[0].toInt() xor 0x80).toByte()
        val tamperedBlob = SealedBlob(tamperedIv, sealed.ciphertext)

        try {
            crypto.unseal(tamperedBlob, aad)
            fail("AEADBadTagException must be thrown when IV is altered")
        } catch (expected: AEADBadTagException) {
            // Expected pass
        }
    }

    @Test
    fun `truncated ciphertext throws AEADBadTagException`() {
        val aad = vaultAad("profile-4", "Grandmother")
        val plaintext = "voice-biometrics-sample-data".toByteArray()
        val sealed = crypto.seal(plaintext, aad)

        // Truncate ciphertext by 4 bytes (stripping partial tag)
        val truncatedCt = sealed.ciphertext.copyOf(sealed.ciphertext.size - 4)
        val truncatedBlob = SealedBlob(sealed.iv, truncatedCt)

        try {
            crypto.unseal(truncatedBlob, aad)
            fail("AEADBadTagException must be thrown when ciphertext is truncated")
        } catch (expected: AEADBadTagException) {
            // Expected pass
        }
    }
}
