package com.onndetect.data.local

import com.onndetect.detection.audio.VoicedDurationGate
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import javax.crypto.AEADBadTagException

/**
 * F3: the AES-256-GCM voiceprint vault. Invariants tested on the JVM with
 * the in-memory double; the AndroidKeyStore implementation is exercised on
 * device by the instrumentation suite.
 */
class VoiceVaultCryptoTest {

    private val crypto: VoiceVaultCrypto = InMemoryVoiceVaultCrypto()

    @Test
    fun `seal-unseal round trip preserves the sealed bytes`() {
        val aad = vaultAad("id-2", "Dad")
        val blob = crypto.seal("0.100000,-0.200000,0.300000".toByteArray(), aad)
        assertEquals(12, blob.iv.size)
        val decoded = crypto.unseal(blob, aad)
        assertEquals("0.100000,-0.200000,0.300000", decoded.toString(Charsets.US_ASCII))
    }

    @Test
    fun `a swapped AAD fails authentication instead of decrypting`() {
        // The transplant attack: ciphertext sealed for Mom's row is replayed
        // against Dad's row identity. GCM must reject it loudly.
        val momBlob = crypto.seal("0.5,0.5,0.5".toByteArray(), vaultAad("mom-row", "Mom"))
        var threw = false
        try {
            crypto.unseal(momBlob, vaultAad("dad-row", "Dad"))
        } catch (expected: AEADBadTagException) {
            threw = true
        }
        assertTrue("AAD swap must throw AEADBadTagException", threw)
    }

    @Test
    fun `a tampered ciphertext fails authentication`() {
        val aad = vaultAad("id-3", "Sister")
        val blob = crypto.seal("0.123456".toByteArray(), aad)
        val tampered = SealedBlob(
            blob.iv,
            blob.ciphertext.clone().also { it[it.size / 2] = (it[it.size / 2].toInt() xor 0x40).toByte() }
        )
        var threw = false
        try {
            crypto.unseal(tampered, aad)
        } catch (expected: AEADBadTagException) {
            threw = true
        }
        assertTrue("ciphertext edit must throw AEADBadTagException", threw)
    }

    @Test
    fun `every seal produces a fresh IV`() {
        val aad = vaultAad("id-4", "Brother")
        val a = crypto.seal("same".toByteArray(), aad)
        val b = crypto.seal("same".toByteArray(), aad)
        assertFalse("IVs must not repeat", a.iv.contentEquals(b.iv))
        assertFalse("identical plaintexts must not leak equality", a.ciphertext.contentEquals(b.ciphertext))
    }

    @Test
    fun `vaultAad binds id and contact name deterministically`() {
        // Byte arrays: content equality, never identity.
        assertTrue(vaultAad("x", "Mom").contentEquals(vaultAad("x", "Mom")))
        assertFalse(vaultAad("x", "Mom").contentEquals(vaultAad("x", "Dad")))
        assertFalse(vaultAad("x", "Mom").contentEquals(vaultAad("y", "Mom")))
        assertTrue(vaultAad("x", "Mom").toString(Charsets.UTF_8).startsWith("onn_voice_vault|v1|"))
    }

    // ── F4: the voiced-duration enrollment gate ──────────────────────────

    @Test
    fun `voiced gate counts only energetic low-ZCR frames`() {
        // 10 s of a 220 Hz voiced tone at amplitude 0.3: all frames voiced.
        val pcm = FloatArray(160_000) { i ->
            (0.3 * kotlin.math.sin(2.0 * Math.PI * 220.0 * i / 16_000.0)).toFloat()
        }
        val ms = VoicedDurationGate.voicedMs(pcm)
        assertTrue("expected near-10s voiced, got ${ms}ms", ms in 9_000L..10_000L)
        assertTrue(VoicedDurationGate.passes(pcm))
    }

    @Test
    fun `a short voiced clip fails the 8 second requirement`() {
        val pcm = FloatArray(32_000) { i -> // 2 s
            (0.3 * kotlin.math.sin(2.0 * Math.PI * 220.0 * i / 16_000.0)).toFloat()
        }
        val ms = VoicedDurationGate.voicedMs(pcm)
        assertTrue("expected at most 2s voiced, got ${ms}ms", ms <= 2_000L)
        assertFalse("2 s of speech must not enroll", VoicedDurationGate.passes(pcm))
    }

    @Test
    fun `silence is never voiced`() {
        val pcm = FloatArray(160_000)
        assertEquals(0L, VoicedDurationGate.voicedMs(pcm))
        assertFalse(VoicedDurationGate.passes(pcm))
    }

    @Test
    fun `loud noise with high ZCR is not voiced speech`() {
        // White-ish noise at high amplitude: energetic but not glottal.
        val rnd = java.util.Random(7)
        val pcm = FloatArray(160_000) { (rnd.nextFloat() * 2 - 1) * 0.5f }
        val ms = VoicedDurationGate.voicedMs(pcm)
        // A few frames may straddle the ZCR edge, but the bulk must be rejected.
        assertTrue("high-ZCR noise must not count as voiced (got ${ms}ms)", ms < 1_500L)
    }

    // ── SealedBlob Value Equality & HashCode Contract Tests ──────────────

    @Test
    fun `SealedBlob value equality holds for distinct instances with identical content`() {
        val iv1 = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        val iv2 = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        val ct1 = byteArrayOf(10, 20, 30, 40, 50, 60, 70, 80)
        val ct2 = byteArrayOf(10, 20, 30, 40, 50, 60, 70, 80)

        val blob1 = SealedBlob(iv1, ct1)
        val blob2 = SealedBlob(iv2, ct2)

        // Verify instances are distinct heap allocations
        assertTrue("References must be distinct", blob1 !== blob2)
        assertTrue("IV references must be distinct", blob1.iv !== blob2.iv)
        assertTrue("Ciphertext references must be distinct", blob1.ciphertext !== blob2.ciphertext)

        // Value equality assertions
        assertEquals("blob1 must equal blob2", blob1, blob2)
        assertEquals("blob2 must equal blob1 (symmetry)", blob2, blob1)
        assertTrue("blob1 == blob2 operator", blob1 == blob2)
        assertTrue("blob1.equals(blob2)", blob1.equals(blob2))
    }

    @Test
    fun `SealedBlob hashCode satisfies the general contract for equal instances`() {
        val iv1 = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        val iv2 = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        val ct1 = byteArrayOf(10, 20, 30, 40, 50)
        val ct2 = byteArrayOf(10, 20, 30, 40, 50)

        val blob1 = SealedBlob(iv1, ct1)
        val blob2 = SealedBlob(iv2, ct2)

        // HashCode contract: equal objects must have equal hashCodes
        assertEquals("Hash codes must match for equal blobs", blob1.hashCode(), blob2.hashCode())

        // Consistency across invocations
        assertEquals("HashCode must be deterministic", blob1.hashCode(), blob1.hashCode())

        // Hash-based collection functionality
        val set = hashSetOf(blob1)
        assertTrue("HashSet must contain equal blob instance", set.contains(blob2))

        val map = hashMapOf(blob1 to "verified")
        assertEquals("HashMap must retrieve value by equal blob key", "verified", map[blob2])
    }

    @Test
    fun `SealedBlob inequality detects differences in iv, ciphertext, or length`() {
        val baseIv = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        val baseCt = byteArrayOf(10, 20, 30, 40, 50)
        val baseBlob = SealedBlob(baseIv, baseCt)

        // Altered IV byte
        val diffIv = SealedBlob(baseIv.clone().also { it[0] = (it[0] + 1).toByte() }, baseCt)
        assertNotEquals("Different IV must not equal", baseBlob, diffIv)
        assertFalse(baseBlob == diffIv)

        // Altered Ciphertext byte
        val diffCt = SealedBlob(baseIv, baseCt.clone().also { it[0] = (it[0] + 1).toByte() })
        assertNotEquals("Different Ciphertext must not equal", baseBlob, diffCt)
        assertFalse(baseBlob == diffCt)

        // Truncated arrays
        val shortIv = SealedBlob(byteArrayOf(1, 2, 3), baseCt)
        assertNotEquals("Truncated IV must not equal", baseBlob, shortIv)

        val shortCt = SealedBlob(baseIv, byteArrayOf(10, 20))
        assertNotEquals("Truncated Ciphertext must not equal", baseBlob, shortCt)
    }

    @Test
    fun `SealedBlob copy preserves equality and detects parameter modifications`() {
        val blob = SealedBlob(
            iv = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
            ciphertext = byteArrayOf(100, 101, 102),
        )
        val copied = blob.copy()

        assertTrue("Copy creates a distinct instance", blob !== copied)
        assertEquals("Copy must equal original", blob, copied)
        assertEquals("Copy hashCode must equal original", blob.hashCode(), copied.hashCode())

        val alteredCopy = blob.copy(iv = byteArrayOf(9, 9, 9))
        assertNotEquals("Mutated copy must not equal original", blob, alteredCopy)
    }

    @Test
    fun `SealedBlob equals handles null, reflexivity, and foreign types safely`() {
        val blob = SealedBlob(
            iv = byteArrayOf(1, 2, 3, 4),
            ciphertext = byteArrayOf(5, 6, 7),
        )

        // Reflexivity
        assertEquals("Blob must equal itself", blob, blob)
        assertTrue(blob == blob)

        // Null check
        assertFalse("Blob must not equal null", blob.equals(null))

        // Type check
        assertFalse("Blob must not equal String", blob.equals("not a blob"))
        assertFalse("Blob must not equal Any()", blob.equals(Any()))
    }

    @Test
    fun `SealedBlob handles empty byte arrays correctly`() {
        val empty1 = SealedBlob(byteArrayOf(), byteArrayOf())
        val empty2 = SealedBlob(byteArrayOf(), byteArrayOf())

        assertEquals("Empty blobs must equal", empty1, empty2)
        assertEquals("Empty blobs must have identical hashCodes", empty1.hashCode(), empty2.hashCode())
    }
}
