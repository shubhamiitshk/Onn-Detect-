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
}
