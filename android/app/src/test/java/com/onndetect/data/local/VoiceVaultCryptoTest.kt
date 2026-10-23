package com.onndetect.data.local

import org.junit.Assert.*
import org.junit.Test

class VoiceVaultCryptoTest {

    private val crypto = VoiceVaultCrypto()

    @Test
    fun `encrypt and decrypt roundtrip restores exact embedding floats`() {
        val original = FloatArray(160) { i -> (i * 0.0123f) - 0.5f }
        val id = "profile-uuid-1234"
        val contactName = "Alice Mother"

        val enc = crypto.encrypt(original, id, contactName)
        assertNotNull(enc.ciphertext)
        assertNotNull(enc.iv)
        assertEquals(12, enc.iv.size)
        assertTrue(enc.ciphertext.size >= original.size * 4 + 16) // Plaintext bytes + GCM tag

        val decrypted = crypto.decrypt(enc.ciphertext, enc.iv, id, contactName)
        assertEquals(original.size, decrypted.size)
        for (i in original.indices) {
            assertEquals(original[i], decrypted[i], 1e-6f)
        }
    }

    @Test
    fun `decrypt fails closed when contactName AAD is altered`() {
        val original = FloatArray(160) { 0.42f }
        val id = "profile-uuid-1234"
        val contactName = "Alice Mother"

        val enc = crypto.encrypt(original, id, contactName)

        // Attempt decryption with tampered contact name
        val tamperedDecrypted = crypto.decrypt(enc.ciphertext, enc.iv, id, "Attacker Imposter")
        assertEquals(0, tamperedDecrypted.size)
    }

    @Test
    fun `decrypt fails closed when id AAD is altered`() {
        val original = FloatArray(160) { 0.42f }
        val id = "profile-uuid-1234"
        val contactName = "Alice Mother"

        val enc = crypto.encrypt(original, id, contactName)

        // Attempt decryption with altered ID
        val tamperedDecrypted = crypto.decrypt(enc.ciphertext, enc.iv, "profile-uuid-9999", contactName)
        assertEquals(0, tamperedDecrypted.size)
    }

    @Test
    fun `decrypt fails closed when ciphertext bytes are corrupted`() {
        val original = FloatArray(160) { 0.42f }
        val id = "profile-uuid-1234"
        val contactName = "Alice Mother"

        val enc = crypto.encrypt(original, id, contactName)
        val corruptedCiphertext = enc.ciphertext.copyOf()
        corruptedCiphertext[0] = (corruptedCiphertext[0].toInt() xor 0xFF).toByte()

        val tamperedDecrypted = crypto.decrypt(corruptedCiphertext, enc.iv, id, contactName)
        assertEquals(0, tamperedDecrypted.size)
    }

    @Test
    fun `VoiceProfileEntity seals embedding into AES-GCM ciphertext`() {
        val original = floatArrayOf(0.123f, -0.456f, 0.789f)
        val entity = VoiceProfileEntity.fromEmbeddingArray(
            id = "test-id-1",
            contactName = "Mom",
            relationship = "Mother",
            enrolledAtMs = 12345678L,
            sampleDurationMs = 8500L,
            embedding = original,
            crypto = crypto,
        )

        // Verify entity fields are encrypted binary
        assertTrue(entity.encryptedEmbedding.isNotEmpty())
        assertEquals(12, entity.encryptionIv.size)

        // Verify accurate recovery via toEmbeddingArray
        val recovered = entity.toEmbeddingArray(crypto)
        assertEquals(original.size, recovered.size)
        for (i in original.indices) {
            assertEquals(original[i], recovered[i], 1e-6f)
        }

        // Verify record tampering detection
        val tamperedEntity = entity.copy(contactName = "Dad")
        val tamperedRecovered = tamperedEntity.toEmbeddingArray(crypto)
        assertEquals(0, tamperedRecovered.size)
    }
}
