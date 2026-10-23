package com.onndetect.data.local

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.KeyStore
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Android Keystore AES-256-GCM hardware/TEE cryptographic wrapper for the Family Safe-Voice Vault.
 *
 * Security Invariants:
 * 1. Hardware Backed: Uses Android Keystore AES-256-GCM (`alias = "onn_voice_vault_aes256"`).
 * 2. Authenticated Additional Data (AAD): Cryptographically binds `$id|$contactName` to the ciphertext.
 *    Any SQLite record tampering (modifying profile id or contact name) invalidates the GCM MAC tag.
 * 3. Unique Nonce: Generates a cryptographically random 12-byte IV for every encryption operation.
 * 4. Fallback for JVM: Gracefully shares an in-memory AES-256 key when AndroidKeyStore is absent (e.g. JVM unit tests).
 */
@Singleton
class VoiceVaultCrypto @Inject constructor() {

    companion object {
        const val KEY_ALIAS = "onn_voice_vault_aes256"
        private const val ANDROID_KEYSTORE = "AndroidKeyStore"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
        private const val GCM_TAG_LENGTH = 128
        private const val IV_LENGTH_BYTES = 12

        @Volatile
        private var jvmFallbackKey: SecretKey? = null
        private val lock = Any()
    }

    private val secureRandom = SecureRandom()

    private fun getOrCreateKey(): SecretKey {
        return try {
            val ks = KeyStore.getInstance(ANDROID_KEYSTORE).apply { load(null) }
            if (!ks.containsAlias(KEY_ALIAS)) {
                val kgen = KeyGenerator.getInstance("AES", ANDROID_KEYSTORE)
                val spec = android.security.keystore.KeyGenParameterSpec.Builder(
                    KEY_ALIAS,
                    android.security.keystore.KeyProperties.PURPOSE_ENCRYPT or
                        android.security.keystore.KeyProperties.PURPOSE_DECRYPT
                )
                    .setBlockModes(android.security.keystore.KeyProperties.BLOCK_MODE_GCM)
                    .setEncryptionPaddings(android.security.keystore.KeyProperties.ENCRYPTION_PADDING_NONE)
                    .setKeySize(256)
                    .build()
                kgen.init(spec)
                kgen.generateKey()
            }
            (ks.getEntry(KEY_ALIAS, null) as KeyStore.SecretKeyEntry).secretKey
        } catch (_: Throwable) {
            // JVM environment fallback (shared across instances during standard OpenJDK unit tests)
            jvmFallbackKey ?: synchronized(lock) {
                jvmFallbackKey ?: run {
                    val kgen = KeyGenerator.getInstance("AES")
                    kgen.init(256)
                    val k = kgen.generateKey()
                    jvmFallbackKey = k
                    k
                }
            }
        }
    }

    data class EncryptionOutput(
        val ciphertext: ByteArray,
        val iv: ByteArray,
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as EncryptionOutput
            if (!ciphertext.contentEquals(other.ciphertext)) return false
            if (!iv.contentEquals(other.iv)) return false
            return true
        }

        override fun hashCode(): Int {
            var result = ciphertext.contentHashCode()
            result = 31 * result + iv.contentHashCode()
            return result
        }
    }

    /**
     * Serializes a FloatArray into a little-endian byte array.
     */
    fun serializeFloats(floats: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(floats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (f in floats) buffer.putFloat(f)
        return buffer.array()
    }

    /**
     * Deserializes a little-endian byte array into a FloatArray.
     */
    fun deserializeFloats(bytes: ByteArray): FloatArray {
        if (bytes.size % 4 != 0) return FloatArray(0)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val floats = FloatArray(bytes.size / 4)
        for (i in floats.indices) floats[i] = buffer.float
        return floats
    }

    /**
     * Encrypts a voiceprint embedding vector using AES-256-GCM.
     * Binds [id] and [contactName] as AAD.
     */
    fun encrypt(embedding: FloatArray, id: String, contactName: String): EncryptionOutput {
        val plaintext = serializeFloats(embedding)
        val key = getOrCreateKey()
        val cipher = Cipher.getInstance(TRANSFORMATION)
        val iv = ByteArray(IV_LENGTH_BYTES).also { secureRandom.nextBytes(it) }
        val spec = GCMParameterSpec(GCM_TAG_LENGTH, iv)
        cipher.init(Cipher.ENCRYPT_MODE, key, spec)

        val aad = "$id|$contactName".toByteArray(Charsets.UTF_8)
        cipher.updateAAD(aad)
        val ciphertext = cipher.doFinal(plaintext)
        return EncryptionOutput(ciphertext = ciphertext, iv = iv)
    }

    /**
     * Decrypts a voiceprint embedding vector using AES-256-GCM.
     * Verifies AAD bound to [id] and [contactName].
     * Returns empty FloatArray if authentication fails or data is corrupted.
     */
    fun decrypt(ciphertext: ByteArray, iv: ByteArray, id: String, contactName: String): FloatArray {
        return try {
            val key = getOrCreateKey()
            val cipher = Cipher.getInstance(TRANSFORMATION)
            val spec = GCMParameterSpec(GCM_TAG_LENGTH, iv)
            cipher.init(Cipher.DECRYPT_MODE, key, spec)

            val aad = "$id|$contactName".toByteArray(Charsets.UTF_8)
            cipher.updateAAD(aad)
            val plaintext = cipher.doFinal(ciphertext)
            deserializeFloats(plaintext)
        } catch (_: Exception) {
            FloatArray(0)
        }
    }
}
