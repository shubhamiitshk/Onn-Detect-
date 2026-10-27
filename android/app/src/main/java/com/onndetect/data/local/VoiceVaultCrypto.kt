package com.onndetect.data.local

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.security.KeyStore
import java.util.Arrays
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec
import javax.inject.Inject
import javax.inject.Singleton

/**
 * AES-256-GCM sealing for biometric voiceprints — the F3 fix.
 *
 * Pre-audit state (commit 66fce33): the Family Safe-Voice Vault stored
 * speaker embeddings as PLAINTEXT CSV in Room while its documentation
 * claimed encryption. Any local attacker (or a leaked database file) could
 * read the raw biometric vectors of family members.
 *
 * Post-fix invariants:
 *  1. The AES-256 key is generated inside AndroidKeyStore (TEE/StrongBox)
 *     under [KEY_ALIAS] and is NON-EXPORTABLE: it never exists outside
 *     secure hardware and never enters the database or backups.
 *  2. Every seal binds the ciphertext to its row identity via GCM AAD
 *     (`aad(id, contactName)`): swapping ciphertexts between profiles —
 *     or editing a row's contact name — fails authentication loudly
 *     instead of decrypting to a wrong-but-plausible embedding.
 *  3. 12-byte random IV per seal, AES/GCM/NoPadding, 128-bit tag.
 *  4. AAD failures throw; callers must treat them as corruption, never
 *     as an empty embedding.
 */
interface VoiceVaultCrypto {
    /** Seals [plaintext]; each call produces a fresh IV. */
    fun seal(plaintext: ByteArray, aad: ByteArray): SealedBlob

    /**
     * Unseals a blob. Throws [javax.crypto.AEADBadTagException] when the
     * ciphertext, the IV or the AAD does not authenticate.
     */
    fun unseal(blob: SealedBlob, aad: ByteArray): ByteArray
}

/** IV + ciphertext(+tag) pair as stored in Room. */
data class SealedBlob(val iv: ByteArray, val ciphertext: ByteArray) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as SealedBlob

        if (!Arrays.equals(iv, other.iv)) return false
        if (!Arrays.equals(ciphertext, other.ciphertext)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = Arrays.hashCode(iv)
        result = 31 * result + Arrays.hashCode(ciphertext)
        return result
    }
}

/** Builds the AAD that binds a sealed embedding to its row identity. */
fun vaultAad(profileId: String, contactName: String): ByteArray =
    ("onn_voice_vault|v1|" + profileId + "|" + contactName).toByteArray(Charsets.UTF_8)

@Singleton
class AndroidKeyStoreVoiceVaultCrypto @Inject constructor() : VoiceVaultCrypto {

    companion object {
        const val KEY_ALIAS = "onn_voice_vault"
        const val IV_BYTES = 12
        const val TAG_BITS = 128
        private const val TRANSFORM = "AES/GCM/NoPadding"
    }

    private val keyStore: KeyStore by lazy {
        KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
    }

    private fun key(): SecretKey = synchronized(this) {
        (keyStore.getEntry(KEY_ALIAS, null) as? KeyStore.SecretKeyEntry)?.let { return it.secretKey }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        generator.init(
            KeyGenParameterSpec.Builder(KEY_ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .setRandomizedEncryptionRequired(true)
                .build()
        )
        generator.generateKey()
        (keyStore.getEntry(KEY_ALIAS, null) as? KeyStore.SecretKeyEntry)?.secretKey
            ?: error("AndroidKeyStore key generation failed for $KEY_ALIAS")
    }

    override fun seal(plaintext: ByteArray, aad: ByteArray): SealedBlob {
        val cipher = Cipher.getInstance(TRANSFORM)
        cipher.init(Cipher.ENCRYPT_MODE, key())
        cipher.updateAAD(aad)
        val ciphertext = cipher.doFinal(plaintext)
        return SealedBlob(iv = cipher.iv.copyOf().also { require(it.size == IV_BYTES) { "unexpected IV length" } },
            ciphertext = ciphertext)
    }

    override fun unseal(blob: SealedBlob, aad: ByteArray): ByteArray {
        val cipher = Cipher.getInstance(TRANSFORM)
        cipher.init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(TAG_BITS, blob.iv))
        cipher.updateAAD(aad)
        return cipher.doFinal(blob.ciphertext)
    }
}

/**
 * JVM test double: identical GCM discipline with a locally generated
 * AES-256 key (no AndroidKeyStore on the desktop). Used ONLY by unit tests;
 * production always binds [AndroidKeyStoreVoiceVaultCrypto].
 */
class InMemoryVoiceVaultCrypto : VoiceVaultCrypto {
    private val key: SecretKey = KeyGenerator.getInstance("AES").apply { init(256) }.generateKey()
    override fun seal(plaintext: ByteArray, aad: ByteArray): SealedBlob {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, key)
        cipher.updateAAD(aad)
        return SealedBlob(cipher.iv, cipher.doFinal(plaintext))
    }
    override fun unseal(blob: SealedBlob, aad: ByteArray): ByteArray {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, key, GCMParameterSpec(128, blob.iv))
        cipher.updateAAD(aad)
        return cipher.doFinal(blob.ciphertext)
    }
}
