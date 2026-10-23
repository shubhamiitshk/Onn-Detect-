package com.onndetect.data.local

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Biometric voiceprint profile stored in the local, hardware-encrypted Family Safe-Voice Vault.
 * Holds the normalized acoustic speaker embedding, sealed via Android Keystore AES-256-GCM
 * with Authenticated Additional Data (AAD) bound to the profile id and contact name.
 */
@Entity(tableName = "voice_vault_profiles")
data class VoiceProfileEntity(
    @PrimaryKey
    val id: String,

    @ColumnInfo(name = "contact_name")
    val contactName: String,

    @ColumnInfo(name = "relationship")
    val relationship: String,

    @ColumnInfo(name = "enrolled_at_ms")
    val enrolledAtMs: Long,

    @ColumnInfo(name = "sample_duration_ms")
    val sampleDurationMs: Long,

    /**
     * AES-256-GCM encrypted binary representation of the speaker embedding vector.
     * Authenticated with AAD bound to "$id|$contactName".
     */
    @ColumnInfo(name = "encrypted_embedding")
    val encryptedEmbedding: ByteArray,

    /** 12-byte IV used for AES-GCM encryption. */
    @ColumnInfo(name = "encryption_iv")
    val encryptionIv: ByteArray,

    @ColumnInfo(name = "sample_sha256")
    val sampleSha256: String? = null,
) {
    /**
     * Decrypts the sealed voiceprint using default crypto.
     * Returns empty array if AAD verification fails (e.g. record tampering).
     */
    fun toEmbeddingArray(): FloatArray {
        return toEmbeddingArray(VoiceVaultCrypto())
    }

    /**
     * Decrypts the sealed voiceprint using [crypto].
     * Returns empty array if AAD verification fails (e.g. record tampering).
     */
    fun toEmbeddingArray(crypto: VoiceVaultCrypto = VoiceVaultCrypto()): FloatArray {
        return crypto.decrypt(encryptedEmbedding, encryptionIv, id, contactName)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as VoiceProfileEntity
        if (id != other.id) return false
        if (contactName != other.contactName) return false
        if (relationship != other.relationship) return false
        if (enrolledAtMs != other.enrolledAtMs) return false
        if (sampleDurationMs != other.sampleDurationMs) return false
        if (!encryptedEmbedding.contentEquals(other.encryptedEmbedding)) return false
        if (!encryptionIv.contentEquals(other.encryptionIv)) return false
        if (sampleSha256 != other.sampleSha256) return false
        return true
    }

    override fun hashCode(): Int {
        var result = id.hashCode()
        result = 31 * result + contactName.hashCode()
        result = 31 * result + relationship.hashCode()
        result = 31 * result + enrolledAtMs.hashCode()
        result = 31 * result + sampleDurationMs.hashCode()
        result = 31 * result + encryptedEmbedding.contentHashCode()
        result = 31 * result + encryptionIv.contentHashCode()
        result = 31 * result + (sampleSha256?.hashCode() ?: 0)
        return result
    }

    companion object {
        /**
         * Creates a VoiceProfileEntity from a raw embedding array (for testing and simple use cases).
         * Uses default encryption with a generated key.
         */
        fun fromEmbeddingArray(
            id: String,
            contactName: String,
            relationship: String,
            enrolledAtMs: Long,
            sampleDurationMs: Long,
            embedding: FloatArray,
        ): VoiceProfileEntity {
            // Use default crypto for simplicity in tests
            val crypto = VoiceVaultCrypto()
            val enc = crypto.encrypt(embedding, id, contactName)
            return VoiceProfileEntity(
                id = id,
                contactName = contactName,
                relationship = relationship,
                enrolledAtMs = enrolledAtMs,
                sampleDurationMs = sampleDurationMs,
                encryptedEmbedding = enc.ciphertext,
                encryptionIv = enc.iv,
                sampleSha256 = null,
            )
        }

        /**
         * Creates a VoiceProfileEntity from a raw embedding array with custom crypto.
         */
        fun fromEmbeddingArray(
            id: String,
            contactName: String,
            relationship: String,
            enrolledAtMs: Long,
            sampleDurationMs: Long,
            embedding: FloatArray,
            sampleSha256: String? = null,
            crypto: VoiceVaultCrypto = VoiceVaultCrypto(),
        ): VoiceProfileEntity {
            val enc = crypto.encrypt(embedding, id, contactName)
            return VoiceProfileEntity(
                id = id,
                contactName = contactName,
                relationship = relationship,
                enrolledAtMs = enrolledAtMs,
                sampleDurationMs = sampleDurationMs,
                encryptedEmbedding = enc.ciphertext,
                encryptionIv = enc.iv,
                sampleSha256 = sampleSha256,
            )
        }
    }
}
