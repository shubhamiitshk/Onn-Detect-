package com.onndetect.data.local

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Biometric voiceprint profile stored in the Family Safe-Voice Vault.
 *
 * F3: the embedding is stored SEALED — AES-256-GCM under AndroidKeyStore
 * (alias `onn_voice_vault`), with the GCM AAD bound to this row's
 * (id, contactName). [encryptedEmbedding]/[embeddingIv] hold the sealed
 * bytes; the legacy [embeddingCsv] plaintext column exists ONLY so
 * pre-v2 rows can be migrated, and is NULL for every sealed row.
 *
 * [spoofScore] records the LCNN-MFM anti-spoof score at enrollment time
 * (F4: enrollment is refused when the sample looks synthetic), and
 * [embeddingVersion] tags the embedding frontend (1 = 160-dim DSP
 * voiceprint with windowed averaging).
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

    /** LEGACY: plaintext CSV. Non-null ONLY for not-yet-migrated pre-v2 rows. */
    @ColumnInfo(name = "embedding_csv", defaultValue = "")
    val embeddingCsv: String? = null,

    @ColumnInfo(name = "sample_sha256")
    val sampleSha256: String? = null,

    /** AES-256-GCM ciphertext of the embedding (128-bit tag appended). */
    @ColumnInfo(name = "encrypted_embedding", defaultValue = "")
    val encryptedEmbedding: ByteArray? = null,

    /** 12-byte GCM IV for [encryptedEmbedding]. */
    @ColumnInfo(name = "embedding_iv", defaultValue = "")
    val embeddingIv: ByteArray? = null,

    /** LCNN-MFM score observed at enrollment (must have been < 0.375). */
    @ColumnInfo(name = "spoof_score")
    val spoofScore: Float? = null,

    /** Embedding frontend version tag (1 = 160-dim DSP, windowed average). */
    @ColumnInfo(name = "embedding_version")
    val embeddingVersion: Int? = null,
) {
    val isSealed: Boolean get() = encryptedEmbedding != null && embeddingIv != null
    val hasLegacyPlaintext: Boolean get() = !embeddingCsv.isNullOrEmpty()

    /**
     * LEGACY decoder: parses the pre-v2 plaintext CSV column. Returns an
     * empty array for sealed rows (nothing plaintext exists to parse). The
     * production matching path uses the repository's unsealing decoder.
     */
    fun toEmbeddingArrayLegacy(): FloatArray {
        val csv = embeddingCsv ?: return FloatArray(0)
        if (csv.isBlank()) return FloatArray(0)
        return runCatching {
            csv.split(",").map { it.trim().toFloat() }.toFloatArray()
        }.getOrDefault(FloatArray(0))
    }

    override fun equals(other: Any?) = this === other
    override fun hashCode() = System.identityHashCode(this)

    companion object {
        const val EMBEDDING_DIM = 160
        const val CURRENT_EMBEDDING_VERSION = 1
    }
}
