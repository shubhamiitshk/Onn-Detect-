package com.onndetect.data.local

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Biometric voiceprint profile stored in the local, encrypted Family Safe-Voice Vault.
 * Holds the normalized acoustic speaker embedding for matching forwarded voice notes.
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

    /** Comma-separated normalized acoustic speaker embedding vector. */
    @ColumnInfo(name = "embedding_csv")
    val embeddingCsv: String,

    @ColumnInfo(name = "sample_sha256")
    val sampleSha256: String? = null,
) {
    fun toEmbeddingArray(): FloatArray {
        if (embeddingCsv.isBlank()) return FloatArray(0)
        return try {
            embeddingCsv.split(",")
                .map { it.trim().toFloat() }
                .toFloatArray()
        } catch (_: Exception) {
            FloatArray(0)
        }
    }

    companion object {
        fun fromEmbeddingArray(
            id: String,
            contactName: String,
            relationship: String,
            enrolledAtMs: Long,
            sampleDurationMs: Long,
            embedding: FloatArray,
            sampleSha256: String? = null,
        ): VoiceProfileEntity {
            val csv = embedding.joinToString(",") { "%.6f".format(java.util.Locale.US, it) }
            return VoiceProfileEntity(
                id = id,
                contactName = contactName,
                relationship = relationship,
                enrolledAtMs = enrolledAtMs,
                sampleDurationMs = sampleDurationMs,
                embeddingCsv = csv,
                sampleSha256 = sampleSha256,
            )
        }
    }
}
