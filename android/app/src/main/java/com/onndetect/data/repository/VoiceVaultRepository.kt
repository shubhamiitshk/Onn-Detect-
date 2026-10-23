package com.onndetect.data.repository

import com.onndetect.data.local.SealedBlob
import com.onndetect.data.local.VoiceProfileDao
import com.onndetect.data.local.VoiceProfileEntity
import com.onndetect.data.local.VoiceVaultCrypto
import com.onndetect.data.local.vaultAad
import kotlinx.coroutines.flow.Flow
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Family Safe-Voice Vault repository.
 *
 * F3: every embedding is SEALED with AES-256-GCM before it reaches Room.
 * The GCM AAD is bound to the row identity (`vaultAad(id, contactName)`),
 * so a sealed blob cannot be transplanted onto another profile row, and a
 * rename of a row invalidates its stored ciphertext loudly.
 */
@Singleton
class VoiceVaultRepository @Inject constructor(
    private val dao: VoiceProfileDao,
    private val crypto: VoiceVaultCrypto,
) {
    fun getAllProfiles(): Flow<List<VoiceProfileEntity>> = dao.getAll()

    suspend fun getAllProfilesList(): List<VoiceProfileEntity> = dao.getAllList()

    /**
     * Unseals a stored embedding. Returns null ONLY for rows that are
     * neither sealed nor legacy (a corrupt row) — an authentication failure
     * (tampered ciphertext / swapped AAD) is rethrown, because it must never
     * degrade into "no match".
     */
    fun unsealedEmbedding(profile: VoiceProfileEntity): FloatArray? {
        profile.encryptedEmbedding?.let { sealed ->
            val iv = profile.embeddingIv ?: return null
            val plaintext = crypto.unseal(SealedBlob(iv, sealed), vaultAad(profile.id, profile.contactName))
            return decodeEmbedding(plaintext)
        }
        profile.embeddingCsv?.let { csv ->
            if (csv.isNotBlank()) return decodeEmbedding(csv.encodeToByteArray())
        }
        return null
    }

    suspend fun enrollProfile(
        contactName: String,
        relationship: String,
        sampleDurationMs: Long,
        embedding: FloatArray,
        spoofScore: Float?,
        sampleSha256: String? = null,
    ): VoiceProfileEntity {
        require(contactName.isNotBlank()) { "Contact name cannot be blank" }
        require(embedding.isNotEmpty()) { "Voiceprint embedding cannot be empty" }

        val id = UUID.randomUUID().toString()
        val contact = contactName.trim()
        val blob = crypto.seal(encodeEmbedding(embedding), vaultAad(id, contact))
        val entity = VoiceProfileEntity(
            id = id,
            contactName = contact,
            relationship = relationship.trim().ifBlank { "Family" },
            enrolledAtMs = System.currentTimeMillis(),
            sampleDurationMs = sampleDurationMs,
            embeddingCsv = null, // sealed rows never carry plaintext
            sampleSha256 = sampleSha256,
            encryptedEmbedding = blob.ciphertext,
            embeddingIv = blob.iv,
            spoofScore = spoofScore,
            embeddingVersion = VoiceProfileEntity.CURRENT_EMBEDDING_VERSION,
        )
        dao.insert(entity)
        return entity
    }

    /**
     * Seals every pre-v2 row still stored as plaintext and blanks the CSV
     * column. Called once at vault open; idempotent. Rows whose plaintext
     * fails to parse (corrupt) are left untouched and reported, never
     * silently deleted.
     */
    suspend fun ensureLegacyRowsSealed(): Int {
        var sealed = 0
        for (row in dao.legacyPlaintextRows()) {
            val csv = row.embeddingCsv ?: continue
            val embedding = runCatching { decodeEmbedding(csv.encodeToByteArray()) }.getOrNull()
            if (embedding == null || embedding.isEmpty()) continue // corrupt: report, never destroy
            val blob = crypto.seal(encodeEmbedding(embedding), vaultAad(row.id, row.contactName))
            dao.updateSealed(
                id = row.id,
                sealed = blob.ciphertext,
                iv = blob.iv,
                spoofScore = row.spoofScore,
                version = row.embeddingVersion ?: VoiceProfileEntity.CURRENT_EMBEDDING_VERSION,
            )
            sealed++
        }
        return sealed
    }

    suspend fun deleteProfile(id: String): Boolean = dao.deleteById(id) > 0

    suspend fun getProfileCount(): Int = dao.count()

    // ── float[] <-> CSV/bytes codecs (IEEE-754 layout preserved) ──────────

    private fun encodeEmbedding(embedding: FloatArray): ByteArray {
        // "dim,hex-per-float" keeps ASCII debuggability of v1 while never
        // storing a parseable plaintext: this runs in memory only, and the
        // bytes are immediately AES-GCM sealed.
        val sb = StringBuilder(embedding.size * 10)
        for ((i, v) in embedding.withIndex()) {
            if (i > 0) sb.append(',')
            sb.append(java.lang.Float.toString(v))
        }
        return sb.toString().toByteArray(Charsets.US_ASCII)
    }

    private fun decodeEmbedding(bytes: ByteArray): FloatArray? {
        val text = bytes.toString(Charsets.US_ASCII)
        if (text.isBlank()) return null
        return runCatching {
            text.split(',').map { it.trim().toFloat() }.toFloatArray()
        }.getOrNull()
    }
}
