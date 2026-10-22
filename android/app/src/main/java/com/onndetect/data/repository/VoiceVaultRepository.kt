package com.onndetect.data.repository

import com.onndetect.data.local.VoiceProfileDao
import com.onndetect.data.local.VoiceProfileEntity
import kotlinx.coroutines.flow.Flow
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class VoiceVaultRepository @Inject constructor(
    private val dao: VoiceProfileDao,
) {
    fun getAllProfiles(): Flow<List<VoiceProfileEntity>> = dao.getAll()

    suspend fun getAllProfilesList(): List<VoiceProfileEntity> = dao.getAllList()

    suspend fun enrollProfile(
        contactName: String,
        relationship: String,
        sampleDurationMs: Long,
        embedding: FloatArray,
        sampleSha256: String? = null,
    ): VoiceProfileEntity {
        require(contactName.isNotBlank()) { "Contact name cannot be blank" }
        require(embedding.isNotEmpty()) { "Voiceprint embedding cannot be empty" }

        val entity = VoiceProfileEntity.fromEmbeddingArray(
            id = UUID.randomUUID().toString(),
            contactName = contactName.trim(),
            relationship = relationship.trim().ifBlank { "Family" },
            enrolledAtMs = System.currentTimeMillis(),
            sampleDurationMs = sampleDurationMs,
            embedding = embedding,
            sampleSha256 = sampleSha256,
        )
        dao.insert(entity)
        return entity
    }

    suspend fun deleteProfile(id: String): Boolean {
        return dao.deleteById(id) > 0
    }

    suspend fun getProfileCount(): Int = dao.count()
}
