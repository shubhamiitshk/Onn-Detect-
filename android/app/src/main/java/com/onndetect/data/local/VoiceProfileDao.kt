package com.onndetect.data.local

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface VoiceProfileDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(profile: VoiceProfileEntity): Long

    @Delete
    suspend fun delete(profile: VoiceProfileEntity): Int

    @Query("DELETE FROM voice_vault_profiles WHERE id = :id")
    suspend fun deleteById(id: String): Int

    @Query("SELECT * FROM voice_vault_profiles ORDER BY enrolled_at_ms DESC")
    fun getAll(): Flow<List<VoiceProfileEntity>>

    @Query("SELECT * FROM voice_vault_profiles ORDER BY enrolled_at_ms DESC")
    suspend fun getAllList(): List<VoiceProfileEntity>

    @Query("SELECT * FROM voice_vault_profiles WHERE id = :id LIMIT 1")
    suspend fun findById(id: String): VoiceProfileEntity?

    @Query("SELECT COUNT(*) FROM voice_vault_profiles")
    suspend fun count(): Int

    @Query("DELETE FROM voice_vault_profiles")
    suspend fun clearAll(): Int

    /** Rows still carrying pre-v2 plaintext embeddings (F3 migration input). */
    @Query("SELECT * FROM voice_vault_profiles WHERE embedding_csv IS NOT NULL AND embedding_csv != ''")
    suspend fun legacyPlaintextRows(): List<VoiceProfileEntity>

    /** Replaces a legacy plaintext row with its sealed form and blanks the CSV. */
    @Query(
        "UPDATE voice_vault_profiles SET encrypted_embedding = :sealed, embedding_iv = :iv, " +
            "spoof_score = :spoofScore, embedding_version = :version, embedding_csv = NULL " +
            "WHERE id = :id"
    )
    suspend fun updateSealed(
        id: String,
        sealed: ByteArray,
        iv: ByteArray,
        spoofScore: Float?,
        version: Int,
    )
}
