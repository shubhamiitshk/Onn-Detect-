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
}
