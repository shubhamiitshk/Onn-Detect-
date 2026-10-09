package com.onndetect.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface ScanHistoryDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(entry: ScanHistoryEntity)

    @Query("SELECT * FROM scan_history ORDER BY timestamp DESC LIMIT 1000")
    fun observeAll(): Flow<List<ScanHistoryEntity>>

    @Query("SELECT * FROM scan_history WHERE id = :id LIMIT 1")
    suspend fun findById(id: String): ScanHistoryEntity?

    @Query("DELETE FROM scan_history WHERE id = :id")
    suspend fun deleteById(id: String)

    @Query("DELETE FROM scan_history")
    suspend fun clearAll()
}
