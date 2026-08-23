package com.onndetect.data.local

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * One row per completed scan. Stored locally only; backup rules exclude the
 * database from any cloud/device transfer.
 */
@Entity(tableName = "scan_history")
data class ScanHistoryEntity(
    @PrimaryKey val id: String,
    @ColumnInfo(name = "timestamp") val timestamp: Long,
    @ColumnInfo(name = "scan_mode") val scanMode: String,
    @ColumnInfo(name = "score") val score: Float,
    @ColumnInfo(name = "is_ai_generated") val isAiGenerated: Boolean,
    @ColumnInfo(name = "confidence") val confidence: Float,
    @ColumnInfo(name = "duration_ms") val durationMs: Long,
)
