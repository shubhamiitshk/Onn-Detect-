package com.onndetect.data.local

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/** Local assessments only. Source media/URIs and biometric data are not stored. */
@Entity(tableName = "scan_history", indices = [Index(value = ["timestamp"])])
data class ScanHistoryEntity(
    @PrimaryKey val id: String,
    @ColumnInfo(name = "timestamp") val timestamp: Long,
    @ColumnInfo(name = "scan_mode") val scanMode: String,
    @ColumnInfo(name = "score") val score: Float,
    @ColumnInfo(name = "is_ai_generated") val isAiGenerated: Boolean,
    @ColumnInfo(name = "confidence") val confidence: Float,
    @ColumnInfo(name = "duration_ms") val durationMs: Long,
    @ColumnInfo(name = "signals_json") val signalsJson: String? = null,
    @ColumnInfo(name = "status", defaultValue = "'INCONCLUSIVE'") val status: String = "INCONCLUSIVE",
    @ColumnInfo(name = "verdict", defaultValue = "'INCONCLUSIVE'") val verdict: String = "INCONCLUSIVE",
    @ColumnInfo(name = "reason") val reason: String? = null,
    @ColumnInfo(name = "model_validated", defaultValue = "0") val modelValidated: Boolean = false,
    @ColumnInfo(name = "evidence_json") val evidenceJson: String? = null,
)
