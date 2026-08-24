package com.onndetect.data.repository

import com.onndetect.data.local.ScanHistoryDao
import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.data.local.SignalBreakdownJson
import com.onndetect.domain.models.ScanResult
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ScanHistoryRepository @Inject constructor(
    private val dao: ScanHistoryDao,
) {
    fun observeAll(): Flow<List<ScanHistoryEntity>> = dao.observeAll()

    suspend fun findById(id: String): ScanHistoryEntity? = dao.findById(id)

    suspend fun record(result: ScanResult) {
        dao.insert(
            ScanHistoryEntity(
                id = result.id,
                timestamp = result.timestamp,
                scanMode = result.scanMode.name,
                score = result.overallScore,
                isAiGenerated = result.isAiGenerated,
                confidence = result.confidence,
                durationMs = result.scanDurationMs,
                signalsJson = SignalBreakdownJson.encode(result.signals),
            )
        )
    }

    suspend fun delete(id: String) = dao.deleteById(id)

    suspend fun clear() = dao.clearAll()
}
