package com.onndetect.data.repository

import com.onndetect.data.local.ScanHistoryDao
import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.data.local.SignalBreakdownJson
import com.onndetect.domain.models.ScanResult
import kotlinx.coroutines.flow.Flow
import org.json.JSONArray
import org.json.JSONObject
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ScanHistoryRepository @Inject constructor(private val dao: ScanHistoryDao) {
    fun observeAll(): Flow<List<ScanHistoryEntity>> = dao.observeAll()
    suspend fun findById(id: String): ScanHistoryEntity? = dao.findById(id)
    suspend fun record(result: ScanResult) = dao.insert(toEntry(result))
    suspend fun delete(id: String) = dao.deleteById(id)
    suspend fun clear() = dao.clearAll()

    companion object {
        fun toEntry(result: ScanResult) = ScanHistoryEntity(
            id = result.id, timestamp = result.timestamp, scanMode = result.scanMode.name,
            score = result.overallScore.takeIf { it.isFinite() } ?: -1f,
            isAiGenerated = result.isAiGenerated, confidence = result.confidence,
            durationMs = result.scanDurationMs, signalsJson = SignalBreakdownJson.encode(result.signals),
            status = result.status.name, verdict = if (result.modelValidated && result.status.name == "SUCCESS") result.verdict.name else "INCONCLUSIVE", reason = result.reason,
            modelValidated = result.modelValidated && result.status.name == "SUCCESS",
            evidenceJson = JSONObject().apply {
                put("schemaVersion", 1)
                put("modelVersion", result.modelVersion)
                put("preprocessingVersion", result.preprocessingVersion)
                put("calibrationVersion", result.calibrationVersion)
                // Whether an ON-DEVICE evaluation verified the release. The
                // result screen shows this next to the verdict, so a
                // desktop-verified release is never presented as a fully
                // device-verified one.
                put("deviceValidated", result.releaseDeviceValidated)
                result.decisionThreshold?.takeIf { it.isFinite() }?.let { put("threshold", it) }
                put("analyzedDurationMs", result.analyzedDurationMs)
                put("totalDurationMs", result.totalDurationMs)
                put("timestampsMs", JSONArray(result.analyzedTimestampsMs))
                put("resolutionMinEdge", result.qualityInfo.resolution)
                put("compressed", result.qualityInfo.isCompressed)
                put("blurry", result.qualityInfo.isBlurry)
                put("dark", result.qualityInfo.isDark)
                // A detected marker is unverified. Never persist source EXIF/GPS by default.
                put("provenance", if (result.provenance?.hasC2PA == true) "unverified marker" else "not verified")
                result.heatmap?.takeIf { it.isNotEmpty() }?.let { hm ->
                    put("heatmap", hm.joinToString(",") { "%.3f".format(java.util.Locale.US, it) })
                }
            }.toString(),
        )
    }
}
