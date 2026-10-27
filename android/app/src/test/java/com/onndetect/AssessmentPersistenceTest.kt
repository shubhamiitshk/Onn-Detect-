package com.onndetect

import com.onndetect.data.local.SignalBreakdownJson
import com.onndetect.data.repository.ScanHistoryRepository
import com.onndetect.domain.models.*
import com.onndetect.ui.components.assessmentStringId
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class AssessmentPersistenceTest {
    private val signals = SignalBreakdown(
        srmForensicScore = 0f,
        klDivergenceScore = Float.POSITIVE_INFINITY,
        audioModelScore = 0f,
    )
    private fun result() = ScanResult("test",ScanMode.AUDIO_ONLY,1L,0.2f,false,0f,signals,null,null,
        QualityInfo(100,false,null,false,false),10L)
    @Test fun zeroSurvivesRoundTripAndUnavailableValuesAreNotNumbers() {
        val encoded = SignalBreakdownJson.encode(signals)
        val o = JSONObject(encoded)
        assertTrue(o.isNull("fftSpectralScore")); assertTrue(o.isNull("klDivergenceScore"))
        val decoded = SignalBreakdownJson.decode(encoded)!!
        assertEquals(0f,decoded.srmForensicScore,0f); assertEquals(0f,decoded.audioModelScore,0f)
        assertTrue(decoded.fftSpectralScore.isNaN())
    }
    @Test fun corruptAndMissingSignalsDoNotCreateEvidence() {
        assertNull(SignalBreakdownJson.decode("broken"))
        assertNull(SignalBreakdownJson.decode("{\"schemaVersion\":99}"))
        assertTrue(SignalBreakdownJson.decode("{\"audioModelScore\":2}")!!.audioModelScore.isNaN())
        assertTrue(SignalBreakdownJson.decode("{}")!!.audioModelScore.isNaN())
    }
    @Test fun legacyOrUnknownVerdictsAlwaysAbstain() {
        assertEquals(R.string.inconclusive,assessmentStringId("SYNTHETIC",false))
        assertEquals(R.string.inconclusive,assessmentStringId("invalid",true))
    }
    @Test fun failureCannotPersistClassifiedVerdictEvenIfCallerContradictsIt() {
        assertThrows(IllegalArgumentException::class.java) {
            ScanHistoryRepository.toEntry(result().copy(status=AnalysisStatus.MODEL_FAILURE,
                verdict=DetectionVerdict.SYNTHETIC,modelValidated=true))
        }
    }
    @Test fun unvalidatedLegacyAssessmentCannotPersistAuthenticity() {
        assertEquals("INCONCLUSIVE",ScanHistoryRepository.toEntry(result()).verdict)
    }
    @Test fun evidencePersistsModelVersionsCoverageAndNoSourceExif() {
        val entry = ScanHistoryRepository.toEntry(result().copy(modelVersion="release1",analyzedDurationMs=4000L,
            totalDurationMs=10000L,analyzedTimestampsMs=listOf(1000L,5000L),provenance=ProvenanceInfo(false,false,mapOf("GPS" to "private"))))
        val json = JSONObject(entry.evidenceJson!!)
        assertEquals("release1",json.getString("modelVersion")); assertEquals(4000L,json.getLong("analyzedDurationMs"))
        assertEquals(2,json.getJSONArray("timestampsMs").length()); assertFalse(entry.evidenceJson!!.contains("private"))
    }
    @Test fun evidencePersistsKycInjectionReport() {
        val kyc = com.onndetect.detection.kyc.KycInjectionReport(
            isHardwareGenuine = false,
            virtualCameraDetected = true,
            hookDetected = true,
            emulatorDetected = false,
            cameraHardwareLevel = "LEGACY",
            physicalSensorVerified = false,
            injectionScore = 0.85f,
            flags = listOf("Dynamic hooking framework active", "Missing physical silicon sensor")
        )
        val entry = ScanHistoryRepository.toEntry(result().copy(kycInjectionReport = kyc))
        val json = JSONObject(entry.evidenceJson!!)
        assertTrue(json.has("kycInjection"))
        val kycJson = json.getJSONObject("kycInjection")
        assertFalse(kycJson.getBoolean("isHardwareGenuine"))
        assertTrue(kycJson.getBoolean("virtualCameraDetected"))
        assertTrue(kycJson.getBoolean("hookDetected"))
        assertEquals("LEGACY", kycJson.getString("cameraHardwareLevel"))
        assertEquals(0.85f, kycJson.getDouble("injectionScore").toFloat(), 0.001f)
        assertEquals(2, kycJson.getJSONArray("flags").length())
    }
    @Test fun evidencePersistsVoiceMatchReport() {
        val voiceMatch = com.onndetect.detection.audio.VoiceMatchReport(
            hasEnrolledProfiles = true,
            isMatch = true,
            matchedContactName = "Mom",
            matchedRelationship = "Mother",
            matchedProfileId = "mom-id",
            maxSimilarity = 0.92f,
            threshold = 0.80f,
            confidence = com.onndetect.detection.audio.VoiceMatchConfidence.HIGH_MATCH,
            candidateScores = mapOf("mom-id" to 0.92f),
            flags = listOf("High-confidence acoustic match with enrolled profile 'Mom'")
        )
        val entry = ScanHistoryRepository.toEntry(result().copy(voiceMatchReport = voiceMatch))
        val json = JSONObject(entry.evidenceJson!!)
        assertTrue(json.has("voiceMatch"))
        val vmJson = json.getJSONObject("voiceMatch")
        assertTrue(vmJson.getBoolean("hasEnrolledProfiles"))
        assertTrue(vmJson.getBoolean("isMatch"))
        assertEquals("Mom", vmJson.getString("matchedContactName"))
        assertEquals("Mother", vmJson.getString("matchedRelationship"))
        assertEquals(0.92f, vmJson.getDouble("maxSimilarity").toFloat(), 0.001f)
        assertEquals("HIGH_MATCH", vmJson.getString("confidence"))
        assertEquals(1, vmJson.getJSONArray("flags").length())
    }
}
