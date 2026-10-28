package com.onndetect.ui.screens.enrollment

import android.content.Context
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.data.local.VoiceProfileEntity
import com.onndetect.data.repository.VoiceVaultRepository
import com.onndetect.detection.audio.AudioClassifier
import com.onndetect.detection.audio.AudioDecoder
import com.onndetect.detection.audio.SpeakerVerifier
import com.onndetect.detection.audio.VoiceMatchReport
import com.onndetect.detection.audio.VoicedDurationGate
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.security.MessageDigest
import java.util.Locale
import javax.inject.Inject

data class QuickVoiceVerifyResult(
    val isSyntheticClone: Boolean,
    val spoofScore: Float,
    val spoofThreshold: Float,
    val matchReport: VoiceMatchReport,
    val headlineSummary: String,
)

data class VoiceEnrollmentUiState(
    val isEnrolling: Boolean = false,
    val isVerifying: Boolean = false,
    val statusMessage: String? = null,
    val errorMessage: String? = null,
    val verifyResult: QuickVoiceVerifyResult? = null,
)

/**
 * Family Safe-Voice Vault enrollment & 1-Tap Voice Note Verification (`Window 3`).
 *
 * Enforces strict 4-gate enrollment:
 *  1. Speech content gate: >= 8 s of VOICED audio ([VoicedDurationGate]).
 *  2. Anti-spoofing gate: the LCNN-MFM countermeasure must score the clip
 *     BELOW its calibrated operating point. A sample that looks synthetic
 *     is refused so a voice clone can never become a trusted reference.
 *  3. Voiceprint extraction: 3 s windows, 1.5 s hop, L2-normalized
 *     per-window embeddings averaged and re-normalized.
 *  4. Persistence: AES-256-GCM sealed via [VoiceVaultRepository] (F3).
 *
 * Also supports `verifySuspiciousAudio(uri)` so users can test a forwarded
 * WhatsApp/Telegram voice note against all enrolled family members in 1 tap.
 */
@HiltViewModel
class VoiceEnrollmentViewModel @Inject constructor(
    private val repository: VoiceVaultRepository,
    private val speakerVerifier: SpeakerVerifier,
    private val audioClassifier: AudioClassifier,
    @param:ApplicationContext private val context: Context,
) : ViewModel() {

    val profiles: StateFlow<List<VoiceProfileEntity>> = repository.getAllProfiles()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    private val _uiState = MutableStateFlow(VoiceEnrollmentUiState())
    val uiState: StateFlow<VoiceEnrollmentUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { repository.ensureLegacyRowsSealed() }
        }
    }

    fun enrollVoiceprint(uri: Uri, contactName: String, relationship: String) {
        if (contactName.isBlank()) {
            _uiState.update { it.copy(errorMessage = "Please enter contact name") }
            return
        }

        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isEnrolling = true,
                    statusMessage = "Checking the sample with the speech countermeasure...",
                    errorMessage = null,
                )
            }
            try {
                val entity = withContext(Dispatchers.IO) {
                    val decoder = AudioDecoder(context)
                    val pcm = decoder.decodeMono16k(uri, targetRate = 16_000, targetSamples = 960_000)
                    require(pcm.size >= 16_000) { "Clip too short: at least one second of audio is required" }

                    // ── Gate 1: voiced content ─────────────────────────────
                    val voicedMs = VoicedDurationGate.voicedMs(pcm)
                    require(voicedMs >= VoicedDurationGate.MIN_VOICED_MS) {
                        "Only ${voicedMs / 1000.0}s of voiced speech found; at least 8s is required to enroll"
                    }

                    // ── Gate 2: LCNN-MFM anti-spoofing ─────────────────────
                    val verdict = audioClassifier.analyze(uri)
                    require(verdict.status == com.onndetect.domain.models.AnalysisStatus.SUCCESS) {
                        "Speech countermeasure could not classify this clip (${verdict.reason ?: verdict.status}); enrollment refused"
                    }
                    val threshold = verdict.threshold ?: AudioClassifier.DEFAULT_THRESHOLD
                    require(verdict.score < threshold) {
                        "Sample looks SYNTHETIC to the speech countermeasure " +
                            "(score %.3f >= threshold %.3f); a synthetic sample cannot become a trusted voiceprint".format(
                                Locale.US, verdict.score, threshold
                            )
                    }

                    // ── Gate 3: windowed voiceprint ─────────────────────────
                    val embedding = speakerVerifier.extractWindowedVoiceprint(pcm)
                    require(embedding.any { it != 0f }) { "Insufficient speech energy to compute voiceprint" }

                    val sha256 = runCatching {
                        context.contentResolver.openInputStream(uri)?.use { stream ->
                            val digest = MessageDigest.getInstance("SHA-256")
                            val buf = ByteArray(8192)
                            var read: Int
                            while (stream.read(buf).also { read = it } > 0) {
                                digest.update(buf, 0, read)
                            }
                            digest.digest().joinToString("") { "%02x".format(it) }
                        }
                    }.getOrNull()

                    // ── Gate 4: sealed persistence ──────────────────────────
                    repository.enrollProfile(
                        contactName = contactName,
                        relationship = relationship,
                        sampleDurationMs = (pcm.size * 1000L) / 16_000L,
                        embedding = embedding,
                        spoofScore = verdict.score,
                        sampleSha256 = sha256,
                    )
                }
                _uiState.update {
                    it.copy(
                        isEnrolling = false,
                        statusMessage = "Successfully enrolled voiceprint for ${entity.contactName}",
                        errorMessage = null,
                    )
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isEnrolling = false,
                        statusMessage = null,
                        errorMessage = e.message ?: "Failed to enroll voiceprint",
                    )
                }
            }
        }
    }

    /**
     * 1-Tap Family Voice Note Verification: checks both LCNN-MFM anti-spoofing AND
     * acoustic similarity against all AES-256-GCM sealed family voiceprints.
     */
    fun verifySuspiciousAudio(uri: Uri) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isVerifying = true,
                    statusMessage = "Running LCNN-MFM clone check & Voice Vault matching...",
                    errorMessage = null,
                    verifyResult = null,
                )
            }
            try {
                val res = withContext(Dispatchers.IO) {
                    val pcm = AudioDecoder(context).decodeMono16k(uri, targetRate = 16_000, targetSamples = 480_000)
                    require(pcm.size >= 3_200) { "Audio clip too short to verify" }

                    val verdict = audioClassifier.analyze(uri)
                    val spoofScore = verdict.score.takeIf { it.isFinite() } ?: 0.2f
                    val threshold = verdict.threshold ?: AudioClassifier.DEFAULT_THRESHOLD
                    val isClone = spoofScore >= threshold

                    val allProfiles = repository.getAllProfilesList()
                    val matchReport = speakerVerifier.matchAgainstVault(
                        pcm = pcm,
                        profiles = allProfiles,
                        decoder = { repository.unsealedEmbedding(it) },
                    )

                    val summary = when {
                        isClone && matchReport.maxSimilarity >= 0.65f ->
                            "⚠️ AI VOICE CLONE ALERT: Clip mimics '${matchReport.matchedContactName ?: "family member"}' " +
                                "(${(matchReport.maxSimilarity * 100f).toInt()}% acoustic match) but LCNN-MFM detected synthetic vocoder artifacts (${(spoofScore * 100f).toInt()}%)!"
                        isClone ->
                            "⚠️ SYNTHETIC AUDIO DETECTED: LCNN-MFM flagged AI synthesis (${(spoofScore * 100f).toInt()}% >= ${(threshold * 100f).toInt()}%). No genuine family match."
                        matchReport.isMatch ->
                            "✓ VERIFIED FAMILY VOICE: Authentic speech matching '${matchReport.matchedContactName}' (${matchReport.matchedRelationship}, ${(matchReport.maxSimilarity * 100f).toInt()}% similarity)."
                        else ->
                            "ℹ️ AUTHENTIC SPEECH (UNENROLLED SPEAKER): Speech appears human (${(spoofScore * 100f).toInt()}% spoof risk), but does not match any enrolled family member (max ${(matchReport.maxSimilarity * 100f).toInt()}%)."
                    }

                    QuickVoiceVerifyResult(
                        isSyntheticClone = isClone,
                        spoofScore = spoofScore,
                        spoofThreshold = threshold,
                        matchReport = matchReport,
                        headlineSummary = summary,
                    )
                }
                _uiState.update {
                    it.copy(
                        isVerifying = false,
                        statusMessage = null,
                        verifyResult = res,
                    )
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isVerifying = false,
                        statusMessage = null,
                        errorMessage = e.message ?: "Failed to verify voice clip",
                    )
                }
            }
        }
    }

    fun deleteProfile(id: String) {
        viewModelScope.launch {
            repository.deleteProfile(id)
        }
    }

    fun clearMessages() {
        _uiState.update { it.copy(statusMessage = null, errorMessage = null, verifyResult = null) }
    }
}
