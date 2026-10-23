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
import javax.inject.Inject

data class VoiceEnrollmentUiState(
    val isEnrolling: Boolean = false,
    val statusMessage: String? = null,
    val errorMessage: String? = null,
)

/**
 * Family Safe-Voice Vault enrollment — F4 rewrite.
 *
 * The pre-audit flow decoded exactly 4 s of audio, accepted any non-zero
 * voiceprint, and had NO idea whether the sample was synthetic. It enrolled
 * whatever it was given, including a clone of the very voice it should
 * distrust. The gate order below is fixed on purpose: authenticity is
 * checked BEFORE the voiceprint is written, and every requirement is a
 * hard rejection with a human-readable reason:
 *
 *  1. Speech content gate: >= 8 s of VOICED audio ([VoicedDurationGate]).
 *  2. Anti-spoofing gate: the LCNN-MFM countermeasure must score the clip
 *     BELOW its calibrated 0.375 operating point (and reach a SUCCESS
 *     verdict, i.e. actually classify as speech). A sample that looks
 *     synthetic is refused — a voice clone is never allowed to become the
 *     trusted reference for a family member.
 *  3. Voiceprint extraction: 3 s windows, 1.5 s hop, L2-normalized
 *     per-window embeddings averaged and re-normalized
 *     ([SpeakerVerifier.extractWindowedVoiceprint]).
 *  4. Persistence: AES-256-GCM sealed via [VoiceVaultRepository] (F3).
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
        // F3: upgrade any pre-v2 plaintext rows to sealed form at open time.
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

                    // Full-clip decode (the 4 s cap is gone): bounded by the
                    // decoder's own budget so a 10-minute file cannot hang the UI.
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
                                verdict.score, threshold
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

    fun deleteProfile(id: String) {
        viewModelScope.launch {
            repository.deleteProfile(id)
        }
    }

    fun clearMessages() {
        _uiState.update { it.copy(statusMessage = null, errorMessage = null) }
    }
}
