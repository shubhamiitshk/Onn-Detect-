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
import com.onndetect.domain.models.AnalysisStatus
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
import kotlin.math.sqrt

data class VoiceEnrollmentUiState(
    val isEnrolling: Boolean = false,
    val statusMessage: String? = null,
    val errorMessage: String? = null,
)

@HiltViewModel
class VoiceEnrollmentViewModel @Inject constructor(
    private val repository: VoiceVaultRepository,
    private val speakerVerifier: SpeakerVerifier,
    private val audioClassifier: AudioClassifier,
    @param:ApplicationContext private val context: Context,
) : ViewModel() {

    companion object {
        const val MIN_VOICED_DURATION_MS = 8000L // 8.0s minimum voiced speech
        const val MAX_DEEPFAKE_SCORE = 0.375f    // LCNN threshold gate
        const val WINDOW_DURATION_SAMPLES = 48000 // 3.0s at 16kHz
        const val HOP_DURATION_SAMPLES = 24000   // 1.5s at 16kHz
        const val DECODE_TARGET_SAMPLES = 192000 // Up to 12.0s at 16kHz
    }

    val profiles: StateFlow<List<VoiceProfileEntity>> = repository.getAllProfiles()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    private val _uiState = MutableStateFlow(VoiceEnrollmentUiState())
    val uiState: StateFlow<VoiceEnrollmentUiState> = _uiState.asStateFlow()

    fun enrollVoiceprint(uri: Uri, contactName: String, relationship: String) {
        if (contactName.isBlank()) {
            _uiState.update { it.copy(errorMessage = "Please enter contact name") }
            return
        }

        viewModelScope.launch {
            _uiState.update { it.copy(isEnrolling = true, statusMessage = "Verifying acoustic authenticity...", errorMessage = null) }
            try {
                val entity = withContext(Dispatchers.Default) {
                    // 1. Anti-Spoofing Gate: Verify audio is not synthetic / cloned
                    val antiSpoofVerdict = audioClassifier.analyze(uri)
                    if (antiSpoofVerdict.status == AnalysisStatus.SUCCESS) {
                        require(antiSpoofVerdict.score < MAX_DEEPFAKE_SCORE && !antiSpoofVerdict.isAiGenerated) {
                            "Enrollment rejected: Audio detected as synthetic/AI-cloned (score: ${"%.3f".format(java.util.Locale.US, antiSpoofVerdict.score)} >= $MAX_DEEPFAKE_SCORE). Cannot enroll deepfakes in Family Vault."
                        }
                    }

                    // 2. Decode audio up to 12 seconds
                    val decoder = AudioDecoder(context)
                    val pcm = decoder.decodeMono16k(uri, targetRate = 16000, targetSamples = DECODE_TARGET_SAMPLES)
                    require(pcm.size >= 16000) { "Audio sample is too short to analyze" }

                    // 3. Voiced Duration Gate (>= 8,000 ms)
                    val voicedDurationMs = computeVoicedDurationMs(pcm)
                    require(voicedDurationMs >= MIN_VOICED_DURATION_MS) {
                        "Insufficient voiced speech (${voicedDurationMs}ms < ${MIN_VOICED_DURATION_MS}ms required). Please provide at least 8 seconds of clear speech."
                    }

                    // 4. Sliding-Window Centroid Extraction (3s window, 1.5s hop)
                    val centroidEmbedding = computeCentroidVoiceprint(pcm)
                    require(centroidEmbedding.any { it != 0f }) { "Insufficient acoustic speech energy to compute voiceprint" }

                    // 5. SHA-256 Media Digest
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

                    // 6. Seal with AES-256-GCM + AAD into encrypted Vault
                    repository.enrollProfile(
                        contactName = contactName,
                        relationship = relationship,
                        sampleDurationMs = (pcm.size * 1000L) / 16000L,
                        embedding = centroidEmbedding,
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

    private fun computeVoicedDurationMs(pcm: FloatArray, sampleRate: Int = 16000): Long {
        val frameSize = (sampleRate * 0.025).toInt() // 25ms
        val frameHop = (sampleRate * 0.010).toInt()  // 10ms
        if (pcm.size < frameSize) return 0L
        var voicedFrames = 0
        val silenceRmsFloor = 0.012f
        var i = 0
        while (i + frameSize <= pcm.size) {
            var sumSq = 0.0
            for (j in 0 until frameSize) {
                val s = pcm[i + j]
                sumSq += s * s
            }
            val rms = sqrt(sumSq / frameSize).toFloat()
            if (rms >= silenceRmsFloor) {
                voicedFrames++
            }
            i += frameHop
        }
        return voicedFrames * 10L
    }

    private fun computeCentroidVoiceprint(pcm: FloatArray): FloatArray {
        val windowEmbeddings = mutableListOf<FloatArray>()
        var offset = 0
        while (offset + WINDOW_DURATION_SAMPLES <= pcm.size) {
            val window = pcm.copyOfRange(offset, offset + WINDOW_DURATION_SAMPLES)
            var energy = 0.0
            for (s in window) energy += s * s
            val rms = sqrt(energy / WINDOW_DURATION_SAMPLES).toFloat()
            if (rms >= 0.012f) {
                val emb = speakerVerifier.extractVoiceprint(window)
                if (emb.any { it != 0f }) {
                    windowEmbeddings.add(emb)
                }
            }
            offset += HOP_DURATION_SAMPLES
        }

        if (windowEmbeddings.isEmpty()) {
            return speakerVerifier.extractVoiceprint(pcm)
        }

        val dim = windowEmbeddings[0].size
        val centroid = FloatArray(dim)
        for (emb in windowEmbeddings) {
            for (d in 0 until dim) {
                centroid[d] += emb[d]
            }
        }
        var normSq = 0.0
        for (d in 0 until dim) {
            centroid[d] /= windowEmbeddings.size
            normSq += centroid[d] * centroid[d]
        }
        val norm = sqrt(normSq).toFloat()
        if (norm > 1e-6f) {
            for (d in 0 until dim) {
                centroid[d] /= norm
            }
        }
        return centroid
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
