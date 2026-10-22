package com.onndetect.ui.screens.enrollment

import android.content.Context
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.data.local.VoiceProfileEntity
import com.onndetect.data.repository.VoiceVaultRepository
import com.onndetect.detection.audio.AudioDecoder
import com.onndetect.detection.audio.SpeakerVerifier
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

@HiltViewModel
class VoiceEnrollmentViewModel @Inject constructor(
    private val repository: VoiceVaultRepository,
    private val speakerVerifier: SpeakerVerifier,
    @param:ApplicationContext private val context: Context,
) : ViewModel() {

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
            _uiState.update { it.copy(isEnrolling = true, statusMessage = "Extracting acoustic voiceprint...", errorMessage = null) }
            try {
                val entity = withContext(Dispatchers.Default) {
                    val decoder = AudioDecoder(context)
                    val pcm = decoder.decodeMono16k(uri, targetRate = 16000, targetSamples = 64000)
                    val embedding = speakerVerifier.extractVoiceprint(pcm)
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

                    repository.enrollProfile(
                        contactName = contactName,
                        relationship = relationship,
                        sampleDurationMs = (pcm.size * 1000L) / 16000L,
                        embedding = embedding,
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
