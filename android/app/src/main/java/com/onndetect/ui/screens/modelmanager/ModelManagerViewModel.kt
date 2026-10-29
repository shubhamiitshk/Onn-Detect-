package com.onndetect.ui.screens.modelmanager

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.chat.ChatModelStore
import com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer
import com.onndetect.detection.visual.VlmPackManager
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import javax.inject.Inject

data class ModelManagerUiState(
    val deviceProfile: VlmPackManager.DeviceHardwareProfile,
    val installedPack: VlmPackManager.InstalledVlmPackDetails?,
    val chatStatus: ChatModelStore.Status = ChatModelStore.Status(null, false, "Checking conversation model…", false),
    val isImporting: Boolean = false,
    val importProgressText: String = "",
    val errorMessage: String? = null,
    val successMessage: String? = null,
    val allowCustomKey: Boolean = false,
    val cascadeEnabled: Boolean = false,
)

@HiltViewModel
class ModelManagerViewModel @Inject constructor(application: Application) : AndroidViewModel(application) {
    private val _uiState = MutableStateFlow(ModelManagerUiState(
        deviceProfile = VlmPackManager.inspectDevice(application),
        installedPack = VlmPackManager.getInstalledPackDetails(application),
        allowCustomKey = OnDemandForensicVlmAnalyzer.allowUntrustedPublisherForTesting,
        cascadeEnabled = OnDemandForensicVlmAnalyzer.experimentalVlmCascadeEnabled,
    ))
    val uiState: StateFlow<ModelManagerUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch { ChatModelStore.status.collect { status -> _uiState.update { it.copy(chatStatus = status) } } }
        refreshInstalledPack()
    }

    fun refreshInstalledPack() {
        viewModelScope.launch {
            try {
                val app = getApplication<Application>()
                ChatModelStore.refresh(app)
                val installed = withContext(Dispatchers.IO) { VlmPackManager.getInstalledPackDetails(app) }
                _uiState.update { it.copy(installedPack = installed, deviceProfile = VlmPackManager.inspectDevice(app)) }
            } catch (cancelled: CancellationException) { throw cancelled }
            catch (error: Exception) { _uiState.update { it.copy(errorMessage = error.message ?: "Could not inspect local model storage.") } }
        }
    }

    fun importChatZip(uri: Uri) {
        if (_uiState.value.isImporting) return
        viewModelScope.launch {
            _uiState.update { it.copy(isImporting = true, importProgressText = "Opening conversation pack…", errorMessage = null, successMessage = null) }
            try {
                val result = ChatModelStore.importPackZip(getApplication(), uri) { progress -> _uiState.update { it.copy(importProgressText = progress) } }
                result.fold(
                    onSuccess = { pack -> _uiState.update { it.copy(successMessage = "${pack.displayName} is ready for chat${if (pack.supportsVision) " and visual questions" else ""}.") } },
                    onFailure = { failure -> _uiState.update { it.copy(errorMessage = failure.message ?: "Could not import this conversation pack.") } },
                )
            } finally { _uiState.update { it.copy(isImporting = false, importProgressText = "") } }
        }
    }

    fun removeChatModel() {
        if (_uiState.value.isImporting) return
        viewModelScope.launch {
            try {
                ChatModelStore.uninstall(getApplication())
                _uiState.update { it.copy(successMessage = "Conversation model removed. Bundled detectors are unchanged.", errorMessage = null) }
            } catch (cancelled: CancellationException) { throw cancelled }
            catch (error: Exception) { _uiState.update { it.copy(errorMessage = error.message ?: "Could not remove the conversation model.") } }
        }
    }

    /** Legacy forensic research packs remain a separate explicit workflow. */
    fun importZip(uri: Uri) {
        if (_uiState.value.isImporting) return
        viewModelScope.launch {
            _uiState.update { it.copy(isImporting = true, importProgressText = "Opening forensic research pack…", errorMessage = null, successMessage = null) }
            try {
                val result = VlmPackManager.importPackZip(getApplication(), uri, _uiState.value.allowCustomKey) { progress ->
                    _uiState.update { it.copy(importProgressText = progress) }
                }
                result.fold(
                    onSuccess = { installed -> _uiState.update { it.copy(installedPack = installed,
                        cascadeEnabled = OnDemandForensicVlmAnalyzer.experimentalVlmCascadeEnabled,
                        successMessage = "Forensic research pack imported. Conversation model selection is unchanged.") } },
                    onFailure = { failure -> _uiState.update { it.copy(errorMessage = failure.message ?: "Could not import the research pack.") } },
                )
            } finally { _uiState.update { it.copy(isImporting = false, importProgressText = "") } }
        }
    }

    fun toggleAllowCustomKey(enabled: Boolean) {
        OnDemandForensicVlmAnalyzer.allowUntrustedPublisherForTesting = enabled
        _uiState.update { it.copy(allowCustomKey = enabled) }
    }

    fun clearMessages() = _uiState.update { it.copy(errorMessage = null, successMessage = null) }
}
