package com.onndetect.ui.screens.modelmanager

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.chat.ChatModelStore
import com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer
import com.onndetect.detection.visual.VlmPackManager
import com.onndetect.ui.screens.scanner.cancellableProviderWork
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import javax.inject.Inject

data class ModelManagerUiState(
    val deviceProfile: VlmPackManager.DeviceHardwareProfile,
    val installedPack: VlmPackManager.InstalledVlmPackDetails?,
    val chatStatus: ChatModelStore.Status = ChatModelStore.Status(null, false, "Checking conversation model…", false),
    val isImporting: Boolean = false,
    val importProgressText: String = "",
    val selectedFileName: String? = null,
    val errorMessage: String? = null,
    val successMessage: String? = null,
    val allowCustomKey: Boolean = false,
    val cascadeEnabled: Boolean = false,
)

@HiltViewModel
class ModelManagerViewModel @Inject constructor(application: Application) : AndroidViewModel(application) {
    private var importJob: Job? = null
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
        _uiState.update { it.copy(isImporting = true, selectedFileName = null, importProgressText = "Opening selected file…", errorMessage = null, successMessage = null) }
        importJob = viewModelScope.launch {
            try {
                val name = selectedName(uri)
                _uiState.update { it.copy(selectedFileName = name) }
                val result = ChatModelStore.importPackZip(getApplication(), uri) { progress -> _uiState.update { it.copy(importProgressText = progress) } }
                result.fold(
                    onSuccess = { pack -> _uiState.update { it.copy(successMessage = "Added ${pack.displayName}.") } },
                    onFailure = { failure -> _uiState.update { it.copy(errorMessage = failure.message ?: "Could not import this conversation pack.") } },
                )
            } catch (cancelled: CancellationException) {
                _uiState.update { it.copy(successMessage = "Import stopped.") }
                throw cancelled
            } catch (failure: Exception) {
                _uiState.update { it.copy(errorMessage = failure.message ?: "This file could not be added. Choose a Qwen3.5 GGUF or ZIP.") }
            } finally {
                importJob = null
                _uiState.update { it.copy(isImporting = false, importProgressText = "") }
                refreshInstalledPack()
            }
        }
    }

    fun cancelImport() {
        if (!_uiState.value.isImporting) return
        _uiState.update { it.copy(importProgressText = "Stopping import…") }
        importJob?.cancel()
    }

    private suspend fun selectedName(uri: Uri): String = try {
        withTimeoutOrNull(2_000) {
            cancellableProviderWork {
                own(getApplication<Application>().contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null, signal))?.use { cursor ->
                    checkActive()
                    val column = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (column >= 0 && cursor.moveToFirst() && !cursor.isNull(column)) {
                        cursor.getString(column)?.filterNot { it.isISOControl() }?.take(160)
                    } else null
                }
            }
        }?.takeIf { it.isNotBlank() } ?: "Selected model file"
    } catch (cancelled: CancellationException) { throw cancelled }
    catch (_: Exception) { "Selected model file" }

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
        _uiState.update { it.copy(isImporting = true, selectedFileName = null, importProgressText = "Opening research pack…", errorMessage = null, successMessage = null) }
        importJob = viewModelScope.launch {
            try {
                val name = selectedName(uri)
                _uiState.update { it.copy(selectedFileName = name) }
                val result = VlmPackManager.importPackZip(getApplication(), uri, _uiState.value.allowCustomKey) { progress ->
                    _uiState.update { it.copy(importProgressText = progress) }
                }
                result.fold(
                    onSuccess = { installed -> _uiState.update { it.copy(installedPack = installed,
                        cascadeEnabled = OnDemandForensicVlmAnalyzer.experimentalVlmCascadeEnabled,
                        successMessage = "Forensic research pack imported. Conversation model selection is unchanged.") } },
                    onFailure = { failure -> _uiState.update { it.copy(errorMessage = failure.message ?: "Could not import the research pack.") } },
                )
            } catch (cancelled: CancellationException) {
                _uiState.update { it.copy(successMessage = "Import stopped.") }
                throw cancelled
            } catch (failure: Exception) {
                _uiState.update { it.copy(errorMessage = failure.message ?: "This research pack could not be added.") }
            } finally {
                importJob = null
                _uiState.update { it.copy(isImporting = false, importProgressText = "") }
                refreshInstalledPack()
            }
        }
    }

    fun toggleAllowCustomKey(enabled: Boolean) {
        OnDemandForensicVlmAnalyzer.allowUntrustedPublisherForTesting = enabled
        _uiState.update { it.copy(allowCustomKey = enabled) }
    }

    fun clearMessages() = _uiState.update { it.copy(errorMessage = null, successMessage = null) }
}
