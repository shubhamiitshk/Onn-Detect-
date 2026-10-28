package com.onndetect.ui.screens.modelmanager

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer
import com.onndetect.detection.visual.VlmPackManager
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class ModelManagerUiState(
    val deviceProfile: VlmPackManager.DeviceHardwareProfile,
    val catalog: List<VlmPackManager.VlmCatalogEntry>,
    val installedPack: VlmPackManager.InstalledVlmPackDetails?,
    val isImporting: Boolean = false,
    val importProgressText: String = "",
    val errorMessage: String? = null,
    val successMessage: String? = null,
    val allowCustomKey: Boolean = false,
    val cascadeEnabled: Boolean = false,
)

@HiltViewModel
class ModelManagerViewModel @Inject constructor(
    application: Application,
) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow(buildInitialState())
    val uiState: StateFlow<ModelManagerUiState> = _uiState.asStateFlow()

    private fun buildInitialState(): ModelManagerUiState {
        val app = getApplication<Application>()
        val profile = VlmPackManager.inspectDevice(app)
        val catalog = VlmPackManager.getCatalog(profile)
        val installed = VlmPackManager.getInstalledPackDetails(app)

        return ModelManagerUiState(
            deviceProfile = profile,
            catalog = catalog,
            installedPack = installed,
            allowCustomKey = OnDemandForensicVlmAnalyzer.allowUntrustedPublisherForTesting,
            cascadeEnabled = OnDemandForensicVlmAnalyzer.experimentalVlmCascadeEnabled,
        )
    }

    fun refreshInstalledPack() {
        val app = getApplication<Application>()
        val installed = VlmPackManager.getInstalledPackDetails(app)
        _uiState.update {
            it.copy(
                installedPack = installed,
                cascadeEnabled = OnDemandForensicVlmAnalyzer.experimentalVlmCascadeEnabled,
                allowCustomKey = OnDemandForensicVlmAnalyzer.allowUntrustedPublisherForTesting,
            )
        }
    }

    fun importZip(uri: Uri) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isImporting = true,
                    importProgressText = "Preparing pack import...",
                    errorMessage = null,
                    successMessage = null,
                )
            }

            val result = VlmPackManager.importPackZip(
                context = getApplication(),
                zipUri = uri,
                allowCustomKey = _uiState.value.allowCustomKey,
                onProgress = { step ->
                    _uiState.update { it.copy(importProgressText = step) }
                }
            )

            result.fold(
                onSuccess = { pack ->
                    _uiState.update {
                        it.copy(
                            isImporting = false,
                            installedPack = pack,
                            cascadeEnabled = true,
                            successMessage = "Successfully installed and verified '${pack.packId}'! Tier-2 escalation is active.",
                            errorMessage = null,
                        )
                    }
                },
                onFailure = { err ->
                    _uiState.update {
                        it.copy(
                            isImporting = false,
                            errorMessage = err.localizedMessage ?: "Failed to import pack archive",
                            successMessage = null,
                        )
                    }
                }
            )
        }
    }

    fun uninstall() {
        val app = getApplication<Application>()
        val success = VlmPackManager.uninstallPack(app)
        if (success) {
            _uiState.update {
                it.copy(
                    installedPack = null,
                    cascadeEnabled = false,
                    successMessage = "VLM feature pack uninstalled successfully. Storage freed.",
                    errorMessage = null,
                )
            }
        } else {
            _uiState.update {
                it.copy(errorMessage = "Failed to uninstall pack")
            }
        }
    }

    fun toggleCascade(enabled: Boolean) {
        OnDemandForensicVlmAnalyzer.experimentalVlmCascadeEnabled = enabled
        _uiState.update { it.copy(cascadeEnabled = enabled) }
    }

    fun toggleAllowCustomKey(enabled: Boolean) {
        OnDemandForensicVlmAnalyzer.allowUntrustedPublisherForTesting = enabled
        _uiState.update { it.copy(allowCustomKey = enabled) }
    }

    fun clearMessages() {
        _uiState.update { it.copy(errorMessage = null, successMessage = null) }
    }
}
