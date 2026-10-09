package com.onndetect.ui.screens.result

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.data.repository.ScanHistoryRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

sealed interface ResultState {
    data object Loading : ResultState
    data object NotFound : ResultState
    data class Ready(val entry: ScanHistoryEntity) : ResultState
    data class Error(val message: String) : ResultState
}
@HiltViewModel
class ResultViewModel @Inject constructor(savedStateHandle: SavedStateHandle, private val repository: ScanHistoryRepository) : ViewModel() {
    private val scanId: String? = savedStateHandle["scanId"]
    private val _state = MutableStateFlow<ResultState>(ResultState.Loading)
    val state = _state.asStateFlow()
    init { reload() }
    fun reload() {
        _state.value = ResultState.Loading
        viewModelScope.launch {
            try {
                _state.value = scanId?.takeIf { it.isNotBlank() }?.let { repository.findById(it) }
                    ?.let { ResultState.Ready(it) } ?: ResultState.NotFound
            } catch (e: CancellationException) { throw e
            } catch (_: Exception) { _state.value = ResultState.Error("Could not read local history.") }
        }
    }
}
