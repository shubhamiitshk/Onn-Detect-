package com.onndetect.ui.screens.result

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.data.repository.ScanHistoryRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class ResultViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val repository: ScanHistoryRepository,
) : ViewModel() {

    private val scanId: String? = savedStateHandle["scanId"]

    private val _entry = MutableStateFlow<ScanHistoryEntity?>(null)
    val entry: StateFlow<ScanHistoryEntity?> = _entry.asStateFlow()

    init {
        scanId?.let { id ->
            viewModelScope.launch { _entry.value = repository.findById(id) }
        }
    }
}
