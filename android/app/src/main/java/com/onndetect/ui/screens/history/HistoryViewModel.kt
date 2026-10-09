package com.onndetect.ui.screens.history

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.data.repository.ScanHistoryRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class HistoryViewModel @Inject constructor(private val repository: ScanHistoryRepository) : ViewModel() {
    val error = MutableStateFlow<String?>(null)
    val entries = repository.observeAll().catch { error.value = "Could not read local history."; emit(emptyList()) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList<ScanHistoryEntity>())
    fun delete(id: String) = mutate { repository.delete(id) }
    fun clearAll() = mutate { repository.clear() }
    private fun mutate(action: suspend () -> Unit) {
        viewModelScope.launch {
            try { action(); error.value = null
            } catch (e: CancellationException) { throw e
            } catch (_: Exception) { error.value = "Could not update local history. Try again." }
        }
    }
}
