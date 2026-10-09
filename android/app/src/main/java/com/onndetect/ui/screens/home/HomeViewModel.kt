package com.onndetect.ui.screens.home

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.inference.ModelIntegrityChecker
import com.onndetect.inference.ModelIntegrityGate
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class HomeViewModel @Inject constructor(@ApplicationContext context: Context) : ViewModel() {
    data class Readiness(val image: Boolean? = null, val audio: Boolean? = null)
    private val _readiness = MutableStateFlow(Readiness())
    val readiness = _readiness.asStateFlow()
    init {
        viewModelScope.launch(Dispatchers.IO) {
            _readiness.value = Readiness(
                ModelIntegrityGate.report(context,setOf(ModelIntegrityChecker.IMAGE_MODEL)).allOk,
                ModelIntegrityGate.report(context,setOf(ModelIntegrityChecker.AUDIO_MODEL)).allOk)
        }
    }
}
