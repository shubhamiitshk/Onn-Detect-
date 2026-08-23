package com.onndetect.ui.screens.home

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.inference.ModelIntegrityChecker
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class HomeViewModel @Inject constructor(
    @ApplicationContext context: Context,
) : ViewModel() {

    /** True when every shipped model matches its export-time SHA-256. */
    private val _modelsVerified = MutableStateFlow<Boolean?>(null)
    val modelsVerified: StateFlow<Boolean?> = _modelsVerified.asStateFlow()

    init {
        viewModelScope.launch(Dispatchers.IO) {
            _modelsVerified.value = ModelIntegrityChecker.verifyAllModels(context)
        }
    }
}
