package com.onndetect.ui.screens.scanner

import android.content.Context
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.data.repository.ScanHistoryRepository
import com.onndetect.detection.audio.AudioClassifier
import com.onndetect.detection.video.VideoForensicAnalyzer
import com.onndetect.domain.models.QualityInfo
import com.onndetect.domain.models.ScanMode
import com.onndetect.domain.models.ScanResult
import com.onndetect.domain.models.SignalBreakdown
import com.onndetect.domain.pipeline.DetectionPipeline
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import javax.inject.Inject

@HiltViewModel
class ScannerViewModel @Inject constructor(
    private val pipeline: DetectionPipeline,
    private val audioClassifier: AudioClassifier,
    private val videoAnalyzer: VideoForensicAnalyzer,
    private val historyRepository: ScanHistoryRepository,
    @ApplicationContext private val appContext: Context,
) : ViewModel() {

    data class UiState(
        val scanning: Boolean = false,
        val stage: String = "",
        val percent: Float = 0f,
        val indeterminate: Boolean = true,
        val completedScanId: String? = null,
        val error: String? = null,
    )

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    fun scanImage(uri: Uri) {
        if (_state.value.scanning) return
        _state.update { it.copy(scanning = true, error = null, completedScanId = null) }

        viewModelScope.launch {
            try {
                val bitmap = withContext(Dispatchers.IO) { decodeScaled(uri) }
                    ?: run {
                        _state.update {
                            it.copy(
                                scanning = false,
                                error = "Could not decode image ($uri)",
                            )
                        }
                        return@launch
                    }

                pipeline.analyzeMedia(bitmap, ScanMode.IMAGE_ONLY).collect { progress ->
                    val result = progress.result
                    if (result != null) {
                        historyRepository.record(result)
                        _state.update {
                            it.copy(
                                scanning = false,
                                percent = 100f,
                                completedScanId = result.id,
                            )
                        }
                    } else {
                        _state.update {
                            it.copy(
                                stage = progress.stage,
                                percent = progress.percent,
                                indeterminate = progress.indeterminate,
                            )
                        }
                    }
                }
            } catch (t: Throwable) {
                _state.update { it.copy(scanning = false, error = t.message ?: "Scan failed") }
            }
        }
    }

    fun consumeCompletion() = _state.update { it.copy(completedScanId = null) }

    /** Video path: frame sampling -> per-frame forensics -> meta fusion
     *  (+ embedded-audio LCNN when present). */
    fun scanVideo(uri: Uri) {
        if (_state.value.scanning) return
        _state.update { it.copy(scanning = true, error = null, completedScanId = null) }

        viewModelScope.launch {
            try {
                val outcome = withContext(Dispatchers.Default) {
                    videoAnalyzer.analyze(uri) { stage ->
                        _state.update { it.copy(stage = stage, indeterminate = true) }
                    }
                }

                val result = ScanResult(
                    id = java.util.UUID.randomUUID().toString(),
                    scanMode = ScanMode.VIDEO_ONLY,
                    timestamp = System.currentTimeMillis(),
                    overallScore = outcome.overallScore,
                    isAiGenerated = outcome.isAiGenerated,
                    confidence = outcome.confidence,
                    signals = SignalBreakdown(
                        mobileClipScore = -1f,
                        srmForensicScore = outcome.srmScore,
                        fftSpectralScore = outcome.fftScore,
                        klDivergenceScore = outcome.klScore,
                        perturbationScore = -1f,
                        audioDspScore = -1f,
                        audioModelScore = outcome.audioScore,
                        temporalScore = outcome.temporalScore,
                        provenanceScore = outcome.overallScore,
                    ),
                    heatmap = null,
                    provenance = null,
                    qualityInfo = QualityInfo(
                        resolution = minOf(outcome.width, outcome.height),
                        isCompressed = true,
                        jpegQuality = null,
                        isBlurry = false,
                        isDark = false,
                    ),
                    scanDurationMs = 0L,
                )
                historyRepository.record(result)
                _state.update {
                    it.copy(scanning = false, percent = 100f, completedScanId = result.id)
                }
            } catch (t: Throwable) {
                _state.update {
                    it.copy(scanning = false, error = t.message ?: "Video scan failed")
                }
            }
        }
    }

    /** Audio path: decode -> LFCC -> LCNN tflite -> EER threshold verdict. */
    fun scanAudio(uri: Uri) {
        if (_state.value.scanning) return
        _state.update { it.copy(scanning = true, error = null, completedScanId = null) }

        viewModelScope.launch {
            try {
                val verdict = withContext(Dispatchers.Default) {
                    audioClassifier.analyze(uri) { stage ->
                        _state.update { it.copy(stage = stage, indeterminate = true) }
                    }
                }

                val started = android.os.SystemClock.elapsedRealtime()
                val result = ScanResult(
                    id = java.util.UUID.randomUUID().toString(),
                    scanMode = ScanMode.AUDIO_ONLY,
                    timestamp = System.currentTimeMillis(),
                    overallScore = verdict.score,
                    isAiGenerated = verdict.isAiGenerated,
                    confidence = kotlin.math.abs(verdict.score - 0.5f) * 2f,
                    signals = SignalBreakdown(
                        mobileClipScore = -1f,
                        srmForensicScore = -1f,
                        fftSpectralScore = -1f,
                        klDivergenceScore = -1f,
                        perturbationScore = -1f,
                        audioDspScore = -1f,
                        audioModelScore = verdict.score,
                        temporalScore = 0f,
                        provenanceScore = verdict.score,
                    ),
                    heatmap = null,
                    provenance = null,
                    qualityInfo = com.onndetect.domain.models.QualityInfo(0, false, null, false, false),
                    scanDurationMs = android.os.SystemClock.elapsedRealtime() - started,
                )
                historyRepository.record(result)
                _state.update {
                    it.copy(scanning = false, percent = 100f, completedScanId = result.id)
                }
            } catch (t: Throwable) {
                _state.update {
                    it.copy(scanning = false, error = t.message ?: "Audio scan failed")
                }
            }
        }
    }

    /** Decodes at most ~1024px on the long edge to bound memory. */
    private fun decodeScaled(uri: Uri): android.graphics.Bitmap? {
        val resolver = appContext.contentResolver
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        // NOTE: decodeStream returns null in bounds-only mode by design; the
        // pass succeeds iff it fills outWidth/outHeight.
        resolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, bounds)
        }
        android.util.Log.d(
            "ScannerVM",
            "decode uri=$uri ${bounds.outWidth}x${bounds.outHeight} mime=${bounds.outMimeType}",
        )
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > 1024) sample *= 2

        val opts = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = android.graphics.Bitmap.Config.ARGB_8888
        }
        return resolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, opts)
        }
    }
}
