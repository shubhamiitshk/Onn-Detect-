package com.onndetect.ui.screens.scanner

import android.content.Context
import android.graphics.*
import android.net.Uri
import android.os.SystemClock
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.R
import com.onndetect.data.repository.ScanHistoryRepository
import com.onndetect.detection.audio.AudioClassifier
import com.onndetect.detection.video.VideoForensicAnalyzer
import com.onndetect.domain.models.*
import com.onndetect.domain.pipeline.DetectionPipeline
import com.onndetect.inference.ModelIntegrityGate
import com.onndetect.inference.ModelIntegrityChecker
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import java.io.File
import java.util.UUID
import javax.inject.Inject

@HiltViewModel
class ScannerViewModel @Inject constructor(
    private val pipeline: DetectionPipeline,
    private val audioClassifier: AudioClassifier,
    private val videoAnalyzer: VideoForensicAnalyzer,
    private val historyRepository: ScanHistoryRepository,
    @param:ApplicationContext private val appContext: Context,
    private val savedStateHandle: SavedStateHandle,
) : ViewModel() {
    data class UiState(
        val scanning: Boolean = false, val stage: String = "", val percent: Float = 0f,
        val indeterminate: Boolean = true, val completedScanId: String? = null,
        val error: String? = null, val unsavedResult: ScanResult? = null,
    )
    private val _state = MutableStateFlow(UiState(error = if (savedStateHandle.contains("scanStarted")) message(R.string.scan_interrupted) else null))
    val state = _state.asStateFlow()
    private var scanJob: Job? = null
    private var requested = savedStateHandle.get<Boolean>("scanStarted") == true
    private val importer = ProviderMediaImporter(appContext)
    private fun message(id: Int) = appContext.getString(id)

    fun start(uri: Uri, mode: String?, retry: Boolean = false) {
        if (scanJob?.isActive == true || (requested && !retry)) return
        requested = true
        savedStateHandle["scanStarted"] = true
        _state.value = UiState(scanning = true, stage = message(R.string.preparing))
        scanJob = viewModelScope.launch {
            try {
                require(uri.scheme == "content") { message(R.string.invalid_input) }
                withTimeout(120_000) {
                    runInterruptible(Dispatchers.IO) {
                        val required = if (mode == "AUDIO_ONLY") ModelIntegrityChecker.AUDIO_MODEL else ModelIntegrityChecker.IMAGE_MODEL
                        check(ModelIntegrityGate.report(appContext, setOf(required)).allOk) { message(R.string.assets_failed) }
                    }
                    val maxBytes = when (mode) {
                        "IMAGE_ONLY" -> 64L * 1024 * 1024
                        "AUDIO_ONLY" -> 128L * 1024 * 1024
                        "VIDEO_ONLY" -> 512L * 1024 * 1024
                        else -> throw IllegalArgumentException(message(R.string.unsupported_mode))
                    }
                    val imported = importer.import(uri, maxBytes)
                    try {
                        when (mode) {
                            "IMAGE_ONLY" -> scanImage(imported)
                            "AUDIO_ONLY" -> scanAudio(Uri.fromFile(imported))
                            "VIDEO_ONLY" -> scanVideo(Uri.fromFile(imported))
                        }
                    } finally { imported.delete() }
                }
            } catch (_: TimeoutCancellationException) {
                _state.update { it.copy(scanning = false, stage = "", error = message(R.string.scan_timeout)) }
            } catch (e: CancellationException) {
                _state.update { it.copy(scanning = false, stage = "", error = message(R.string.scan_cancelled)) }
                throw e
            } catch (e: MediaImportException) {
                _state.update { it.copy(scanning = false, stage = "", error = message(e.messageResource)) }
            } catch (_: OutOfMemoryError) {
                _state.update { it.copy(scanning = false, stage = "", error = message(R.string.scan_memory_failed)) }
            } catch (_: Exception) {
                _state.update { it.copy(scanning = false, stage = "", error = message(R.string.scan_failed)) }
            }
        }
    }
    fun cancel() { scanJob?.cancel() }
    fun consumeCompletion() = _state.update { it.copy(completedScanId = null) }
    fun retrySave() {
        val result = _state.value.unsavedResult ?: return
        if (scanJob?.isActive == true) return
        scanJob = viewModelScope.launch { finish(result) }
    }
    private suspend fun finish(result: ScanResult) {
        currentCoroutineContext().ensureActive()
        // Preserve the computed assessment if storage fails. Never replace it with an authenticity verdict.
        try {
            historyRepository.record(result)
            _state.value = UiState(percent = 100f, completedScanId = result.id)
        } catch (e: CancellationException) { throw e
        } catch (_: Exception) {
            _state.value = UiState(error = message(R.string.save_failed), unsavedResult = result)
        }
    }
    private suspend fun scanImage(file: File) {
        val decoded = decodeScaled(file)
        try {
            pipeline.analyzeMedia(decoded.bitmap, ScanMode.IMAGE_ONLY, sourceUri = Uri.fromFile(file),
                isCompressed = decoded.mime == "image/jpeg", originalMinEdge = decoded.originalMinEdge).collect { progress ->
                currentCoroutineContext().ensureActive()
                progress.result?.let { finish(it) } ?: run {
                    if (progress.percent >= 100f && !progress.indeterminate) {
                        _state.update { it.copy(scanning = false, error = progress.stage, stage = "") }
                    } else _state.update { it.copy(stage = progress.stage, percent = progress.percent, indeterminate = progress.indeterminate) }
                }
            }
        } finally { decoded.bitmap.recycle() }
    }
    private suspend fun scanAudio(uri: Uri) {
        val started = SystemClock.elapsedRealtime()
        val v = runInterruptible(Dispatchers.Default) { audioClassifier.analyze(uri) { stage -> _state.update { it.copy(stage = stage) } } }
        finish(ScanResult(UUID.randomUUID().toString(), ScanMode.AUDIO_ONLY, System.currentTimeMillis(),
            v.score, v.isAiGenerated, 0f, SignalBreakdown(-1f,-1f,-1f,-1f,-1f,-1f,v.score,-1f,-1f),
            null, null, QualityInfo(0,false,null,false,false), SystemClock.elapsedRealtime() - started,
            status = v.status, verdict = if (v.modelValidated && v.status == AnalysisStatus.SUCCESS) {
                if (v.isAiGenerated) DetectionVerdict.SYNTHETIC else DetectionVerdict.AUTHENTIC
            } else DetectionVerdict.INCONCLUSIVE, reason = v.reason, modelValidated = v.modelValidated,
            decisionThreshold = v.threshold, analyzedDurationMs = v.analyzedDurationMs, totalDurationMs = v.totalDurationMs,
            analyzedTimestampsMs = v.analyzedTimestampsMs, modelVersion = v.modelVersion,
            preprocessingVersion = v.preprocessingVersion, calibrationVersion = v.calibrationVersion))
    }
    private suspend fun scanVideo(uri: Uri) {
        val started = SystemClock.elapsedRealtime()
        val o = runInterruptible(Dispatchers.Default) { videoAnalyzer.analyze(uri) { stage -> _state.update { it.copy(stage = stage) } } }
        val verdict = VerdictPolicy.decide(o.overallScore, o.status, o.modelValidated, o.decisionThreshold)
        finish(ScanResult(UUID.randomUUID().toString(), ScanMode.VIDEO_ONLY, System.currentTimeMillis(), o.overallScore,
            verdict == DetectionVerdict.SYNTHETIC, 0f, SignalBreakdown(o.semanticScore,o.srmScore,o.fftScore,o.klScore,-1f,-1f,o.audioScore,o.temporalScore,-1f),
            null, null, QualityInfo(minOf(o.width,o.height),true,null,false,false), SystemClock.elapsedRealtime() - started,
            status = o.status, verdict = verdict, reason = o.reason, modelValidated = o.modelValidated,
            decisionThreshold = o.decisionThreshold, modelVersion = o.modelVersion,
            preprocessingVersion = o.preprocessingVersion, calibrationVersion = o.calibrationVersion,
            totalDurationMs = o.durationMs, analyzedDurationMs = o.audioAnalyzedDurationMs, analyzedTimestampsMs = o.timestampsMs))
    }
    private data class DecodedImage(val bitmap: Bitmap, val originalMinEdge: Int, val mime: String?)
    private suspend fun decodeScaled(file: File): DecodedImage = cancellableProviderWork(dispose = { it: DecodedImage -> it.bitmap.recycle() }) {
        var owned: Bitmap? = null
        try {
            checkActive()
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(file.absolutePath, bounds)
            require(bounds.outWidth > 0 && bounds.outHeight > 0 && bounds.outWidth.toLong() * bounds.outHeight <= 100_000_000L) { message(R.string.invalid_input) }
            require(bounds.outMimeType in setOf("image/jpeg", "image/png", "image/webp", "image/heif", "image/avif")) { message(R.string.invalid_input) }
            var sample = 1
            while (maxOf(bounds.outWidth, bounds.outHeight) / sample > 1024) sample *= 2
            val options = BitmapFactory.Options().apply {
                inSampleSize = sample; inPreferredConfig = Bitmap.Config.ARGB_8888
                inPreferredColorSpace = ColorSpace.get(ColorSpace.Named.SRGB)
            }
            val decoded = BitmapFactory.decodeFile(file.absolutePath, options) ?: throw IllegalArgumentException(message(R.string.invalid_input))
            owned = decoded
            val orientation = runCatching { ExifInterface(file).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }.getOrDefault(1)
            val matrix = Matrix().apply {
                when (orientation) {
                    2 -> setScale(-1f,1f); 3 -> setRotate(180f); 4 -> setScale(1f,-1f)
                    5 -> { setRotate(90f); postScale(-1f,1f) }; 6 -> setRotate(90f)
                    7 -> { setRotate(270f); postScale(-1f,1f) }; 8 -> setRotate(270f)
                }
            }
            val oriented = Bitmap.createBitmap(decoded,0,0,decoded.width,decoded.height,matrix,true)
            if (oriented !== decoded) { decoded.recycle(); owned = oriented }
            val rgb = Bitmap.createBitmap(oriented.width,oriented.height,Bitmap.Config.ARGB_8888)
            Canvas(rgb).apply { drawColor(Color.WHITE); drawBitmap(oriented,0f,0f,null) }
            oriented.recycle(); owned = rgb
            checkActive()
            DecodedImage(rgb,minOf(bounds.outWidth,bounds.outHeight),bounds.outMimeType).also { owned = null }
        } finally { owned?.recycle() }
    }
}
