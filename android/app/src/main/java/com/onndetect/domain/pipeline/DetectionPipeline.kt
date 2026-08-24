package com.onndetect.domain.pipeline

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import com.onndetect.detection.fusion.XGBoostMetaClassifier
import com.onndetect.detection.visual.CalibrationStore
import com.onndetect.detection.visual.FFTSpectralAnalyzer
import com.onndetect.detection.visual.KLCooccurrenceAnalyzer
import com.onndetect.detection.visual.SRMForensicAnalyzer
import com.onndetect.domain.models.DeviceTier
import com.onndetect.domain.models.ProvenanceInfo
import com.onndetect.domain.models.QualityInfo
import com.onndetect.domain.models.ScanMode
import com.onndetect.domain.models.ScanProgress
import com.onndetect.domain.models.ScanResult
import com.onndetect.domain.models.SignalBreakdown
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.max
import kotlin.math.min

enum class DegradationLevel {
    FULL, PARTIAL_VISUAL, PARTIAL_AUDIO, MATH_ONLY, ABSTAIN
}

/**
 * End-to-end image scan: quality gate -> tier-aware parallel analyzers ->
 * XGBoost meta-fusion (transpiled trees) -> provenance override.
 *
 * Feature contract with the exported classifier (order matters!):
 *   0 srm_score, 1 fft_score, 2 kl_score, 3 flatness_score,
 *   4 resolution_norm, 5 is_compressed
 */
@Singleton
class DetectionPipeline @Inject constructor(
    private val deviceTier: DeviceTier,
    private val signalFusion: SignalFusion,
    private val srmAnalyzer: SRMForensicAnalyzer,
    private val fftAnalyzer: FFTSpectralAnalyzer,
    private val klAnalyzer: KLCooccurrenceAnalyzer,
    private val metaClassifier: XGBoostMetaClassifier,
    private val exifAnalyzer: com.onndetect.detection.provenance.EXIFAnalyzer,
    private val c2paChecker: com.onndetect.detection.provenance.C2PAManifestChecker,
    private val lipSync: com.onndetect.detection.crossmodal.LipSyncCorrelation,
    private val jitterDetector: com.onndetect.detection.temporal.LandmarkJitterDetector,
    private val prnuAnalyzer: com.onndetect.detection.visual.PRNUAnalyzer,
    private val opticalFlow: com.onndetect.detection.video.OpticalFlowAnalyzer,
    private val speakerVerifier: com.onndetect.detection.audio.SpeakerVerifier,
    @ApplicationContext private val appContext: Context,
) {

    suspend fun analyzeMedia(
        bitmap: Bitmap,
        mode: ScanMode,
        provenance: ProvenanceInfo? = null,
    ): Flow<ScanProgress> = flow {
        val started = SystemClock.elapsedRealtime()
        emit(ScanProgress("Input validation…", 5f))

        if (mode != ScanMode.IMAGE_ONLY && mode != ScanMode.VIDEO_WITH_AUDIO &&
            mode != ScanMode.CROSS_MODAL
        ) {
            // Audio-only path is handled by AnalyzeAudioUseCase; abstain here.
            emit(aborted(mode))
            return@flow
        }

        when (checkHardwareConstraints()) {
            DegradationLevel.ABSTAIN -> { emit(aborted(mode)); return@flow }
            else -> Unit
        }

        emit(ScanProgress("SRM forensic noise analysis…", 20f))
        val srmScore = srmAnalyzer.analyze(bitmap)

        val spectral: com.onndetect.detection.visual.SpectralResult
        val klScore: Float
        if (deviceTier == DeviceTier.HIGH_END) {
            val (s, k) = coroutineScope {
                val fftDef = async { fftAnalyzer.analyze(bitmap) }
                val klDef = async { klAnalyzer.analyze(bitmap) }
                emit(ScanProgress("Spectral + co-occurrence forensics…", 50f))
                fftDef.await() to klDef.await()
            }
            spectral = s
            klScore = k
        } else {
            emit(ScanProgress("Spectral analysis…", 40f))
            spectral = fftAnalyzer.analyze(bitmap)
            emit(ScanProgress("Co-occurrence statistics…", 60f))
            klScore = klAnalyzer.analyze(bitmap)
        }

        emit(ScanProgress("Fusing signals (XGBoost meta-classifier)…", 80f))

        val quality = qualityInfo(bitmap)
        val features = floatArrayOf(
            srmScore,
            spectral.score,
            klScore,
            spectral.spectralFlatness,
            quality.resolution / 1024f,
            if (quality.isCompressed) 1f else 0f,
            // New Offline features - passing safe defaults for Image scan mode
            0.1f, // exif_anomaly
            0.05f, // c2pa_present 
            0.9f, // lip_sync_corr
            0.1f, // landmark_jitter
        )
        val rawScore = metaClassifier.predict(features).coerceIn(0f, 1f)
        
        // --- ADVANCED OFFLINE FORENSICS (Phases 6-8) ---
        emit(ScanProgress("Extracting PRNU Sensor Noise...", 85f))
        val prnuAnomaly = prnuAnalyzer.analyze(bitmap)
        
        // Simulating Optical Flow & Speaker Liveness (since we only have an Image here)
        val opticalFlowAnomaly = 0.5f // Neutral for image
        val voiceAnomaly = 0.5f // Neutral for image
        
        // Hard-override final score logic with Advanced Signals
        val enhancedScore = ((rawScore * 0.6f) + (prnuAnomaly * 0.4f)).coerceIn(0f, 1f)
        
        val finalScore = signalFusion.fuse(enhancedScore, provenance)

        val threshold = CalibrationStore.metaDecisionThreshold(appContext)
        val signals = SignalBreakdown(
            mobileClipScore = -1f,
            srmForensicScore = srmScore,
            fftSpectralScore = spectral.score,
            klDivergenceScore = klScore,
            perturbationScore = -1f,
            audioDspScore = 0f,
            audioModelScore = 0f,
            temporalScore = 0f,
            provenanceScore = finalScore,
        )

        val result = ScanResult(
            id = UUID.randomUUID().toString(),
            scanMode = mode,
            timestamp = System.currentTimeMillis(),
            overallScore = finalScore,
            isAiGenerated = finalScore >= threshold,
            confidence = confidenceOf(finalScore),
            signals = signals,
            heatmap = null,
            provenance = provenance,
            qualityInfo = quality,
            scanDurationMs = SystemClock.elapsedRealtime() - started,
        )
        emit(ScanProgress("Scan complete", 100f, indeterminate = false, result = result))
    }.flowOn(Dispatchers.Default)

    private fun aborted(mode: ScanMode) = ScanProgress(
        stage = "Scan aborted: insufficient memory for analysis",
        percent = 100f,
        indeterminate = false,
        result = ScanResult(
            id = UUID.randomUUID().toString(),
            scanMode = mode,
            timestamp = System.currentTimeMillis(),
            overallScore = 0.5f,
            isAiGenerated = false,
            confidence = 0f,
            signals = SignalBreakdown(-1f, -1f, -1f, -1f, -1f, 0f, 0f, 0f, 0.5f),
            heatmap = null,
            provenance = null,
            qualityInfo = QualityInfo(0, false, null, false, true),
            scanDurationMs = 0L,
        )
    )

    /** Maps score distance from the neutral point onto [0,1]. */
    private fun confidenceOf(score: Float): Float =
        kotlin.math.abs(score - 0.5f) * 2f

    private fun qualityInfo(bitmap: Bitmap): QualityInfo {
        val w = bitmap.width
        val h = bitmap.height
        // Local name must not be `step` - it collides with the range infix.
        val margin = max(1, min(w, h) / 64)
        val stride = margin * 2
        var sumLuma = 0.0
        var samples = 0
        var lapSum = 0.0
        var lapSamples = 0
        for (y in margin until h - margin step stride) {
            for (x in margin until w - margin step stride) {
                val c = bitmap.getPixel(x, y)
                val luma = 0.299 * (c shr 16 and 0xFF) +
                    0.587 * (c shr 8 and 0xFF) + 0.114 * (c and 0xFF)
                sumLuma += luma
                samples++
                if (samples % 7 == 0) {
                    // 4-neighbor Laplacian response (blur detector)
                    val right = bitmap.getPixel(min(x + 1, w - 1), y)
                    val left = bitmap.getPixel(max(x - 1, 0), y)
                    val down = bitmap.getPixel(x, min(y + 1, h - 1))
                    val up = bitmap.getPixel(x, max(y - 1, 0))
                    val lap = 4.0 * luma -
                        lumaOf(right) - lumaOf(left) - lumaOf(down) - lumaOf(up)
                    lapSum += lap * lap
                    lapSamples++
                }
            }
        }
        val meanLuma = if (samples > 0) sumLuma / samples else 128.0
        val lapVar = if (lapSamples > 0) lapSum / lapSamples else 100.0
        return QualityInfo(
            resolution = min(w, h),
            isCompressed = false,
            jpegQuality = null,
            isBlurry = lapVar < 25.0,
            isDark = meanLuma < 40.0,
        )
    }

    private fun lumaOf(c: Int) = 0.299 * (c shr 16 and 0xFF) +
        0.587 * (c shr 8 and 0xFF) + 0.114 * (c and 0xFF)

    private fun checkHardwareConstraints(): DegradationLevel {
        val am = appContext.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo().also(am::getMemoryInfo)
        val availMb = memInfo.availMem / (1024 * 1024)
        return when {
            availMb < 250 -> DegradationLevel.ABSTAIN
            availMb < 600 -> DegradationLevel.MATH_ONLY
            else -> DegradationLevel.FULL
        }
    }
}
