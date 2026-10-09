package com.onndetect.domain.pipeline

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import com.onndetect.detection.fusion.XGBoostMetaClassifier
import com.onndetect.detection.visual.FFTSpectralAnalyzer
import com.onndetect.detection.visual.KLCooccurrenceAnalyzer
import com.onndetect.detection.visual.SRMForensicAnalyzer
import com.onndetect.inference.ModelIntegrityChecker
import com.onndetect.inference.ModelIntegrityGate
import com.onndetect.domain.models.*
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.util.UUID
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.max
import kotlin.math.min

enum class DegradationLevel { FULL, PARTIAL_VISUAL, PARTIAL_AUDIO, MATH_ONLY, ABSTAIN }

/** Image diagnostics only become verdicts for a validated detector release. */
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
    private val prnuAnalyzer: com.onndetect.detection.visual.PRNUAnalyzer,
    @param:ApplicationContext private val appContext: Context,
) {
    companion object {
        const val NEUTRAL_LIP_SYNC = 0.9f
        const val NEUTRAL_JITTER = 0.1f
        const val NEUTRAL_EXIF = 0.1f
        const val NEUTRAL_C2PA = 0.05f
    }
    @Suppress("UNUSED_PARAMETER")
    suspend fun analyzeMedia(bitmap: Bitmap, mode: ScanMode, provenance: ProvenanceInfo? = null,
        sourceUri: android.net.Uri? = null, isCompressed: Boolean = false, originalMinEdge: Int? = null): Flow<ScanProgress> = flow {
        val started = SystemClock.elapsedRealtime()
        val jobContext = currentCoroutineContext()
        val checkStop = {
            jobContext.ensureActive()
            check(SystemClock.elapsedRealtime() - started <= 30_000) { "Image analysis deadline exceeded" }
        }
        var signals = SignalBreakdown(-1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f)
        var quality = QualityInfo(originalMinEdge ?: 0, isCompressed, null, false, false)
        fun result(status: AnalysisStatus, reason: String?, score: Float = -1f,
            report: ModelIntegrityChecker.Report? = null): ScanResult {
            val verdict = VerdictPolicy.decide(score, status, report?.validated == true, report?.threshold)
            return ScanResult(UUID.randomUUID().toString(), mode, System.currentTimeMillis(), score,
                verdict == DetectionVerdict.SYNTHETIC, 0f, signals, null, provenance, quality,
                SystemClock.elapsedRealtime() - started, status, verdict, reason, report?.validated == true,
                report?.releaseId ?: "unknown", report?.frontendId ?: "unknown",
                if (report?.validated == true) report.releaseId else "unvalidated", report?.threshold)
        }
        try {
            emit(ScanProgress("Validating input...", 5f))
            if (mode != ScanMode.IMAGE_ONLY) {
                emit(ScanProgress("Unsupported image scan mode", 100f, result = result(AnalysisStatus.UNSUPPORTED_INPUT, "Use the audio or video analysis route")))
                return@flow
            }
            if (bitmap.isRecycled || min(bitmap.width, bitmap.height) < 16) {
                emit(ScanProgress("Unsupported image", 100f, result = result(AnalysisStatus.UNSUPPORTED_INPUT, "Image is too small or unavailable")))
                return@flow
            }
            quality = qualityInfo(bitmap, originalMinEdge).copy(isCompressed = isCompressed)
            val report = ModelIntegrityGate.report(appContext, setOf(ModelIntegrityChecker.IMAGE_MODEL))
            if (!report.allOk) {
                emit(ScanProgress("Model unavailable", 100f, result = result(AnalysisStatus.MODEL_FAILURE, report.failures.joinToString())))
                return@flow
            }
            val budget = checkHardwareConstraints()
            if (budget == DegradationLevel.ABSTAIN) {
                emit(ScanProgress("Insufficient memory", 100f, result = result(AnalysisStatus.INCONCLUSIVE, "Not enough memory for analysis")))
                return@flow
            }
            checkStop()
            var srm = -1f
            if (budget != DegradationLevel.MATH_ONLY) {
                emit(ScanProgress("Image model diagnostics...", 20f))
                srm = srmAnalyzer.analyze(bitmap, checkStop)
                signals = signals.copy(srmForensicScore = srm)
            }
            emit(ScanProgress("Spectral descriptors...", 45f))
            val spectral = fftAnalyzer.analyze(bitmap, checkStop)
            signals = signals.copy(fftSpectralScore = spectral.score)
            checkStop()
            emit(ScanProgress("Pixel difference descriptors...", 65f))
            val kl = klAnalyzer.analyze(bitmap)
            signals = signals.copy(klDivergenceScore = kl)
            checkStop()
            if (budget == DegradationLevel.FULL) {
                signals = signals.copy(prnuScore = prnuAnalyzer.analyze(bitmap))
                checkStop()
            }
            // Unverified EXIF and provenance never enter verdict fusion.
            // Legacy trees were trained on incompatible metadata; they are
            // not even exposed as an overall probability.
            if (!report.validated || srm < 0f) {
                val reason = if (srm < 0f) "Memory budget allowed diagnostic math only; no validated fallback" else report.validationReason
                emit(ScanProgress("Analysis inconclusive", 100f, result = result(AnalysisStatus.INCONCLUSIVE, reason, report = report)))
                return@flow
            }
            val features = floatArrayOf(srm, spectral.score, kl, spectral.spectralFlatness,
                (quality.resolution / 1024f).coerceIn(0f, 1f), if (isCompressed) 1f else 0f,
                NEUTRAL_EXIF, NEUTRAL_C2PA, NEUTRAL_LIP_SYNC, NEUTRAL_JITTER)
            val score = signalFusion.fuse(metaClassifier.predict(features), provenance)
            emit(ScanProgress("Analysis complete", 100f, result = result(AnalysisStatus.SUCCESS, null, score, report)))
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) {
            emit(ScanProgress("Analysis inconclusive", 100f, result = result(AnalysisStatus.MODEL_FAILURE, e.message ?: "Analysis failed")))
        }
    }.flowOn(Dispatchers.Default)

    private fun qualityInfo(bitmap: Bitmap, originalMinEdge: Int? = null): QualityInfo {
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
            // True source edge when the caller decoded bounds first; the
            // working bitmap is long-edge-capped and would flatten this.
            resolution = originalMinEdge ?: min(w, h),
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
        // memoryClass is the app's REAL per-process heap budget. availMem is
        // system-wide free memory, which other apps can bleed to near zero and
        // make this gate fire unpredictably - it is not our budget.
        val heapBudgetMb = am.memoryClass
        return when {
            heapBudgetMb < 64 || availMb < 64 -> DegradationLevel.ABSTAIN
            heapBudgetMb <= 128 || availMb < 150 -> DegradationLevel.MATH_ONLY
            else -> DegradationLevel.FULL
        }
    }
}
