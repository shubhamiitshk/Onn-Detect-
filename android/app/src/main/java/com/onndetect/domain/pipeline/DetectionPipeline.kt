package com.onndetect.domain.pipeline

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import com.onndetect.detection.fusion.XGBoostMetaClassifier
import com.onndetect.detection.provenance.ProvenanceEvidence
import com.onndetect.detection.provenance.VerdictConfidence
import com.onndetect.detection.visual.FFTSpectralAnalyzer
import com.onndetect.detection.visual.KLCooccurrenceAnalyzer
import com.onndetect.detection.visual.SRMForensicAnalyzer
import com.onndetect.detection.visual.VfmProbeAnalyzer
import com.onndetect.detection.visual.CalibrationStore
import com.onndetect.detection.visual.ResidualHeatmapExtractor
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
    private val vfmProbe: VfmProbeAnalyzer,
    @param:ApplicationContext private val appContext: Context,
) {
    private val c2paVerifier: com.onndetect.detection.provenance.C2PAVerifier =
        com.onndetect.detection.provenance.C2PAVerifier(c2paChecker)
    private val documentAnalyzer = com.onndetect.detection.document.DocumentForgeryAnalyzer()
    companion object {
        const val NEUTRAL_LIP_SYNC = 0.9f
        const val NEUTRAL_JITTER = 0.1f
        const val NEUTRAL_C2PA = 0.05f

        /**
         * Legacy neutral for the former slot-6 constant.
         *
         * Slot 6 now carries the frozen SigLIP-B/16 tower's measurement and is
         * UNAVAILABLE (NaN) when the probe did not run. A neutral constant is
         * only safe while nothing is ever measured in the slot, because a
         * tree that splits on 0.1 cannot tell "low anomaly" from "not run".
         */
        const val NEUTRAL_EXIF = 0.1f

        /**
         * Fusion slot carrying the frozen SigLIP-B/16 tower's measurement.
         *
         * MUST stay identical to `meta_classifier.feature_extractor`
         * `SEMANTIC_PROBE_INDEX` and to the slot documented in
         * `XGBoostMetaClassifier.kt`.
         */
        const val SEMANTIC_PROBE_INDEX = 6
    }

    @Suppress("UNUSED_PARAMETER")
    suspend fun analyzeMedia(bitmap: Bitmap, mode: ScanMode, provenance: ProvenanceInfo? = null,
        sourceUri: android.net.Uri? = null, isCompressed: Boolean = false, originalMinEdge: Int? = null,
        /**
         * The SHA-256 of the media bytes, when the caller has them. It is
         * carried into the result so a forensic receipt can bind itself to
         * the exact content it describes.
         */
        mediaSha256: String? = null): Flow<ScanProgress> = flow {
        val started = SystemClock.elapsedRealtime()
        val jobContext = currentCoroutineContext()
        val checkStop = {
            jobContext.ensureActive()
            check(SystemClock.elapsedRealtime() - started <= 30_000) { "Image analysis deadline exceeded" }
        }
        var signals = SignalBreakdown(-1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f)
        var quality = QualityInfo(originalMinEdge ?: 0, isCompressed, null, false, false)
        var residualHeatmap: FloatArray? = null
        var docReport: com.onndetect.detection.document.DocumentForgeryReport? = null
        // Provenance evidence is collected independently of the statistical
        // detectors and NEVER enters the score. It is categorical: a
        // VERIFIED claim or a significant watermark detection is a
        // different KIND of answer, not a higher probability.
        var provenanceEvidence: ProvenanceEvidence = ProvenanceEvidence.Absent
        fun result(status: AnalysisStatus, reason: String?, score: Float = -1f,
            report: ModelIntegrityChecker.Report? = null,
            decisionThreshold: Float? = report?.threshold,
            mediaSha256: String? = null): ScanResult {
            val verdict = VerdictPolicy.decide(score, status, report?.validated == true, decisionThreshold)
            val confirmed = when (val evidence = provenanceEvidence) {
                is ProvenanceEvidence.C2PAClaim -> evidence.confirmsAiInvolvement
                is ProvenanceEvidence.WatermarkMark -> evidence.confirmsAiInvolvement
                else -> false
            }
            val confidence = when {
                confirmed -> VerdictConfidence.PROVENANCE_CONFIRMED
                report?.validated == true && verdict != DetectionVerdict.INCONCLUSIVE -> VerdictConfidence.STATISTICAL
                else -> VerdictConfidence.UNVALIDATED
            }
            val finalVerdict = if (confirmed && (status == AnalysisStatus.SUCCESS || score in 0f..1f)) {
                DetectionVerdict.SYNTHETIC
            } else verdict
            val finalStatus = if (confirmed && finalVerdict == DetectionVerdict.SYNTHETIC && status != AnalysisStatus.SUCCESS) {
                AnalysisStatus.SUCCESS
            } else status
            val finalThreshold = if (confirmed && finalVerdict == DetectionVerdict.SYNTHETIC && decisionThreshold == null) {
                0.5f
            } else decisionThreshold
            val finalScore = if (confirmed && finalVerdict == DetectionVerdict.SYNTHETIC && (score < 0f || !score.isFinite())) {
                0.99f
            } else score
            val finalModelValidated = if (confirmed && finalVerdict == DetectionVerdict.SYNTHETIC) {
                true
            } else report?.validated == true
            val finalReason = if (confirmed && finalVerdict == DetectionVerdict.SYNTHETIC) {
                when (provenanceEvidence) {
                    is ProvenanceEvidence.C2PAClaim -> "Cryptographic Content Credentials confirmed AI generation"
                    is ProvenanceEvidence.WatermarkMark -> "Neural watermark confirmed AI origin"
                    else -> reason
                }
            } else reason
            return ScanResult(UUID.randomUUID().toString(), mode, System.currentTimeMillis(), finalScore,
                finalVerdict == DetectionVerdict.SYNTHETIC, 0f, signals, residualHeatmap, provenance, quality,
                SystemClock.elapsedRealtime() - started, finalStatus, finalVerdict, finalReason, finalModelValidated,
                report?.releaseId ?: "unknown", report?.frontendId ?: "unknown",
                if (finalModelValidated) (report?.releaseId ?: "c2pa-verified") else "unvalidated", finalThreshold,
                verdictConfidence = confidence, provenanceEvidence = provenanceEvidence,
                mediaSha256 = mediaSha256,
                releaseDeviceValidated = report?.deviceValidated == true,
                documentForgeryReport = docReport)
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
            residualHeatmap = runCatching { ResidualHeatmapExtractor.computeResidualHeatmap(bitmap, 32) }.getOrNull()
            docReport = runCatching { documentAnalyzer.analyze(bitmap) }.getOrNull()
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
            // Provenance is collected BEFORE the statistical detectors and
            // never feeds the score. The C2PA checker only DETECTS a
            // manifest; without the signature verifier it cannot report
            // `signed = true`, so it can never raise a confirmed verdict on
            // its own.
            if (sourceUri != null) {
                val verification = c2paVerifier.verify(appContext, sourceUri)
                if (verification.evidence !is ProvenanceEvidence.Absent) {
                    provenanceEvidence = verification.evidence
                }
            }
            checkStop()
            // The SRM CNN runs at every degradation level except ABSTAIN.
            // It is 143 KB and memory-mapped, so it is not what the memory
            // budget is about; the 89.96 MB tower is. Skipping the SRM CNN
            // made the pipeline refuse to score (srm < 0f), which denied the
            // user a verdict for a reason that had nothing to do with the
            // app's capability.
            var srm = -1f
            if (budget != DegradationLevel.ABSTAIN) {
                emit(ScanProgress("Decomposing 20-ch high-pass residuals…", 20f))
                srm = srmAnalyzer.analyze(bitmap, checkStop)
                signals = signals.copy(srmForensicScore = srm)
            }
            emit(ScanProgress("2D FFT power spectrum & radial slope…", 45f))
            val spectral = fftAnalyzer.analyze(bitmap, checkStop)
            signals = signals.copy(fftSpectralScore = spectral.score)
            checkStop()
            emit(ScanProgress("Pixel difference co-occurrence matrix…", 65f))
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

            // ── Slot 6: the frozen SigLIP-B/16 tower, an optional second opinion ─────
            // The probe runs ONLY when the release manifest declares it AND
            // it cleared every integrity gate (hash, contract, calibration
            // above chance, desktop parity, device parity). An unverified
            // second engine must not be able to influence a verdict.
            //
            // When it does not run, slot 6 stays UNAVAILABLE (NaN) and the
            // fusion tree takes its learned missing branch. A neutral
            // constant would be read as a real low measurement.
            var engineCScore: Float? = null
            if (budget == DegradationLevel.FULL && report.cascadeValidated) {
                checkStop()
                emit(ScanProgress("Evaluating the SigLIP-B/16 vision tower…", 78f))
                engineCScore = vfmProbe.analyze(bitmap, checkStop)
                checkStop()
                val score = engineCScore
                if (score != null && score.isFinite()) {
                    signals = signals.copy(vfmProbeScore = score)
                }
            }

            val features = FusionFeatureSchema.build(
                srm, spectral.score, kl, spectral.spectralFlatness,
                (quality.resolution / 1024f).coerceIn(0f, 1f), if (isCompressed) 1f else 0f,
                engineCScore ?: Float.NaN, NEUTRAL_C2PA, NEUTRAL_LIP_SYNC, NEUTRAL_JITTER
            )
            val fusedScore = signalFusion.fuse(metaClassifier.predict(features), provenance)
            // Persist the FUSION score. The cascade policy receives this
            // value as its "Engine A" input, so the telemetry card must
            // show it rather than the raw SRM CNN output.
            signals = signals.copy(fusionScore = fusedScore)

            // ── Cascade arbitration ───────────────────────────────────────────
            // Engine C is the stronger detector, so it decides when confident.
            // Engine A arbitrates inside the measured ambiguity band, and a
            // disagreement abstains rather than picking a winner.
            //
            // The reported score and its decision threshold must come from the
            // SAME engine: the two have different operating points, and
            // VerdictPolicy compares the score against the threshold. Mixing
            // them would report a score that contradicts its own verdict.
            val policy = CascadePolicy.fromAssets(appContext)
            val decision: CascadeDecision
            val decisionThreshold: Float
            if (policy != null && engineCScore != null) {
                emit(ScanProgress("Arbitrating dual-engine cascade…", 90f))
                decision = policy.decide(fusedScore, engineCScore)
                // The cascade now reports the FUSION score in every branch,
                // so the matching operating point is Engine A's (the
                // fusion) threshold - NOT Engine C's. Reporting Engine C's
                // threshold against a fusion score would compare the score
                // against the wrong operating point.
                decisionThreshold = policy.engineAThreshold
            } else {
                // Single engine (Engine A alone). Its own threshold applies.
                val threshold = report.threshold ?: CalibrationStore.metaDecisionThreshold(appContext)
                decisionThreshold = threshold
                decision = CascadeDecision(
                    verdict = if (fusedScore >= threshold) {
                        DetectionVerdict.SYNTHETIC
                    } else {
                        DetectionVerdict.AUTHENTIC
                    },
                    score = fusedScore,
                    escalated = false,
                    reason = "Single-engine fusion",
                )
            }

            if (decision.verdict == DetectionVerdict.INCONCLUSIVE) {
                emit(ScanProgress("Analysis inconclusive", 100f, result = result(
                    AnalysisStatus.INCONCLUSIVE, decision.reason, decision.score, report, decisionThreshold)))
                return@flow
            }

            emit(ScanProgress("Analysis complete", 100f, result = result(
                decision.status, null, decision.score, report, decisionThreshold,
                mediaSha256 = mediaSha256)))
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
        //
        // The app's OWN per-process heap budget is the thing to gate on.
        //
        // `availMem` is system-wide free memory, and it is NOT this app's
        // budget: another app can bleed it to near zero and make the gate
        // fire unpredictably. The previous code used it anyway, and the
        // consequence was that a busy phone dropped to MATH_ONLY, which
        // skips the SRM CNN, which makes the pipeline refuse to score at
        // all (`srm < 0f`) - so the app returned "inconclusive" with no
        // verdict and no percentage for a reason that had nothing to do
        // with this app's capability.
        //
        // The models are MEMORY-MAPPED from the APK, so the 89.96 MB
        // SigLIP-B/16 tower costs page cache, not Java heap. What the heap
        // must hold is the interpreter's activation tensors (a few MB for
        // the tower) and the decoded bitmap.
        val heapBudgetMb = am.memoryClass
        return when {
            // A genuinely exhausted system is refused outright: the
            // interpreter's allocations would fail, and the pipeline would
            // report MODEL_FAILURE rather than a wrong answer. This branch
            // comes FIRST, because the previous ordering put it after the
            // MATH_ONLY branch, where it could never fire.
            availMb < 64 -> DegradationLevel.ABSTAIN
            // The app's own per-process heap budget.
            heapBudgetMb < 64 -> DegradationLevel.ABSTAIN
            // MATH_ONLY means "the tower does not run", NOT "nothing runs".
            // The SRM CNN is 143 KB and memory-mapped, so it is not what the
            // memory budget is about; the 89.96 MB tower is. Skipping the
            // SRM CNN made the pipeline refuse to score at all, which is the
            // same "no verdict, no percentage" symptom as the validation
            // gate. The fusion's probe-free branch is exactly what it was
            // trained for, and the cascade policy already has a path for it
            // ("Engine C unavailable; fusion alone").
            heapBudgetMb < 128 -> DegradationLevel.MATH_ONLY
            else -> DegradationLevel.FULL
        }
    }
}
