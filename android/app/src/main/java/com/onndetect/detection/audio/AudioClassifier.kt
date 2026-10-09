package com.onndetect.detection.audio

import android.content.Context
import android.net.Uri
import com.onndetect.inference.LiteRTInferenceEngine
import com.onndetect.inference.ModelIntegrityChecker
import com.onndetect.inference.ModelIntegrityGate
import com.onndetect.inference.AnalysisFailurePolicy
import com.onndetect.inference.requireModelContract
import com.onndetect.domain.models.AnalysisStatus
import java.util.concurrent.CancellationException

/** Speech-model diagnostics; a missing/invalid signal never becomes a verdict. */
class AudioClassifier(private val context: Context, private val engine: LiteRTInferenceEngine,
    private val decoder: AudioDecoder = AudioDecoder(context), private val lfcc: LFCCExtractor = LFCCExtractor()) {
    companion object {
        const val MODEL_FILE = "lcnn_audio_int8.tflite"
        const val TARGET_RATE = 16000
        const val TARGET_SAMPLES = 64000
    }
    data class Verdict(val score: Float, val isAiGenerated: Boolean, val threshold: Float?,
        val status: AnalysisStatus = AnalysisStatus.INCONCLUSIVE, val reason: String? = null,
        val modelValidated: Boolean = false, val analyzedDurationMs: Long = 0, val totalDurationMs: Long = 0,
        val analyzedTimestampsMs: List<Long> = emptyList(), val modelVersion: String = "unknown",
        val preprocessingVersion: String = LFCCExtractor.FRONTEND_ID, val calibrationVersion: String = "unvalidated")

    fun analyze(uri: Uri, checkStop: () -> Unit = {}, onStage: (String) -> Unit = {}): Verdict {
        fun check() {
            if (Thread.currentThread().isInterrupted) throw CancellationException("Audio analysis cancelled")
            checkStop()
        }
        val report = ModelIntegrityGate.report(context, setOf(ModelIntegrityChecker.AUDIO_MODEL))
        if (!report.allOk) return Verdict(-1f, false, null, AnalysisStatus.MODEL_FAILURE, report.failures.joinToString())
        try {
            check(); onStage("Decoding an audio sample...")
            val decoded = decoder.decodeWindow(uri, TARGET_RATE, TARGET_SAMPLES, checkStop = ::check)
            fun outcome(score: Float, status: AnalysisStatus, reason: String?) = Verdict(score,
                status == AnalysisStatus.SUCCESS && score >= (report.threshold ?: 1f), report.threshold,
                status, reason, report.validated, decoded.analyzedDurationMs, decoded.totalDurationMs,
                listOf(decoded.startMs), report.releaseId, LFCCExtractor.FRONTEND_ID,
                if (report.validated) report.releaseId else "unvalidated")
            if (decoded.validSamples < TARGET_RATE / 2 || decoded.rms < 1e-4f || decoded.peak < 1e-3f) {
                return outcome(-1f, AnalysisStatus.INCONCLUSIVE, "No usable audio or too little material in the sampled window")
            }
            if (decoded.clippedFraction > .1f) return outcome(-1f, AnalysisStatus.INCONCLUSIVE, "Sampled audio is heavily clipped")
            check(); onStage("Extracting LFCC descriptors...")
            val contract = engine.contract(MODEL_FILE)
            requireModelContract(contract.inputShape.contentEquals(intArrayOf(1, 401, 40, 1))) { "Unsupported audio model tensor contract" }
            val features = lfcc.extractBuffer(decoded.pcm, ::check)
            check(); onStage("Audio model diagnostics...")
            val score = engine.predict(MODEL_FILE, features, ::check)
            // This product accepts general audio, but the network is a speech
            // countermeasure. A validated speech/music routing gate is needed
            // before its score can be used as a general-file verdict.
            return outcome(score, AnalysisStatus.INCONCLUSIVE, if (!report.validated) report.validationReason else
                "Speech-content routing and file-level sampled-window calibration are not validated")
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { return Verdict(-1f, false, null, AnalysisFailurePolicy.status(e), e.message ?: "Audio analysis failed") }
    }
}
