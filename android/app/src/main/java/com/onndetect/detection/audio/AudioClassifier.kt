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
            if (!report.validated) {
                return outcome(score, AnalysisStatus.INCONCLUSIVE, report.validationReason)
            }
            if (!SpeechContentGate.isSpeechLike(decoded.pcm, decoded.validSamples, TARGET_RATE)) {
                return outcome(score, AnalysisStatus.INCONCLUSIVE,
                    "Audio does not exhibit speech-like syllabic modulation; speech countermeasure abstains on stationary or non-speech audio")
            }
            return outcome(score, AnalysisStatus.SUCCESS, null)
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { return Verdict(-1f, false, null, AnalysisFailurePolicy.status(e), e.message ?: "Audio analysis failed") }
    }
}

/** Rejects stationary tones and unmodulated noise before speech classification. */
internal object SpeechContentGate {
    fun isSpeechLike(pcm: FloatArray, validSamples: Int, sampleRate: Int = 16000): Boolean {
        val n = validSamples.coerceAtMost(pcm.size)
        val frameLen = (sampleRate / 50).coerceAtLeast(64)
        val numFrames = n / frameLen
        if (numFrames < 10) return false
        val energies = FloatArray(numFrames)
        val zcrs = FloatArray(numFrames)
        var sumE = 0.0
        var sumZ = 0.0
        for (f in 0 until numFrames) {
            val base = f * frameLen
            var eSq = 0.0
            var zc = 0
            for (i in 0 until frameLen) {
                val s = pcm[base + i]
                if (!s.isFinite()) return false
                eSq += s.toDouble() * s.toDouble()
                if (i > 0 && (s >= 0f) != (pcm[base + i - 1] >= 0f)) zc++
            }
            val rms = kotlin.math.sqrt(eSq / frameLen).toFloat()
            val zcr = zc.toFloat() / (frameLen - 1)
            energies[f] = rms
            zcrs[f] = zcr
            sumE += rms
            sumZ += zcr
        }
        val meanE = (sumE / numFrames).toFloat()
        if (meanE < 1e-4f) return false
        val meanZ = (sumZ / numFrames).toFloat()
        var varE = 0.0
        var varZ = 0.0
        var activeFrames = 0
        for (f in 0 until numFrames) {
            val de = energies[f] - meanE
            val dz = zcrs[f] - meanZ
            varE += de * de
            varZ += dz * dz
            if (energies[f] > 0.15f * meanE) activeFrames++
        }
        val cvE = (kotlin.math.sqrt(varE / numFrames) / meanE).toFloat()
        val stdZ = kotlin.math.sqrt(varZ / numFrames).toFloat()
        val activeRatio = activeFrames.toFloat() / numFrames
        return cvE >= 0.20f && stdZ >= 0.015f && activeRatio in 0.15f..0.99f
    }
}

