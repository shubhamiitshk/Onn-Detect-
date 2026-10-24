package com.onndetect.detection.audio

import android.content.Context
import android.net.Uri
import com.onndetect.inference.LiteRTInferenceEngine
import com.onndetect.inference.ModelIntegrityChecker
import com.onndetect.inference.ModelIntegrityGate
import com.onndetect.inference.AnalysisFailurePolicy
import com.onndetect.inference.requireModelContract
import com.onndetect.domain.models.AnalysisStatus
import com.onndetect.detection.provenance.AudioSealDetector
import com.onndetect.detection.provenance.ProvenanceEvidence
import java.util.concurrent.CancellationException

/** Speech-model diagnostics; a missing/invalid signal never becomes a verdict. */
class AudioClassifier(private val context: Context, private val engine: LiteRTInferenceEngine,
    private val decoder: AudioDecoder = AudioDecoder(context), private val lfcc: LFCCExtractor = LFCCExtractor(),
    private val audioSealDetector: AudioSealDetector = AudioSealDetector(engine)) {
    companion object {
        const val MODEL_FILE = "lcnn_audio_int8.tflite"
        const val TARGET_RATE = 16000
        const val TARGET_SAMPLES = 64000
        /**
         * Calibrated file-level operating point of the speech countermeasure
         * (threshold_at_eer in the release manifest). A clip scoring >= this is
         * classified synthetic; enrollment requires strictly BELOW it (F4).
         */
        const val DEFAULT_THRESHOLD = 0.375f
        /** 60 s @ 16 kHz: the F4 enrollment decode budget. */
        const val MAX_DECODE_SAMPLES = 960_000
        /**
         * Number of independent windows scored per file.
         *
         * ONE is the shipped, calibrated behaviour: the released threshold
         * (0.375) was fitted on SINGLE-WINDOW scores, so pooling several
         * windows and comparing the pooled value against that threshold
         * would silently change the operating point.
         *
         * Pooling raises the score (LSE of N samples is above the mean),
         * so multi-window scoring must not be enabled until a file-level
         * threshold has been calibrated on file-level labels. The code path
         * exists and is tested; the default stays at one window until that
         * calibration exists.
         */
        const val WINDOWS_PER_FILE = 1
    }
    data class Verdict(val score: Float, val isAiGenerated: Boolean, val threshold: Float?,
        val status: AnalysisStatus = AnalysisStatus.INCONCLUSIVE, val reason: String? = null,
        val modelValidated: Boolean = false, val analyzedDurationMs: Long = 0, val totalDurationMs: Long = 0,
        val analyzedTimestampsMs: List<Long> = emptyList(), val modelVersion: String = "unknown",
        val preprocessingVersion: String = LFCCExtractor.FRONTEND_ID, val calibrationVersion: String = "unvalidated",
        /** Per-window scores in timestamp order; empty when a single window was scored. */
        val windowScores: List<Float> = emptyList(),
        val provenanceEvidence: ProvenanceEvidence = ProvenanceEvidence.Absent)

    fun analyze(
        uri: Uri,
        windows: Int = WINDOWS_PER_FILE,
        checkStop: () -> Unit = {},
        onStage: (String) -> Unit = {},
    ): Verdict {
        fun check() {
            if (Thread.currentThread().isInterrupted) throw CancellationException("Audio analysis cancelled")
            checkStop()
        }
        val report = ModelIntegrityGate.report(context, setOf(ModelIntegrityChecker.AUDIO_MODEL))
        if (!report.allOk) return Verdict(-1f, false, null, AnalysisStatus.MODEL_FAILURE, report.failures.joinToString())
        try {
            require(windows >= 1) { "Window count must be positive" }
            check(); onStage("Decoding an audio sample...")
            val contract = engine.contract(MODEL_FILE)
            requireModelContract(contract.inputShape.contentEquals(intArrayOf(1, 401, 40, 1))) { "Unsupported audio model tensor contract" }

            val windowMs = TARGET_SAMPLES.toLong() * 1000 / TARGET_RATE
            // The first window reveals the file duration; the remaining
            // windows are laid out across it.
            val first = decoder.decodeWindow(uri, TARGET_RATE, TARGET_SAMPLES, checkStop = ::check)
            val offsets = if (windows == 1) listOf(first.startMs)
                else WindowPooling.windowOffsets(first.totalDurationMs, windowMs, windows)

            val scores = ArrayList<Float>(offsets.size)
            val timestamps = ArrayList<Long>(offsets.size)
            var analyzedMs = 0L
            var durationMs = first.totalDurationMs
            var clippedWindows = 0
            var speechLikeWindows = 0
            var provenanceEvidence: ProvenanceEvidence = ProvenanceEvidence.Absent

            for ((index, offset) in offsets.withIndex()) {
                check()
                val decoded = if (index == 0) first else
                    decoder.decodeWindow(uri, TARGET_RATE, TARGET_SAMPLES, requestedStartMs = offset, checkStop = ::check)
                if (decoded.totalDurationMs > durationMs) durationMs = decoded.totalDurationMs
                analyzedMs += decoded.analyzedDurationMs
                timestamps += decoded.startMs

                // Offline AudioSeal neural watermark detection
                val watermark = audioSealDetector.analyze(decoded.pcm)
                if (watermark is ProvenanceEvidence.WatermarkMark && watermark.confirmsAiInvolvement) {
                    provenanceEvidence = watermark
                }
                // An unusable window is SKIPPED rather than fatal: a file may
                // contain a silent stretch, and the remaining windows still
                // carry evidence. If every window is unusable the file-level
                // decision is inconclusive.
                if (decoded.validSamples < TARGET_RATE / 2 || decoded.rms < 1e-4f || decoded.peak < 1e-3f) continue
                if (decoded.clippedFraction > .1f) { clippedWindows++; continue }
                check(); onStage("Extracting LFCC descriptors...")
                val features = lfcc.extractBuffer(decoded.pcm, ::check)
                check(); onStage("Audio model diagnostics...")
                scores += engine.predict(MODEL_FILE, features, ::check)
                if (SpeechContentGate.isSpeechLike(decoded.pcm, decoded.validSamples, TARGET_RATE) &&
                    VoicedDurationGate.voicedMs(decoded.pcm, TARGET_RATE) >= 300L) speechLikeWindows++
            }

            val score = if (scores.isEmpty()) -1f
                else if (scores.size == 1) scores[0]
                else WindowPooling.poolProbabilities(scores.toFloatArray())
            fun outcome(status: AnalysisStatus, reason: String?): Verdict {
                val isConfirmed = when (val ev = provenanceEvidence) {
                    is ProvenanceEvidence.WatermarkMark -> ev.confirmsAiInvolvement
                    else -> false
                }
                val finalIsAi = isConfirmed || (status == AnalysisStatus.SUCCESS && score >= (report.threshold ?: 1f))
                val finalReason = if (isConfirmed) "Meta AudioSeal neural watermark confirmed AI involvement" else reason
                return Verdict(score,
                    finalIsAi, report.threshold,
                    status, finalReason, report.validated, analyzedMs, durationMs,
                    timestamps, report.releaseId, LFCCExtractor.FRONTEND_ID,
                    if (report.validated) report.releaseId else "unvalidated",
                    scores.toList(),
                    provenanceEvidence)
            }

            if (scores.isEmpty()) {
                return outcome(AnalysisStatus.INCONCLUSIVE,
                    if (clippedWindows > 0) "Sampled audio is heavily clipped"
                    else "No usable audio or too little material in the sampled window")
            }
            if (!report.validated) {
                return outcome(AnalysisStatus.INCONCLUSIVE, report.validationReason)
            }
            // The speech countermeasure abstains on stationary or non-speech
            // audio. A majority rule keeps that behaviour for a single
            // window while tolerating one non-speech window in a file.
            if (speechLikeWindows * 2 <= scores.size) {
                return outcome(AnalysisStatus.INCONCLUSIVE,
                    "Audio does not exhibit speech-like syllabic modulation; speech countermeasure abstains on stationary or non-speech audio")
            }
            return outcome(AnalysisStatus.SUCCESS, null)
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

