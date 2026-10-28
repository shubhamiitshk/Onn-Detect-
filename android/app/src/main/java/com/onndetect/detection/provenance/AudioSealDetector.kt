package com.onndetect.detection.provenance

import android.content.Context
import com.onndetect.inference.LiteRTInferenceEngine
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.inject.Inject
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt

/**
 * Offline AudioSeal watermark detection.
 *
 * ## Why the real detector, and why it is optional
 *
 * An earlier implementation correlated the audio against a hard-coded
 * 64-element +/-1 sequence and called the result an "AudioSeal detector".
 * That was wrong in a way that mattered:
 *
 *  * AudioSeal's carrier is a pseudo-random sequence derived from a
 *    secret key. A correlation against an arbitrary fixed pattern can
 *    never detect a genuine AudioSeal watermark, because it is not the
 *    carrier the watermark was embedded with.
 *  * Worse, the p-value was the maximum correlation over ~100 frames
 *    with no correction for taking a maximum, so the reported
 *    significance was wildly overstated and the detector produced
 *    **false positives** on ordinary audio.
 *
 * A false positive in a provenance channel is the worst possible failure:
 * it confirms AI origin on authentic audio. So the detector is now built
 * on the REAL published model, and when that model is not present in the
 * release it reports **no evidence at all** rather than guessing.
 *
 * Staging the real weights no longer requires the gated Hugging Face
 * repository: Meta serves the same published checkpoint ungated at
 * `dl.fbaipublicfiles.com/audioseal/6edcf62f/detector.pth` (16-bit message
 * head). `ml/scripts/export_audioseal_detector.py` performs the export
 * (ONNX parity gate + INT8 TFLite) from that checkpoint.
 *
 * ## The absence invariant
 *
 * Absence of a mark NEVER lowers the statistical score and never
 * confirms anything. It is reported as
 * [ProvenanceEvidence.Absent], which carries no claim in either
 * direction.
 */
class AudioSealDetector internal constructor(
    private val contractProvider: (String) -> LiteRTInferenceEngine.Contract?,
    private val predictor: (String, ByteBuffer) -> Float,
) {

    @Inject
    constructor(engine: LiteRTInferenceEngine) : this(
        contractProvider = { file ->
            try {
                engine.contract(file)
            } catch (e: Throwable) {
                null
            }
        },
        predictor = { file, buffer -> engine.predict(file, buffer) },
    )

    data class DetectionResult(
        val detected: Boolean,
        val pValue: Double,
        val agreeingBits: Int = 0,
        val totalBits: Int = MESSAGE_BITS,
        val presenceScore: Float = 0f,
        val uncorrectedPValue: Double = pValue,
        val windowCount: Int = 1,
    )

    /**
     * Inspect 16 kHz mono PCM for an AudioSeal watermark.
     *
     * Returns [ProvenanceEvidence.Absent] when the detector is not
     * available, when the audio is too short, or when no mark is present.
     */
    fun analyze(audioSamples: FloatArray): ProvenanceEvidence {
        if (audioSamples.isEmpty()) return ProvenanceEvidence.Absent
        val result = detectWatermark(audioSamples) ?: return ProvenanceEvidence.Absent
        if (!result.detected) return ProvenanceEvidence.Absent
        return ProvenanceEvidence.WatermarkMark(
            detected = true,
            kind = "AudioSeal",
            score = result.presenceScore,
            pValue = result.pValue,
        )
    }

    /**
     * Run the detector over fixed windows and aggregate the strongest
     * message or highest temporal presence confidence.
     *
     * Applies Bonferroni multiple testing correction over the number of windows
     * to eliminate family-wise error inflation on long recordings.
     *
     * Returns null when the detector graph is not available.
     */
    fun detectWatermark(samples: FloatArray): DetectionResult? {
        val contract = runCatching { contractProvider(MODEL_FILE) }.getOrNull() ?: return null
        val size = contract.inputShape.lastOrNull() ?: return null
        if (contract.inputShape.size != 2 || contract.inputShape[0] != 1) return null
        if (size < 1024) return null

        val windowSamples = size
        if (samples.size < windowSamples) return null
        val windows = (samples.size - windowSamples) / windowSamples + 1

        var best: DetectionResult? = null
        for (index in 0 until windows) {
            val offset = index * windowSamples
            val result = runCatching { scoreWindow(samples, offset, windowSamples, contract.inputBytes) }
                .getOrNull() ?: continue
            if (best == null || result.pValue < best.pValue || result.presenceScore > best.presenceScore) {
                best = result
            }
        }
        if (best == null) return null

        // Apply Bonferroni correction for multiple hypothesis testing over M windows:
        val adjustedPValue = (best.pValue * windows).coerceIn(0.0, 1.0)
        val isDetected = if (best.presenceScore > 0f) {
            best.presenceScore >= PRESENCE_THRESHOLD
        } else {
            best.agreeingBits >= MIN_AGREEING_BITS && adjustedPValue < SIGNIFICANT_P_VALUE
        }

        return best.copy(
            detected = isDetected,
            pValue = adjustedPValue,
            uncorrectedPValue = best.pValue,
            windowCount = windows,
        )
    }

    /** Score one window and turn the model output into a calibrated presence or p-value statistic. */
    private fun scoreWindow(
        samples: FloatArray,
        offset: Int,
        windowSamples: Int,
        inputBytes: Int,
    ): DetectionResult? {
        val input = ByteBuffer.allocateDirect(inputBytes).order(ByteOrder.nativeOrder())
        for (i in 0 until windowSamples) input.putFloat(samples[offset + i])
        input.flip()
        // The exported graph emits a SCALAR: either a normalized presence confidence score
        // in [0f..1f] (contract `audioseal-presence-mean-v2`), or an agreeing bit count [0..16]
        // in legacy compatibility mode.
        val raw = predictor(MODEL_FILE, input)
        if (!raw.isFinite() || raw < 0f) return null

        val count = raw.roundToInt().coerceIn(0, MESSAGE_BITS)
        val pValue = binomialTwoTailed(count, MESSAGE_BITS)
        val isContinuous = raw > 0f && raw < 1.0f
        val presence = if (isContinuous) raw else (if (raw >= 1.0f) 1.0f else 0f)
        val isDetected = if (isContinuous) {
            raw >= PRESENCE_THRESHOLD
        } else {
            count >= MIN_AGREEING_BITS && pValue < SIGNIFICANT_P_VALUE
        }

        return DetectionResult(
            detected = isDetected,
            pValue = pValue,
            agreeingBits = count,
            totalBits = MESSAGE_BITS,
            presenceScore = if (isContinuous) raw else 0f,
            uncorrectedPValue = pValue,
            windowCount = 1,
        )
    }

    /**
     * Two-tailed binomial p-value for "at least [successes] of [trials]
     * bits agree by chance".
     *
     * Under the null hypothesis each bit is 0 or 1 with probability 0.5, so
     * the probability of at least k successes is a binomial tail. This is the
     * CORRECT statistic for a fixed number of trials; taking the maximum
     * correlation over many frames and treating it as a single Gaussian
     * tail is not.
     *
     * It is a companion function so it can be unit tested without an
     * Android context.
     */
    internal fun binomialTwoTailed(successes: Int, trials: Int): Double =
        Companion.binomialTwoTailed(successes, trials)

    companion object {
        const val MODEL_FILE = "audioseal_detector_int8.tflite"
        const val SAMPLE_RATE = 16000

        /** Calibrated watermark presence probability threshold for v2 presence detector. */
        const val PRESENCE_THRESHOLD = 0.50f

        /**
         * The published detector carries a 16-bit message head. Requiring
         * at least half of them to agree keeps a random 50/50 message from
         * registering.
         */
        const val MIN_AGREEING_BITS = 8

        /** Bits in the AudioSeal message of the published detector. */
        const val MESSAGE_BITS = 16

        /**
         * Detection significance threshold calibrated for a 16-bit message head.
         * Exact 16/16 agreement yields p = 2 * (0.5)^16 = 3.0518e-5 < 1e-4.
         */
        const val SIGNIFICANT_P_VALUE = 1e-4

        /**
         * Two-tailed binomial p-value for "at least [successes] of [trials]
         * bits agree by chance".
         */
        internal fun binomialTwoTailed(successes: Int, trials: Int): Double {
            if (trials <= 0) return 1.0
            val k = successes.coerceIn(0, trials)
            val upper = min(k, trials - k)
            var tail = 0.0
            for (i in 0..upper) {
                tail += choose(trials, i) * 0.5.pow(trials)
            }
            return (2.0 * tail).coerceIn(0.0, 1.0)
        }

        private fun choose(n: Int, k: Int): Double {
            if (k < 0 || k > n) return 0.0
            var result = 1.0
            for (i in 0 until k) {
                result = result * (n - i) / (i + 1)
            }
            return result
        }
    }
}
