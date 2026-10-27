package com.onndetect.detection.audio

import kotlin.math.exp
import kotlin.math.ln

/**
 * Pooling of per-window detector scores into a single file-level score.
 *
 * ## Why the scores must be converted to logits first
 *
 * `lcnn_audio_int8.tflite` ends in `Dense(1, activation="sigmoid")`
 * (`ml/src/export/torch_to_tflite_keras.py:188`), so the model emits a
 * PROBABILITY in `[0, 1]`, not a logit. Pooling those numbers with a
 * log-sum-exp would apply a soft-max-like nonlinearity to values that are
 * already probabilities, and the result would not be a probability at all.
 *
 * The correct order is: probability -> logit -> LSE in logit space ->
 * sigmoid. The output is then guaranteed to lie in `(0, 1)`, which the
 * `ScanResult` constructor requires (`overallScore in 0f..1f`) and which the
 * calibrated threshold comparison assumes.
 *
 * ## Why not a plain max
 *
 * A plain max over 8 windows inflates the file-level false-positive rate:
 * the more windows are taken, the more likely one of them is a false
 * positive. LSE with a finite temperature interpolates smoothly between
 * mean pooling (a whole file was synthesised) and max pooling (a short
 * splice was inserted), which is the behaviour a file-level decision
 * needs.
 */
internal object WindowPooling {

    /**
     * LSE pooling temperature. Higher approaches max pooling, lower
     * approaches mean pooling.
     */
    const val LSE_TEMPERATURE = 3.0f

    private const val EPS = 1e-6f

    /**
     * Pool per-window sigmoid probabilities into one file-level probability.
     *
     * @return a value in `(0, 1)`, or [Float.NaN] for an empty input.
     */
    fun poolProbabilities(probs: FloatArray, tau: Float = LSE_TEMPERATURE): Float {
        if (probs.isEmpty()) return Float.NaN
        require(tau > 0f && tau.isFinite()) { "LSE temperature must be positive and finite" }
        val logits = FloatArray(probs.size) { index ->
            val p = probs[index]
            require(p.isFinite() && p in 0f..1f) { "Window score must be a probability in [0,1]" }
            val clamped = p.coerceIn(EPS, 1f - EPS)
            ln((clamped / (1f - clamped)).toDouble()).toFloat()
        }
        var maxLogit = -Float.MAX_VALUE
        for (z in logits) if (z > maxLogit) maxLogit = z
        var sum = 0.0
        for (z in logits) sum += exp(((z - maxLogit) / tau).toDouble())
        val pooledLogit = maxLogit + tau * ln(sum / logits.size).toFloat()
        return (1.0 / (1.0 + exp(-pooledLogit.toDouble()))).toFloat()
    }

    /**
     * Blends pooled multi-window LCNN probabilities with utterance-level pre-CMVN
     * non-vocal residual anomaly in log-odds space (v11 PRISM).
     */
    fun combineMultiWindowAndNonVocalAnomaly(
        probs: FloatArray,
        nonVocalAnomaly: Float = Float.NaN,
        tau: Float = LSE_TEMPERATURE,
    ): Float {
        val basePooled = poolProbabilities(probs, tau)
        if (!basePooled.isFinite()) return Float.NaN
        if (!nonVocalAnomaly.isFinite() || nonVocalAnomaly !in 0f..1f) return basePooled
        val eps = 1e-6f
        val pClamped = basePooled.coerceIn(eps, 1f - eps)
        val mClamped = nonVocalAnomaly.coerceIn(eps, 1f - eps)
        val zBase = ln((pClamped / (1f - pClamped)).toDouble()).toFloat()
        val zMnp = ln((mClamped / (1f - mClamped)).toDouble()).toFloat()
        val fusedLogit = 0.80f * zBase + 0.20f * zMnp
        return (1.0 / (1.0 + exp(-fusedLogit.toDouble()))).toFloat().coerceIn(0f, 1f)
    }

    /**
     * Evenly spaced window start offsets across a file of [durationMs].
     *
     * The first and last 5% are excluded so a window never starts on a
     * leading or trailing silence that would carry no evidence.
     */
    fun windowOffsets(durationMs: Long, windowMs: Long, count: Int): List<Long> {
        require(count >= 1 && durationMs > 0 && windowMs > 0) { "Invalid window layout" }
        if (count == 1) return listOf(maxOf(0L, (durationMs - windowMs) / 2))
        val margin = durationMs / 20
        val span = durationMs - 2 * margin - windowMs
        if (span <= 0) return listOf(maxOf(0L, (durationMs - windowMs) / 2))
        return (0 until count).map { margin + span * it / (count - 1) }
    }
}
