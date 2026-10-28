package com.onndetect.detection.visual

import kotlin.math.exp

/**
 * Diagnostic Forensic Feature Head for frozen foundation model embeddings (e.g., EmbeddingGemma 2).
 *
 * Implements the ordered affine dot product matching `ml/scripts/train_frozen_forensic_head.py`.
 * Standardization statistics (mean, std) are pre-folded directly into [weights] and [bias]
 * at training time, allowing on-device mobile execution with zero extra arithmetic passes.
 *
 * All outputs have [diagnosticOnly] = true to preserve baseline production cascade invariants.
 */
class ForensicFeatureHead(
    val modelId: String,
    val featureDim: Int,
    val weights: FloatArray,
    val bias: Float,
    val threshold: Float,
    val diagnosticOnly: Boolean = true,
) {
    init {
        require(weights.size == featureDim) {
            "Weight dimension (${weights.size}) does not match expected featureDim ($featureDim)"
        }
        require(threshold.isFinite() && threshold in 0f..1f) {
            "Invalid threshold: $threshold"
        }
    }

    data class EvaluationResult(
        val probability: Float,
        val isSynthetic: Boolean,
        val threshold: Float,
        val diagnosticOnly: Boolean,
    )

    /**
     * Scores a normalized feature vector in [0f..1f].
     *
     * Affine computation:
     *      logit = dot(features, weights) + bias
     *      prob  = 1 / (1 + exp(-logit))
     */
    fun predict(features: FloatArray): EvaluationResult {
        require(features.size == featureDim) {
            "Input feature vector size (${features.size}) does not match head dimension ($featureDim)"
        }

        var dot = 0.0
        for (i in 0 until featureDim) {
            val f = features[i]
            require(f.isFinite()) { "Non-finite feature at index $i: $f" }
            dot += f.toDouble() * weights[i].toDouble()
        }
        val affine = (dot + bias.toDouble()).toFloat()
        val clipped = affine.coerceIn(-88f, 88f)
        val prob = (1.0 / (1.0 + exp(-clipped.toDouble()))).toFloat()

        return EvaluationResult(
            probability = prob,
            isSynthetic = prob >= threshold,
            threshold = threshold,
            diagnosticOnly = diagnosticOnly,
        )
    }

    companion object {
        /**
         * Parses a serialized JSON head exported by `evaluate_embeddinggemma2_probe.py` or
         * `train_frozen_forensic_head.py`.
         */
        @JvmStatic
        fun parseJson(jsonString: String): ForensicFeatureHead {
            val root = org.json.JSONObject(jsonString)
            val backbone = root.optString("backbone", "unknown")
            val embeddingDim = root.getInt("embedding_dim")
            val weightsArray = root.getJSONArray("weights")
            require(weightsArray.length() == embeddingDim) {
                "JSON weights count (${weightsArray.length()}) != embedding_dim ($embeddingDim)"
            }
            val weights = FloatArray(embeddingDim) { i ->
                weightsArray.getDouble(i).toFloat()
            }
            val bias = root.optDouble("bias", 0.0).toFloat()
            val threshold = root.optDouble("decision_threshold", 0.50).toFloat()
            val diagnosticOnly = root.optBoolean("diagnostic_only", true)

            return ForensicFeatureHead(
                modelId = backbone,
                featureDim = embeddingDim,
                weights = weights,
                bias = bias,
                threshold = threshold,
                diagnosticOnly = diagnosticOnly,
            )
        }
    }
}
