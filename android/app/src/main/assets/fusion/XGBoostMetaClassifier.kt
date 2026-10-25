package com.onndetect.detection.fusion

/**
 * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY.
 * Transpiled XGBoost meta-classifier (1 trees, 10 features).
 * * The margin starts from the slot-6 probe's logit (see FusionBaseMargin); the trees only CORRECT it.
 * Feature order:
 *   0 srm_score, 1 fft_score, 2 kl_score, 3 flatness_score
 *   4 resolution_norm, 5 is_compressed, 6 semantic_probe_score
 *   7 c2pa_present, 8 lip_sync_corr, 9 landmark_jitter
 */
class XGBoostMetaClassifier {
    companion object { const val RELEASE_ID = "onn-prod-v9" }

    /** @return P(AI-generated) in [0,1]. */
    fun predict(features: FloatArray): Float {
        require(features.size == 10) { "Feature schema mismatch" }
        require(features.withIndex().all { (i, v) -> (i in listOf(6)) || v.isFinite() }) { "Non-finite feature outside the optional slot" }
        var margin = -3.8562393556151875e-10f + FusionBaseMargin.logit(features[6])
        // Tree 0
        if (!features[2].isNaN() && features[2] < 0.179994404f) {
            if (features[6].isNaN() || features[6] < 0.0921159908f) {
                margin += -0.122676469f
            } else {
                margin += 0.0361782387f
            }
        } else {
            if (!features[3].isNaN() && features[3] < 1.2664218e-05f) {
                margin += 0.0182012208f
            } else {
                margin += 0.124125563f
            }
        }

        return sigmoid(margin)
    }

    private fun sigmoid(x: Float): Float =
        (1.0 / (1.0 + Math.exp(-x.toDouble()))).toFloat()
}
