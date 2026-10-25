package com.onndetect.detection.fusion

/**
 * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY.
 * Transpiled XGBoost meta-classifier (7 trees, 10 features).
 * * The margin starts from the slot-6 probe's logit (see FusionBaseMargin); the trees only CORRECT it.
 * Feature order:
 *   0 srm_score, 1 fft_score, 2 kl_score, 3 flatness_score
 *   4 resolution_norm, 5 is_compressed, 6 semantic_probe_score
 *   7 c2pa_present, 8 lip_sync_corr, 9 landmark_jitter
 */
class XGBoostMetaClassifier {
    companion object { const val RELEASE_ID = "onn-prod-v10" }

    /** @return P(AI-generated) in [0,1]. */
    fun predict(features: FloatArray): Float {
        require(features.size == 10) { "Feature schema mismatch" }
        require(features.withIndex().all { (i, v) -> (i in listOf(6)) || v.isFinite() }) { "Non-finite feature outside the optional slot" }
        var margin = 6.237644456508207e-09f + FusionBaseMargin.logit(features[6])
        // Tree 0
        if (!features[0].isNaN() && features[0] < 0.4296875f) {
            if (!features[0].isNaN() && features[0] < 0.16796875f) {
                margin += -0.223455817f
            } else {
                margin += -0.0735361502f
            }
        } else {
            if (!features[2].isNaN() && features[2] < 0.125066549f) {
                margin += 0.0565056391f
            } else {
                margin += 0.149120852f
            }
        }
        // Tree 1
        if (!features[0].isNaN() && features[0] < 0.4296875f) {
            if (features[6].isNaN() || features[6] < 0.738344669f) {
                margin += -0.125852376f
            } else {
                margin += -0.348385453f
            }
        } else {
            if (features[6].isNaN() || features[6] < 0.894045115f) {
                margin += 0.151262462f
            } else {
                margin += 0.0708964691f
            }
        }
        // Tree 2
        if (!features[0].isNaN() && features[0] < 0.4296875f) {
            if (!features[0].isNaN() && features[0] < 0.16796875f) {
                margin += -0.191805184f
            } else {
                margin += -0.0484818034f
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.7265625f) {
                margin += 0.0265345424f
            } else {
                margin += 0.132148132f
            }
        }
        // Tree 3
        if (!features[0].isNaN() && features[0] < 0.4296875f) {
            if (!features[0].isNaN() && features[0] < 0.19140625f) {
                margin += -0.166150376f
            } else {
                margin += -0.0458173864f
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.55859375f) {
                margin += -0.000352348172f
            } else {
                margin += 0.118327655f
            }
        }
        // Tree 4
        if (!features[0].isNaN() && features[0] < 0.4140625f) {
            if (!features[0].isNaN() && features[0] < 0.19140625f) {
                margin += -0.1797508f
            } else {
                margin += -0.0349732898f
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.7265625f) {
                margin += 0.0478457883f
            } else {
                margin += 0.130223647f
            }
        }
        // Tree 5
        if (!features[0].isNaN() && features[0] < 0.2421875f) {
            if (features[6].isNaN() || features[6] < 0.738344669f) {
                margin += -0.102728896f
            } else {
                margin += -0.249161124f
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.82421875f) {
                margin += 0.033773195f
            } else {
                margin += 0.132007703f
            }
        }
        // Tree 6
        if (!features[0].isNaN() && features[0] < 0.203125f) {
            if (!features[2].isNaN() && features[2] < 0.0751719996f) {
                margin += -7.58929236e-05f
            } else {
                margin += -0.164556786f
            }
        } else {
            if (!features[6].isNaN() && features[6] < 0.941621184f) {
                margin += -0.0404051393f
            } else {
                margin += 0.0896521732f
            }
        }

        return sigmoid(margin)
    }

    private fun sigmoid(x: Float): Float =
        (1.0 / (1.0 + Math.exp(-x.toDouble()))).toFloat()
}
