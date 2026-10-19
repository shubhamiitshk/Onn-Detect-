package com.onndetect.detection.fusion

/**
 * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY.
 * Transpiled XGBoost meta-classifier (4 trees, 10 features).
 * Feature order:
 *   0 srm_score, 1 fft_score, 2 kl_score, 3 flatness_score
 *   4 resolution_norm, 5 is_compressed, 6 exif_anomaly, 7 c2pa_present
 *   8 lip_sync_corr, 9 landmark_jitter
 */
class XGBoostMetaClassifier {
    companion object { const val RELEASE_ID = "onn-20261019-v4" }

    /** @return P(AI-generated) in [0,1]. */
    fun predict(features: FloatArray): Float {
        require(features.size == 10) { "Feature schema mismatch" }
        require(features.all { it.isFinite() }) { "Non-finite features" }
        var margin = -2.838335417393312e-10f
        // Tree 0
        if (!features[1].isNaN() && features[1] < 0.585281372f) {
            if (!features[2].isNaN() && features[2] < 0.948509514f) {
                if (!features[2].isNaN() && features[2] < 0.515574694f) {
                    margin += -0.0666666701f
                } else {
                    margin += 0.0375000015f
                }
            } else {
                margin += -0.131034479f
            }
        } else {
            if (!features[3].isNaN() && features[3] < 0.00219737738f) {
                if (!features[2].isNaN() && features[2] < 0.160385311f) {
                    margin += -0.0933333337f
                } else {
                    margin += -0.00213523139f
                }
            } else {
                if (!features[2].isNaN() && features[2] < 0.788297117f) {
                    margin += 0.0818181857f
                } else {
                    margin += -0.0117647061f
                }
            }
        }
        // Tree 1
        if (!features[3].isNaN() && features[3] < 0.000450881809f) {
            if (!features[2].isNaN() && features[2] < 0.648802638f) {
                if (!features[2].isNaN() && features[2] < 0.188123807f) {
                    margin += -0.0180708263f
                } else {
                    margin += -0.144254789f
                }
            } else {
                if (!features[3].isNaN() && features[3] < 0.00027068262f) {
                    margin += -0.0954758748f
                } else {
                    margin += 0.0771038383f
                }
            }
        } else {
            if (!features[1].isNaN() && features[1] < 0.583451688f) {
                if (!features[2].isNaN() && features[2] < 0.948509514f) {
                    margin += -0.028616393f
                } else {
                    margin += -0.117788091f
                }
            } else {
                if (!features[3].isNaN() && features[3] < 0.00245911325f) {
                    margin += -0.00140053162f
                } else {
                    margin += 0.0712582618f
                }
            }
        }
        // Tree 2
        if (!features[0].isNaN() && features[0] < 0.6953125f) {
            if (!features[3].isNaN() && features[3] < 0.000714186463f) {
                if (!features[0].isNaN() && features[0] < 0.50390625f) {
                    margin += -0.0293137562f
                } else {
                    margin += -0.0862555802f
                }
            } else {
                if (!features[2].isNaN() && features[2] < 0.0886863992f) {
                    margin += -0.0667780638f
                } else {
                    margin += 0.010130777f
                }
            }
        } else {
            if (!features[3].isNaN() && features[3] < 0.00170605106f) {
                if (!features[3].isNaN() && features[3] < 0.00113839656f) {
                    margin += 0.0689225346f
                } else {
                    margin += -0.0874416754f
                }
            } else {
                if (!features[3].isNaN() && features[3] < 0.00421947986f) {
                    margin += 0.150776073f
                } else {
                    margin += 0.0379193388f
                }
            }
        }
        // Tree 3
        if (!features[0].isNaN() && features[0] < 0.703125f) {
            if (!features[2].isNaN() && features[2] < 0.0886863992f) {
                if (!features[0].isNaN() && features[0] < 0.55859375f) {
                    margin += -0.107448064f
                } else {
                    margin += -0.00906993356f
                }
            } else {
                if (!features[0].isNaN() && features[0] < 0.578125f) {
                    margin += -0.0218570922f
                } else {
                    margin += 0.0218389928f
                }
            }
        } else {
            if (!features[2].isNaN() && features[2] < 0.770737886f) {
                if (!features[2].isNaN() && features[2] < 0.55512321f) {
                    margin += 0.105113424f
                } else {
                    margin += 0.0306207668f
                }
            } else {
                margin += -0.0166630503f
            }
        }

        return sigmoid(margin)
    }

    private fun sigmoid(x: Float): Float =
        (1.0 / (1.0 + Math.exp(-x.toDouble()))).toFloat()
}
