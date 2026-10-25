package com.onndetect.detection.fusion

/**
 * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY.
 * Transpiled XGBoost meta-classifier (16 trees, 10 features).
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
        var margin = 3.190582629694627e-09f + FusionBaseMargin.logit(features[6])
        // Tree 0
        if (features[6].isNaN() || features[6] < 0.503219843f) {
            if (!features[6].isNaN() && features[6] < 0.418713629f) {
                margin += -0.11242342f
            } else {
                margin += -0.0271092895f
            }
        } else {
            margin += 0.115756489f
        }
        // Tree 1
        if (features[6].isNaN() || features[6] < 0.503219843f) {
            if (!features[6].isNaN() && features[6] < 0.418713629f) {
                margin += -0.11195302f
            } else {
                margin += -0.0115123233f
            }
        } else {
            margin += 0.118288331f
        }
        // Tree 2
        if (features[6].isNaN() || features[6] < 0.503219843f) {
            if (!features[0].isNaN() && features[0] < 0.82421875f) {
                margin += -0.0571293794f
            } else {
                margin += 0.0696557835f
            }
        } else {
            margin += 0.110721819f
        }
        // Tree 3
        if (features[6].isNaN() || features[6] < 0.503219843f) {
            if (!features[0].isNaN() && features[0] < 0.7578125f) {
                margin += -0.069763802f
            } else {
                margin += 0.0428784862f
            }
        } else {
            margin += 0.107542731f
        }
        // Tree 4
        if (features[6].isNaN() || features[6] < 0.503219843f) {
            if (!features[0].isNaN() && features[0] < 0.82421875f) {
                margin += -0.0488041379f
            } else {
                margin += 0.0822015405f
            }
        } else {
            margin += 0.105141141f
        }
        // Tree 5
        if (!features[0].isNaN() && features[0] < 0.609375f) {
            if (!features[0].isNaN() && features[0] < 0.3671875f) {
                margin += -0.0606080554f
            } else {
                margin += -0.0101354364f
            }
        } else {
            if (!features[3].isNaN() && features[3] < 0.00211466756f) {
                margin += -0.0143364668f
            } else {
                margin += 0.0774595439f
            }
        }
        // Tree 6
        if (!features[6].isNaN() && features[6] < 0.418713629f) {
            margin += -0.10815157f
        } else {
            if (!features[0].isNaN() && features[0] < 0.34765625f) {
                margin += -0.0591385067f
            } else {
                margin += 0.0454484746f
            }
        }
        // Tree 7
        if (!features[0].isNaN() && features[0] < 0.34765625f) {
            if (!features[2].isNaN() && features[2] < 0.549597681f) {
                margin += -0.0887819678f
            } else {
                margin += 0.0197361112f
            }
        } else {
            if (!features[1].isNaN() && features[1] < 0.501551211f) {
                margin += -0.0590865389f
            } else {
                margin += 0.0322462171f
            }
        }
        // Tree 8
        if (features[6].isNaN() || features[6] < 0.503219843f) {
            if (!features[6].isNaN() && features[6] < 0.418713629f) {
                margin += -0.106664039f
            } else {
                margin += -0.0114041567f
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.83984375f) {
                margin += 0.107197665f
            } else {
                margin += 0.0344161429f
            }
        }
        // Tree 9
        if (features[6].isNaN() || features[6] < 0.503219843f) {
            if (!features[0].isNaN() && features[0] < 0.55078125f) {
                margin += -0.0655273944f
            } else {
                margin += 0.01693522f
            }
        } else {
            margin += 0.0994888544f
        }
        // Tree 10
        if (!features[3].isNaN() && features[3] < 0.00253942376f) {
            if (!features[1].isNaN() && features[1] < 0.810618162f) {
                margin += -0.0471337475f
            } else {
                margin += 0.00338031608f
            }
        } else {
            if (!features[2].isNaN() && features[2] < 0.948509514f) {
                margin += 0.0386383645f
            } else {
                margin += -0.0781068206f
            }
        }
        // Tree 11
        if (!features[6].isNaN() && features[6] < 0.418713629f) {
            margin += -0.10404592f
        } else {
            if (features[6].isNaN() || features[6] < 0.460737795f) {
                margin += 0.00221635844f
            } else {
                margin += 0.0985328481f
            }
        }
        // Tree 12
        if (features[6].isNaN() || features[6] < 0.503219843f) {
            if (!features[6].isNaN() && features[6] < 0.418713629f) {
                margin += -0.102246679f
            } else {
                margin += -0.00195973017f
            }
        } else {
            margin += 0.102508344f
        }
        // Tree 13
        if (!features[6].isNaN() && features[6] < 0.460737795f) {
            margin += -0.102053702f
        } else {
            if (features[6].isNaN() || features[6] < 0.593870878f) {
                margin += 0.00955042522f
            } else {
                margin += 0.0974157155f
            }
        }
        // Tree 14
        if (!features[0].isNaN() && features[0] < 0.62109375f) {
            if (!features[3].isNaN() && features[3] < 0.000804620271f) {
                margin += -0.0503588617f
            } else {
                margin += 0.00120694842f
            }
        } else {
            if (!features[1].isNaN() && features[1] < 0.866419613f) {
                margin += 0.0125847878f
            } else {
                margin += 0.0949433669f
            }
        }
        // Tree 15
        if (features[6].isNaN() || features[6] < 0.418713629f) {
            if (!features[6].isNaN() && features[6] < 0.418713629f) {
                margin += -0.0972558185f
            } else {
                margin += -0.00821388699f
            }
        } else {
            if (!features[2].isNaN() && features[2] < 0.977814555f) {
                margin += 0.106216684f
            } else {
                margin += 0.0176801905f
            }
        }

        return sigmoid(margin)
    }

    private fun sigmoid(x: Float): Float =
        (1.0 / (1.0 + Math.exp(-x.toDouble()))).toFloat()
}
