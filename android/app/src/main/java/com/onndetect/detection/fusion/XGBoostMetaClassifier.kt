package com.onndetect.detection.fusion

/**
 * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY.
 * Transpiled XGBoost meta-classifier (14 trees, 10 features).
 * Feature order:
 *   0 srm_score, 1 fft_score, 2 kl_score, 3 flatness_score
 *   4 resolution_norm, 5 is_compressed, 6 semantic_probe_score
 *   7 c2pa_present, 8 lip_sync_corr, 9 landmark_jitter
 */
class XGBoostMetaClassifier {
    companion object { const val RELEASE_ID = "onn-20261020-v6" }

    /** @return P(AI-generated) in [0,1]. */
    fun predict(features: FloatArray): Float {
        require(features.size == 10) { "Feature schema mismatch" }
        require(features.withIndex().all { (i, v) -> (i in listOf(6)) || v.isFinite() }) { "Non-finite feature outside the optional slot" }
        var margin = 3.465838095684237e-09f
        // Tree 0
        if (features[6].isNaN() || features[6] < 0.25390625f) {
            if (!features[1].isNaN() && features[1] < 0.63346535f) {
                if (features[6].isNaN() || features[6] < 0.03125f) {
                    margin += -0.055319149f
                } else {
                    margin += -0.142857149f
                }
            } else {
                if (!features[3].isNaN() && features[3] < 0.00169446098f) {
                    margin += -0.0761061981f
                } else {
                    margin += 0.0298850574f
                }
            }
        } else {
            if (!features[1].isNaN() && features[1] < 0.95955044f) {
                if (!features[2].isNaN() && features[2] < 0.160385311f) {
                    margin += 0.0f
                } else {
                    margin += 0.0789915994f
                }
            } else {
                if (!features[6].isNaN() && features[6] < 0.74609375f) {
                    margin += -0.103030302f
                } else {
                    margin += 0.0588235334f
                }
            }
        }
        // Tree 1
        if (features[6].isNaN() || features[6] < 0.67578125f) {
            if (!features[6].isNaN() && features[6] < 0.25390625f) {
                if (!features[1].isNaN() && features[1] < 0.754914939f) {
                    margin += -0.128275335f
                } else {
                    margin += -0.0272674803f
                }
            } else {
                if (!features[1].isNaN() && features[1] < 0.99557215f) {
                    margin += 0.00775262434f
                } else {
                    margin += -0.0771652237f
                }
            }
        } else {
            if (!features[3].isNaN() && features[3] < 0.000804620271f) {
                if (!features[3].isNaN() && features[3] < 0.000537588086f) {
                    margin += 0.0249742959f
                } else {
                    margin += -0.0582458079f
                }
            } else {
                if (!features[2].isNaN() && features[2] < 0.0540794469f) {
                    margin += -0.0343183912f
                } else {
                    margin += 0.100227594f
                }
            }
        }
        // Tree 2
        if (features[6].isNaN() || features[6] < 0.5078125f) {
            if (!features[0].isNaN() && features[0] < 0.37890625f) {
                if (features[6].isNaN() || features[6] < 0.25390625f) {
                    margin += -0.0845571384f
                } else {
                    margin += 0.0025027988f
                }
            } else {
                if (!features[3].isNaN() && features[3] < 0.00256615132f) {
                    margin += -0.0356681347f
                } else {
                    margin += 0.0418116786f
                }
            }
        } else {
            if (!features[3].isNaN() && features[3] < 0.000804620271f) {
                if (!features[6].isNaN() && features[6] < 0.60546875f) {
                    margin += 0.0469307229f
                } else {
                    margin += -0.0852229595f
                }
            } else {
                if (!features[2].isNaN() && features[2] < 0.124359392f) {
                    margin += -0.0103105698f
                } else {
                    margin += 0.0828891024f
                }
            }
        }
        // Tree 3
        if (features[6].isNaN() || features[6] < 0.67578125f) {
            if (!features[0].isNaN() && features[0] < 0.61328125f) {
                if (features[6].isNaN() || features[6] < 0.25390625f) {
                    margin += -0.0723967329f
                } else {
                    margin += -0.0213218275f
                }
            } else {
                if (!features[0].isNaN() && features[0] < 0.86328125f) {
                    margin += 0.00518610794f
                } else {
                    margin += 0.0959278792f
                }
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.77734375f) {
                if (!features[0].isNaN() && features[0] < 0.67578125f) {
                    margin += 0.050986696f
                } else {
                    margin += 0.14130871f
                }
            } else {
                if (!features[0].isNaN() && features[0] < 0.890625f) {
                    margin += -0.0117342388f
                } else {
                    margin += 0.0783069581f
                }
            }
        }
        // Tree 4
        if (!features[0].isNaN() && features[0] < 0.3671875f) {
            if (features[6].isNaN() || features[6] < 0.25390625f) {
                if (!features[0].isNaN() && features[0] < 0.1875f) {
                    margin += -0.00459333602f
                } else {
                    margin += -0.0922667459f
                }
            } else {
                if (!features[2].isNaN() && features[2] < 0.159601659f) {
                    margin += -0.0410751216f
                } else {
                    margin += 0.0600106008f
                }
            }
        } else {
            if (!features[3].isNaN() && features[3] < 0.00245911325f) {
                if (!features[3].isNaN() && features[3] < 0.00151619711f) {
                    margin += 0.0227842722f
                } else {
                    margin += -0.063179478f
                }
            } else {
                if (!features[2].isNaN() && features[2] < 0.712438345f) {
                    margin += 0.0760348514f
                } else {
                    margin += -0.0780966207f
                }
            }
        }
        // Tree 5
        if (!features[0].isNaN() && features[0] < 0.38671875f) {
            if (!features[2].isNaN() && features[2] < 0.55512321f) {
                if (!features[3].isNaN() && features[3] < 0.00374627719f) {
                    margin += -0.0858549103f
                } else {
                    margin += -0.0153214736f
                }
            } else {
                if (!features[3].isNaN() && features[3] < 0.000322510052f) {
                    margin += -0.0918665901f
                } else {
                    margin += 0.0187907182f
                }
            }
        } else {
            if (!features[3].isNaN() && features[3] < 0.00213643746f) {
                if (!features[3].isNaN() && features[3] < 0.00151619711f) {
                    margin += 0.0112139396f
                } else {
                    margin += -0.0703471825f
                }
            } else {
                if (!features[2].isNaN() && features[2] < 0.700606227f) {
                    margin += 0.0604676977f
                } else {
                    margin += -0.0536853634f
                }
            }
        }
        // Tree 6
        if (!features[0].isNaN() && features[0] < 0.34765625f) {
            if (features[6].isNaN() || features[6] < 0.25390625f) {
                if (!features[2].isNaN() && features[2] < 0.549597681f) {
                    margin += -0.11427214f
                } else {
                    margin += -0.0272407327f
                }
            } else {
                if (!features[3].isNaN() && features[3] < 0.000337022735f) {
                    margin += -0.0518577173f
                } else {
                    margin += 0.0410190187f
                }
            }
        } else {
            if (!features[1].isNaN() && features[1] < 0.585281372f) {
                if (!features[2].isNaN() && features[2] < 0.886289656f) {
                    margin += -0.0164559651f
                } else {
                    margin += -0.118047491f
                }
            } else {
                if (!features[3].isNaN() && features[3] < 0.000775020744f) {
                    margin += -0.015469457f
                } else {
                    margin += 0.0446771719f
                }
            }
        }
        // Tree 7
        if (!features[1].isNaN() && features[1] < 0.585281372f) {
            if (!features[0].isNaN() && features[0] < 0.8203125f) {
                if (!features[2].isNaN() && features[2] < 0.206540614f) {
                    margin += -0.0294814054f
                } else {
                    margin += -0.0875289366f
                }
            } else {
                margin += 0.0646970421f
            }
        } else {
            if (!features[3].isNaN() && features[3] < 0.00219737738f) {
                if (!features[2].isNaN() && features[2] < 0.721689105f) {
                    margin += -0.0258588884f
                } else {
                    margin += 0.049465239f
                }
            } else {
                if (!features[0].isNaN() && features[0] < 0.5f) {
                    margin += -0.0134203872f
                } else {
                    margin += 0.0883204043f
                }
            }
        }
        // Tree 8
        if (features[6].isNaN() || features[6] < 0.5078125f) {
            if (!features[0].isNaN() && features[0] < 0.35546875f) {
                if (features[6].isNaN() || features[6] < 0.25390625f) {
                    margin += -0.0764832795f
                } else {
                    margin += -0.00221825787f
                }
            } else {
                if (!features[0].isNaN() && features[0] < 0.90625f) {
                    margin += -0.0126959896f
                } else {
                    margin += 0.0986721665f
                }
            }
        } else {
            if (!features[2].isNaN() && features[2] < 0.124359392f) {
                if (!features[6].isNaN() && features[6] < 0.74609375f) {
                    margin += 0.0118088014f
                } else {
                    margin += -0.0703742728f
                }
            } else {
                if (!features[1].isNaN() && features[1] < 0.951207757f) {
                    margin += 0.0566424243f
                } else {
                    margin += -0.0333661139f
                }
            }
        }
        // Tree 9
        if (features[6].isNaN() || features[6] < 0.67578125f) {
            if (!features[0].isNaN() && features[0] < 0.6015625f) {
                if (!features[2].isNaN() && features[2] < 0.561905265f) {
                    margin += -0.0482010953f
                } else {
                    margin += -0.00405784277f
                }
            } else {
                if (!features[3].isNaN() && features[3] < 0.00211466756f) {
                    margin += -0.00645046029f
                } else {
                    margin += 0.0528899245f
                }
            }
        } else {
            if (!features[3].isNaN() && features[3] < 0.000804620271f) {
                if (!features[3].isNaN() && features[3] < 0.000682059152f) {
                    margin += 0.0103666475f
                } else {
                    margin += -0.0822407976f
                }
            } else {
                if (!features[3].isNaN() && features[3] < 0.00137671072f) {
                    margin += 0.125880584f
                } else {
                    margin += 0.0487522595f
                }
            }
        }
        // Tree 10
        if (!features[1].isNaN() && features[1] < 0.489454985f) {
            if (!features[1].isNaN() && features[1] < 0.380583137f) {
                if (!features[2].isNaN() && features[2] < 0.948509514f) {
                    margin += 0.0351178013f
                } else {
                    margin += -0.0938137993f
                }
            } else {
                if (!features[2].isNaN() && features[2] < 0.152821586f) {
                    margin += -0.0139494836f
                } else {
                    margin += -0.11721044f
                }
            }
        } else {
            if (!features[3].isNaN() && features[3] < 0.000714186463f) {
                if (!features[2].isNaN() && features[2] < 0.188123807f) {
                    margin += 0.0481227264f
                } else {
                    margin += -0.0400688536f
                }
            } else {
                if (!features[2].isNaN() && features[2] < 0.101963505f) {
                    margin += -0.0262094177f
                } else {
                    margin += 0.0319391228f
                }
            }
        }
        // Tree 11
        if (!features[1].isNaN() && features[1] < 0.489454985f) {
            if (!features[0].isNaN() && features[0] < 0.4921875f) {
                if (!features[6].isNaN() && features[6] < 0.13671875f) {
                    margin += -0.0123116262f
                } else {
                    margin += -0.118620627f
                }
            } else {
                if (!features[0].isNaN() && features[0] < 0.69140625f) {
                    margin += 0.029000828f
                } else {
                    margin += -0.0864983872f
                }
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.22265625f) {
                margin += -0.118999854f
            } else {
                if (!features[6].isNaN() && features[6] < 0.5078125f) {
                    margin += -0.0154516017f
                } else {
                    margin += 0.0204859041f
                }
            }
        }
        // Tree 12
        if (!features[6].isNaN() && features[6] < 0.25390625f) {
            if (!features[1].isNaN() && features[1] < 0.754914939f) {
                if (!features[6].isNaN() && features[6] < 0.13671875f) {
                    margin += -0.0426157303f
                } else {
                    margin += -0.106377281f
                }
            } else {
                if (!features[1].isNaN() && features[1] < 0.81174773f) {
                    margin += 0.091447182f
                } else {
                    margin += -0.0444111936f
                }
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.62109375f) {
                if (!features[2].isNaN() && features[2] < 0.128502354f) {
                    margin += -0.073119849f
                } else {
                    margin += 0.00880029891f
                }
            } else {
                if (!features[6].isNaN() && features[6] < 0.75390625f) {
                    margin += 0.00588236609f
                } else {
                    margin += 0.0486144908f
                }
            }
        }
        // Tree 13
        if (!features[6].isNaN() && features[6] < 0.2734375f) {
            if (!features[1].isNaN() && features[1] < 0.754914939f) {
                if (!features[0].isNaN() && features[0] < 0.3984375f) {
                    margin += -0.0955311432f
                } else {
                    margin += -0.0434216224f
                }
            } else {
                if (!features[0].isNaN() && features[0] < 0.26171875f) {
                    margin += -0.0980226621f
                } else {
                    margin += 0.0356294811f
                }
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.90625f) {
                if (features[6].isNaN() || features[6] < 0.69140625f) {
                    margin += 0.00169609603f
                } else {
                    margin += 0.0434136353f
                }
            } else {
                if (!features[6].isNaN() && features[6] < 0.796875f) {
                    margin += 0.0234835129f
                } else {
                    margin += 0.122842968f
                }
            }
        }

        return sigmoid(margin)
    }

    private fun sigmoid(x: Float): Float =
        (1.0 / (1.0 + Math.exp(-x.toDouble()))).toFloat()
}
