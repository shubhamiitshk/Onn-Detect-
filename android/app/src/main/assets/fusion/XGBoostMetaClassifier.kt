package com.onndetect.detection.fusion

/**
 * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY.
 * Transpiled XGBoost meta-classifier (37 trees, 10 features).
 * Feature order:
 *   0 srm_score, 1 fft_score, 2 kl_score, 3 flatness_score
 *   4 resolution_norm, 5 is_compressed, 6 exif_anomaly, 7 c2pa_present
 *   8 lip_sync_corr, 9 landmark_jitter
 */
class XGBoostMetaClassifier {
    companion object { const val RELEASE_ID = "onn-20261018-v2" }

    /** @return P(AI-generated) in [0,1]. */
    fun predict(features: FloatArray): Float {
        require(features.size == 10) { "Feature schema mismatch" }
        require(features.all { it.isFinite() }) { "Non-finite features" }
        var margin = -3.910188106637236e-08f
        // Tree 0
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[2].isNaN() && features[2] < 0.0518152267f) {
                margin += 0.0545454584f
            } else {
                margin += 0.191160232f
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.5f) {
                if (!features[4].isNaN() && features[4] < 0.450195312f) {
                    margin += -0.193220347f
                } else {
                    margin += -0.0857142881f
                }
            } else {
                if (!features[3].isNaN() && features[3] < 0.00394030847f) {
                    margin += -0.0592f
                } else {
                    margin += 0.0511627905f
                }
            }
        }
        // Tree 1
        if (!features[5].isNaN() && features[5] < 1.0f) {
            margin += 0.169802412f
        } else {
            if (!features[3].isNaN() && features[3] < 0.00213643746f) {
                if (!features[3].isNaN() && features[3] < 0.000714186463f) {
                    margin += -0.126970142f
                } else {
                    margin += -0.0821409971f
                }
            } else {
                if (!features[1].isNaN() && features[1] < 0.607739747f) {
                    margin += -0.077035062f
                } else {
                    margin += 0.0239959806f
                }
            }
        }
        // Tree 2
        if (!features[5].isNaN() && features[5] < 1.0f) {
            margin += 0.15497826f
        } else {
            if (!features[0].isNaN() && features[0] < 0.5546875f) {
                if (!features[2].isNaN() && features[2] < 0.92575866f) {
                    margin += -0.11541485f
                } else {
                    margin += -0.0336186066f
                }
            } else {
                if (!features[2].isNaN() && features[2] < 0.458656222f) {
                    margin += 0.00600982271f
                } else {
                    margin += -0.0923534855f
                }
            }
        }
        // Tree 3
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[2].isNaN() && features[2] < 0.0518152267f) {
                margin += 0.01788643f
            } else {
                margin += 0.148359761f
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.5f) {
                margin += -0.16378437f
            } else {
                if (!features[0].isNaN() && features[0] < 0.6640625f) {
                    margin += -0.0557080768f
                } else {
                    margin += 0.0821363702f
                }
            }
        }
        // Tree 4
        if (!features[4].isNaN() && features[4] < 0.5f) {
            if (!features[0].isNaN() && features[0] < 0.84765625f) {
                margin += -0.136957124f
            } else {
                margin += 0.0495181382f
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.509765625f) {
                margin += 0.160286769f
            } else {
                if (!features[4].isNaN() && features[4] < 0.741210938f) {
                    margin += -0.135797024f
                } else {
                    margin += 0.045965936f
                }
            }
        }
        // Tree 5
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[2].isNaN() && features[2] < 0.0518152267f) {
                margin += 0.0599700026f
            } else {
                margin += 0.139336631f
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.5f) {
                margin += -0.145484313f
            } else {
                if (!features[0].isNaN() && features[0] < 0.5546875f) {
                    margin += -0.0720492303f
                } else {
                    margin += 0.0490698665f
                }
            }
        }
        // Tree 6
        if (!features[4].isNaN() && features[4] < 0.5f) {
            if (!features[0].isNaN() && features[0] < 0.8515625f) {
                if (!features[4].isNaN() && features[4] < 0.450195312f) {
                    margin += -0.126225173f
                } else {
                    margin += -0.04904465f
                }
            } else {
                margin += 0.0124570439f
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.578125f) {
                if (!features[4].isNaN() && features[4] < 0.9375f) {
                    margin += 0.0594998859f
                } else {
                    margin += -0.0475916229f
                }
            } else {
                if (!features[1].isNaN() && features[1] < 0.480590165f) {
                    margin += -0.0481716655f
                } else {
                    margin += 0.0836796165f
                }
            }
        }
        // Tree 7
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[2].isNaN() && features[2] < 0.0518152267f) {
                margin += 0.0465757437f
            } else {
                margin += 0.130333409f
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.625f) {
                if (!features[2].isNaN() && features[2] < 0.910644352f) {
                    margin += -0.0790334567f
                } else {
                    margin += 0.0100041768f
                }
            } else {
                if (!features[2].isNaN() && features[2] < 0.659588516f) {
                    margin += 0.0376640558f
                } else {
                    margin += -0.0864433795f
                }
            }
        }
        // Tree 8
        if (!features[5].isNaN() && features[5] < 1.0f) {
            margin += 0.117052309f
        } else {
            if (!features[4].isNaN() && features[4] < 0.5f) {
                margin += -0.129799709f
            } else {
                if (!features[4].isNaN() && features[4] < 0.509765625f) {
                    margin += 0.156945527f
                } else {
                    margin += -0.0271153487f
                }
            }
        }
        // Tree 9
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[2].isNaN() && features[2] < 0.0518152267f) {
                margin += 0.0449390672f
            } else {
                margin += 0.118464522f
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.64453125f) {
                if (!features[2].isNaN() && features[2] < 0.92575866f) {
                    margin += -0.0752858818f
                } else {
                    margin += 0.0159758832f
                }
            } else {
                if (!features[2].isNaN() && features[2] < 0.456543237f) {
                    margin += 0.0532860272f
                } else {
                    margin += -0.0748626366f
                }
            }
        }
        // Tree 10
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[4].isNaN() && features[4] < 0.259765625f) {
                margin += 0.0284225941f
            } else {
                margin += 0.121118799f
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.75f) {
                if (!features[4].isNaN() && features[4] < 0.453125f) {
                    margin += -0.126785532f
                } else {
                    margin += -0.0550197251f
                }
            } else {
                if (!features[4].isNaN() && features[4] < 0.87890625f) {
                    margin += 0.125776663f
                } else {
                    margin += -0.0147860413f
                }
            }
        }
        // Tree 11
        if (!features[5].isNaN() && features[5] < 1.0f) {
            margin += 0.104843251f
        } else {
            if (!features[4].isNaN() && features[4] < 0.453125f) {
                margin += -0.122774385f
            } else {
                if (!features[0].isNaN() && features[0] < 0.62890625f) {
                    margin += -0.0396551341f
                } else {
                    margin += 0.0578480437f
                }
            }
        }
        // Tree 12
        if (!features[4].isNaN() && features[4] < 0.5f) {
            if (!features[0].isNaN() && features[0] < 0.83203125f) {
                margin += -0.115174249f
            } else {
                margin += -0.0150958514f
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.509765625f) {
                margin += 0.135052517f
            } else {
                if (!features[4].isNaN() && features[4] < 0.75f) {
                    margin += -0.111696161f
                } else {
                    margin += 0.0315306894f
                }
            }
        }
        // Tree 13
        if (!features[5].isNaN() && features[5] < 1.0f) {
            margin += 0.107626557f
        } else {
            if (!features[4].isNaN() && features[4] < 0.453125f) {
                margin += -0.115874708f
            } else {
                if (!features[0].isNaN() && features[0] < 0.62890625f) {
                    margin += -0.0307131112f
                } else {
                    margin += 0.0596021414f
                }
            }
        }
        // Tree 14
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[4].isNaN() && features[4] < 0.375976562f) {
                margin += 0.0342501663f
            } else {
                margin += 0.110582285f
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.450195312f) {
                margin += -0.113301359f
            } else {
                if (!features[0].isNaN() && features[0] < 0.6484375f) {
                    margin += -0.0262739044f
                } else {
                    margin += 0.0580853038f
                }
            }
        }
        // Tree 15
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[2].isNaN() && features[2] < 0.977814555f) {
                if (!features[2].isNaN() && features[2] < 0.0418483019f) {
                    margin += 0.0269258656f
                } else {
                    margin += 0.101856723f
                }
            } else {
                margin += 0.0309840385f
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.75f) {
                if (!features[4].isNaN() && features[4] < 0.533203125f) {
                    margin += -0.0526797771f
                } else {
                    margin += -0.12258096f
                }
            } else {
                if (!features[4].isNaN() && features[4] < 0.87890625f) {
                    margin += 0.124294937f
                } else {
                    margin += -0.00905244332f
                }
            }
        }
        // Tree 16
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[2].isNaN() && features[2] < 0.0518152267f) {
                margin += -0.00164980476f
            } else {
                if (!features[1].isNaN() && features[1] < 0.840084016f) {
                    margin += 0.107621908f
                } else {
                    margin += 0.0603216588f
                }
            }
        } else {
            if (!features[3].isNaN() && features[3] < 0.00308756996f) {
                if (!features[0].isNaN() && features[0] < 0.21875f) {
                    margin += -0.129866421f
                } else {
                    margin += -0.040075954f
                }
            } else {
                if (!features[4].isNaN() && features[4] < 0.782226562f) {
                    margin += -0.0470462255f
                } else {
                    margin += 0.0677963123f
                }
            }
        }
        // Tree 17
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[2].isNaN() && features[2] < 0.0518152267f) {
                margin += 0.0159520637f
            } else {
                if (!features[3].isNaN() && features[3] < 0.000347442954f) {
                    margin += 0.0154470401f
                } else {
                    margin += 0.102657489f
                }
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.453125f) {
                margin += -0.108814374f
            } else {
                if (!features[0].isNaN() && features[0] < 0.40234375f) {
                    margin += -0.0523808412f
                } else {
                    margin += 0.0202992428f
                }
            }
        }
        // Tree 18
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[2].isNaN() && features[2] < 0.0518152267f) {
                margin += -0.010838273f
            } else {
                if (!features[2].isNaN() && features[2] < 0.977814555f) {
                    margin += 0.0950337574f
                } else {
                    margin += 0.0317375921f
                }
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.453125f) {
                margin += -0.106059007f
            } else {
                if (!features[4].isNaN() && features[4] < 0.509765625f) {
                    margin += 0.0952264443f
                } else {
                    margin += -0.0154646933f
                }
            }
        }
        // Tree 19
        if (!features[4].isNaN() && features[4] < 0.5f) {
            if (!features[0].isNaN() && features[0] < 0.83203125f) {
                if (!features[4].isNaN() && features[4] < 0.458007812f) {
                    margin += -0.0990644172f
                } else {
                    margin += -0.019325994f
                }
            } else {
                margin += -0.0109046837f
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.509765625f) {
                margin += 0.125741169f
            } else {
                if (!features[4].isNaN() && features[4] < 0.75f) {
                    margin += -0.101695739f
                } else {
                    margin += 0.0214538164f
                }
            }
        }
        // Tree 20
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[2].isNaN() && features[2] < 0.0518152267f) {
                margin += -0.000620268052f
            } else {
                if (!features[2].isNaN() && features[2] < 0.977814555f) {
                    margin += 0.0912016481f
                } else {
                    margin += 0.0196258407f
                }
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.75f) {
                if (!features[4].isNaN() && features[4] < 0.533203125f) {
                    margin += -0.040704418f
                } else {
                    margin += -0.115535222f
                }
            } else {
                if (!features[4].isNaN() && features[4] < 0.87890625f) {
                    margin += 0.0958695933f
                } else {
                    margin += -0.0133876419f
                }
            }
        }
        // Tree 21
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[2].isNaN() && features[2] < 0.0518152267f) {
                margin += 0.00961944275f
            } else {
                if (!features[2].isNaN() && features[2] < 0.55512321f) {
                    margin += 0.0999613404f
                } else {
                    margin += 0.0502150953f
                }
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.70703125f) {
                if (!features[2].isNaN() && features[2] < 0.92575866f) {
                    margin += -0.0406695679f
                } else {
                    margin += 0.0371563844f
                }
            } else {
                if (!features[2].isNaN() && features[2] < 0.458656222f) {
                    margin += 0.0566916838f
                } else {
                    margin += -0.0562342778f
                }
            }
        }
        // Tree 22
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[0].isNaN() && features[0] < 0.50390625f) {
                margin += 0.0943024307f
            } else {
                if (!features[3].isNaN() && features[3] < 0.000775020744f) {
                    margin += -0.0381285809f
                } else {
                    margin += 0.0571862273f
                }
            }
        } else {
            if (!features[3].isNaN() && features[3] < 0.00122227706f) {
                if (!features[0].isNaN() && features[0] < 0.28515625f) {
                    margin += -0.013694115f
                } else {
                    margin += -0.082259953f
                }
            } else {
                if (!features[1].isNaN() && features[1] < 0.607739747f) {
                    margin += -0.0408865437f
                } else {
                    margin += 0.022551002f
                }
            }
        }
        // Tree 23
        if (!features[4].isNaN() && features[4] < 0.5f) {
            if (!features[5].isNaN() && features[5] < 1.0f) {
                margin += 0.022697838f
            } else {
                if (!features[4].isNaN() && features[4] < 0.450195312f) {
                    margin += -0.100841723f
                } else {
                    margin += -0.0259870533f
                }
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.509765625f) {
                margin += 0.126727745f
            } else {
                if (!features[4].isNaN() && features[4] < 0.75f) {
                    margin += -0.0998864472f
                } else {
                    margin += 0.0177846011f
                }
            }
        }
        // Tree 24
        if (!features[4].isNaN() && features[4] < 0.5f) {
            if (!features[5].isNaN() && features[5] < 1.0f) {
                margin += 0.0268238969f
            } else {
                if (!features[4].isNaN() && features[4] < 0.46875f) {
                    margin += -0.102342486f
                } else {
                    margin += -0.0277151f
                }
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.509765625f) {
                margin += 0.122099593f
            } else {
                if (!features[4].isNaN() && features[4] < 0.75f) {
                    margin += -0.0744415149f
                } else {
                    margin += 0.0182104968f
                }
            }
        }
        // Tree 25
        if (!features[4].isNaN() && features[4] < 0.44921875f) {
            if (!features[5].isNaN() && features[5] < 1.0f) {
                margin += 0.0235166606f
            } else {
                margin += -0.0972369239f
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.23046875f) {
                if (!features[1].isNaN() && features[1] < 0.593319952f) {
                    margin += 0.0141267702f
                } else {
                    margin += -0.0902927443f
                }
            } else {
                if (!features[5].isNaN() && features[5] < 1.0f) {
                    margin += 0.0727764517f
                } else {
                    margin += 0.0124415131f
                }
            }
        }
        // Tree 26
        if (!features[4].isNaN() && features[4] < 0.453125f) {
            if (!features[0].isNaN() && features[0] < 0.77734375f) {
                margin += -0.0929300487f
            } else {
                margin += -0.0207492467f
            }
        } else {
            if (!features[3].isNaN() && features[3] < 0.0031594797f) {
                if (!features[4].isNaN() && features[4] < 0.509765625f) {
                    margin += 0.0833550692f
                } else {
                    margin += -0.0191407409f
                }
            } else {
                if (!features[1].isNaN() && features[1] < 0.703082442f) {
                    margin += 0.0692378208f
                } else {
                    margin += -0.0015204543f
                }
            }
        }
        // Tree 27
        if (!features[4].isNaN() && features[4] < 0.5f) {
            if (!features[2].isNaN() && features[2] < 0.09869764f) {
                margin += -0.00577616645f
            } else {
                margin += -0.094687365f
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.509765625f) {
                margin += 0.110405818f
            } else {
                if (!features[4].isNaN() && features[4] < 0.75f) {
                    margin += -0.0869672075f
                } else {
                    margin += 0.00953956414f
                }
            }
        }
        // Tree 28
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[2].isNaN() && features[2] < 0.0518152267f) {
                margin += -0.0144813405f
            } else {
                if (!features[2].isNaN() && features[2] < 0.91386652f) {
                    margin += 0.0817160755f
                } else {
                    margin += 0.00742913177f
                }
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.21875f) {
                if (!features[1].isNaN() && features[1] < 0.599956393f) {
                    margin += -0.0334141441f
                } else {
                    margin += -0.11477419f
                }
            } else {
                if (!features[4].isNaN() && features[4] < 0.450195312f) {
                    margin += -0.0927983001f
                } else {
                    margin += 0.00753976405f
                }
            }
        }
        // Tree 29
        if (!features[4].isNaN() && features[4] < 0.5f) {
            if (!features[0].isNaN() && features[0] < 0.71484375f) {
                margin += -0.0900886953f
            } else {
                if (!features[2].isNaN() && features[2] < 0.158265367f) {
                    margin += 0.0475681387f
                } else {
                    margin += -0.0607123375f
                }
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.509765625f) {
                margin += 0.108748734f
            } else {
                if (!features[4].isNaN() && features[4] < 0.75f) {
                    margin += -0.0669047832f
                } else {
                    margin += 0.0149933686f
                }
            }
        }
        // Tree 30
        if (!features[4].isNaN() && features[4] < 0.75f) {
            if (!features[4].isNaN() && features[4] < 0.533203125f) {
                if (!features[4].isNaN() && features[4] < 0.5f) {
                    margin += -0.0663807988f
                } else {
                    margin += 0.0638689324f
                }
            } else {
                if (!features[1].isNaN() && features[1] < 0.827483058f) {
                    margin += -0.124079101f
                } else {
                    margin += 0.00250129937f
                }
            }
        } else {
            if (!features[0].isNaN() && features[0] < 0.26171875f) {
                if (!features[4].isNaN() && features[4] < 1.0f) {
                    margin += 0.0245679207f
                } else {
                    margin += -0.0849954635f
                }
            } else {
                if (!features[3].isNaN() && features[3] < 0.0039858683f) {
                    margin += 0.0109252995f
                } else {
                    margin += 0.0765198842f
                }
            }
        }
        // Tree 31
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[0].isNaN() && features[0] < 0.515625f) {
                margin += 0.0931954607f
            } else {
                if (!features[1].isNaN() && features[1] < 0.867220581f) {
                    margin += 0.0061487346f
                } else {
                    margin += 0.0813953802f
                }
            }
        } else {
            if (!features[3].isNaN() && features[3] < 0.0031594797f) {
                if (!features[2].isNaN() && features[2] < 0.92575866f) {
                    margin += -0.0447495244f
                } else {
                    margin += 0.048854325f
                }
            } else {
                if (!features[1].isNaN() && features[1] < 0.480590165f) {
                    margin += -0.0693900362f
                } else {
                    margin += 0.0542872436f
                }
            }
        }
        // Tree 32
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[1].isNaN() && features[1] < 0.840084016f) {
                margin += 0.0958508924f
            } else {
                if (!features[1].isNaN() && features[1] < 0.894050658f) {
                    margin += -0.0443407707f
                } else {
                    margin += 0.0544700213f
                }
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.453125f) {
                margin += -0.0886523649f
            } else {
                if (!features[0].isNaN() && features[0] < 0.234375f) {
                    margin += -0.0665743724f
                } else {
                    margin += 0.00799957849f
                }
            }
        }
        // Tree 33
        if (!features[0].isNaN() && features[0] < 0.80859375f) {
            if (!features[4].isNaN() && features[4] < 0.453125f) {
                margin += -0.0883086696f
            } else {
                if (!features[4].isNaN() && features[4] < 0.87890625f) {
                    margin += 0.0320291929f
                } else {
                    margin += -0.0226938985f
                }
            }
        } else {
            if (!features[4].isNaN() && features[4] < 0.71875f) {
                if (!features[1].isNaN() && features[1] < 0.81334132f) {
                    margin += -0.0840090439f
                } else {
                    margin += 0.0443401523f
                }
            } else {
                margin += 0.119618371f
            }
        }
        // Tree 34
        if (!features[4].isNaN() && features[4] < 0.87890625f) {
            if (!features[4].isNaN() && features[4] < 0.75f) {
                if (!features[4].isNaN() && features[4] < 0.509765625f) {
                    margin += 0.00975455157f
                } else {
                    margin += -0.0684518442f
                }
            } else {
                if (!features[4].isNaN() && features[4] < 0.8125f) {
                    margin += 0.0666516051f
                } else {
                    margin += 0.130921587f
                }
            }
        } else {
            if (!features[1].isNaN() && features[1] < 0.93133837f) {
                if (!features[0].isNaN() && features[0] < 0.26171875f) {
                    margin += -0.0748260021f
                } else {
                    margin += 0.00862385146f
                }
            } else {
                if (!features[3].isNaN() && features[3] < 0.00151619711f) {
                    margin += -0.111370996f
                } else {
                    margin += 0.0106936563f
                }
            }
        }
        // Tree 35
        if (!features[0].isNaN() && features[0] < 0.8671875f) {
            if (!features[4].isNaN() && features[4] < 0.453125f) {
                margin += -0.0753600225f
            } else {
                if (!features[4].isNaN() && features[4] < 0.509765625f) {
                    margin += 0.083871603f
                } else {
                    margin += -0.0120492475f
                }
            }
        } else {
            if (!features[1].isNaN() && features[1] < 0.64999193f) {
                margin += 0.0203391612f
            } else {
                margin += 0.0980407745f
            }
        }
        // Tree 36
        if (!features[5].isNaN() && features[5] < 1.0f) {
            if (!features[2].isNaN() && features[2] < 0.0518152267f) {
                margin += -0.00817164499f
            } else {
                if (!features[2].isNaN() && features[2] < 0.55512321f) {
                    margin += 0.0895703658f
                } else {
                    margin += 0.0272307098f
                }
            }
        } else {
            if (!features[3].isNaN() && features[3] < 0.00213643746f) {
                if (!features[3].isNaN() && features[3] < 0.00147929788f) {
                    margin += -0.0209277514f
                } else {
                    margin += -0.0709581003f
                }
            } else {
                if (!features[4].isNaN() && features[4] < 0.75f) {
                    margin += -0.0493255481f
                } else {
                    margin += 0.027889628f
                }
            }
        }

        return sigmoid(margin)
    }

    private fun sigmoid(x: Float): Float =
        (1.0 / (1.0 + Math.exp(-x.toDouble()))).toFloat()
}
