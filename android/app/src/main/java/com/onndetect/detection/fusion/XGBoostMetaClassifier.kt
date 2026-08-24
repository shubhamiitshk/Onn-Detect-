package com.onndetect.detection.fusion

/**
 * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY.
 * Transpiled XGBoost meta-classifier (19 trees, 10 features).
 * Feature order:
 *   0 srm_score, 1 fft_score, 2 kl_score, 3 flatness_score,
 *   4 resolution_norm, 5 is_compressed
 */
class XGBoostMetaClassifier {

    /** @return P(AI-generated) in [0,1]. */
    fun predict(features: FloatArray): Float {
        var margin = 0.003000006039550751f
        // Tree 0
        if (features[6] < 0.800000012f) {
            if (features[2] < 0.0576046333f) {
                if (features[3] < 1.38037831e-05f) {
                    margin += -0.0269267205f
                } else {
                    margin += -0.127518415f
                }
            } else {
                if (features[3] < 1.98326006e-06f) {
                    margin += -0.0223889183f
                } else {
                    margin += -0.155200288f
                }
            }
        } else {
            if (features[3] < 8.65685433e-06f) {
                if (features[2] < 0.437803119f) {
                    margin += 0.179152146f
                } else {
                    margin += 0.127296835f
                }
            } else {
                if (features[2] < 0.0591931753f) {
                    margin += 0.053073436f
                } else {
                    margin += 0.147081837f
                }
            }
        }
        // Tree 1
        if (features[6] < 0.800000012f) {
            if (features[2] < 0.0526733696f) {
                if (features[3] < 1.00958641e-05f) {
                    margin += 0.041758921f
                } else {
                    margin += -0.114779048f
                }
            } else {
                if (features[3] < 1.98326006e-06f) {
                    margin += -0.0211462062f
                } else {
                    margin += -0.140653133f
                }
            }
        } else {
            if (features[3] < 8.65685433e-06f) {
                if (features[2] < 0.437803119f) {
                    margin += 0.161784858f
                } else {
                    margin += 0.113078117f
                }
            } else {
                if (features[2] < 0.0576046333f) {
                    margin += 0.0350592025f
                } else {
                    margin += 0.134427205f
                }
            }
        }
        // Tree 2
        if (features[0] < 0.420579582f) {
            if (features[0] < 0.107302226f) {
                margin += -0.161256164f
            } else {
                if (features[0] < 0.284795046f) {
                    margin += -0.0944881365f
                } else {
                    margin += -0.0319719724f
                }
            }
        } else {
            if (features[0] < 0.813860178f) {
                if (features[3] < 1.57003251e-05f) {
                    margin += 0.0648273751f
                } else {
                    margin += 0.0196033008f
                }
            } else {
                if (features[0] < 0.947903574f) {
                    margin += 0.12065196f
                } else {
                    margin += 0.166466102f
                }
            }
        }
        // Tree 3
        if (features[0] < 0.420579582f) {
            if (features[0] < 0.107302226f) {
                if (features[0] < 0.0850145668f) {
                    margin += -0.154738024f
                } else {
                    margin += -0.115382753f
                }
            } else {
                if (features[0] < 0.268509209f) {
                    margin += -0.0870637819f
                } else {
                    margin += -0.0355084389f
                }
            }
        } else {
            if (features[0] < 0.822823405f) {
                if (features[0] < 0.646425784f) {
                    margin += 0.0068558054f
                } else {
                    margin += 0.0579292588f
                }
            } else {
                if (features[0] < 0.934181929f) {
                    margin += 0.118403293f
                } else {
                    margin += 0.154820576f
                }
            }
        }
        // Tree 4
        if (features[0] < 0.319193482f) {
            if (features[0] < 0.0850145668f) {
                margin += -0.143348679f
            } else {
                if (features[3] < 7.03555179e-06f) {
                    margin += -0.0541802309f
                } else {
                    margin += -0.0898429006f
                }
            }
        } else {
            if (features[0] < 0.816632628f) {
                if (features[0] < 0.646425784f) {
                    margin += 0.00469833752f
                } else {
                    margin += 0.0581291914f
                }
            } else {
                if (features[0] < 0.931741357f) {
                    margin += 0.0968255401f
                } else {
                    margin += 0.144135579f
                }
            }
        }
        // Tree 5
        if (features[0] < 0.646425784f) {
            if (features[0] < 0.133482397f) {
                if (features[0] < 0.0850145668f) {
                    margin += -0.135168627f
                } else {
                    margin += -0.0914103016f
                }
            } else {
                if (features[0] < 0.276217908f) {
                    margin += -0.06941396f
                } else {
                    margin += -0.010837811f
                }
            }
        } else {
            if (features[0] < 0.925641537f) {
                if (features[0] < 0.770036399f) {
                    margin += 0.0455904976f
                } else {
                    margin += 0.0924961269f
                }
            } else {
                if (features[0] < 0.958618104f) {
                    margin += 0.115999691f
                } else {
                    margin += 0.140803978f
                }
            }
        }
        // Tree 6
        if (features[0] < 0.420579582f) {
            if (features[0] < 0.101974532f) {
                if (features[2] < 0.038688302f) {
                    margin += -0.0297227632f
                } else {
                    margin += -0.127412334f
                }
            } else {
                if (features[3] < 1.52944467e-05f) {
                    margin += -0.0349940732f
                } else {
                    margin += -0.0761382505f
                }
            }
        } else {
            if (features[0] < 0.816632628f) {
                if (features[3] < 4.46974445e-06f) {
                    margin += 0.0817058086f
                } else {
                    margin += 0.0185658094f
                }
            } else {
                if (features[0] < 0.958618104f) {
                    margin += 0.1028198f
                } else {
                    margin += 0.133167341f
                }
            }
        }
        // Tree 7
        if (features[0] < 0.420579582f) {
            if (features[0] < 0.133482397f) {
                if (features[0] < 0.0850145668f) {
                    margin += -0.125100613f
                } else {
                    margin += -0.0915093496f
                }
            } else {
                if (features[0] < 0.293696761f) {
                    margin += -0.0541192405f
                } else {
                    margin += -0.0158405602f
                }
            }
        } else {
            if (features[0] < 0.83374542f) {
                if (features[2] < 0.216476038f) {
                    margin += 0.0555810295f
                } else {
                    margin += 0.0127693331f
                }
            } else {
                if (features[0] < 0.958618104f) {
                    margin += 0.097972624f
                } else {
                    margin += 0.129174963f
                }
            }
        }
        // Tree 8
        if (features[0] < 0.319193482f) {
            if (features[0] < 0.0850145668f) {
                if (features[2] < 0.0829163566f) {
                    margin += -0.0842869729f
                } else {
                    margin += -0.121996425f
                }
            } else {
                if (features[2] < 0.103342541f) {
                    margin += -0.00209187134f
                } else {
                    margin += -0.0551671386f
                }
            }
        } else {
            if (features[0] < 0.813860178f) {
                if (features[2] < 0.214328751f) {
                    margin += 0.0339884609f
                } else {
                    margin += 0.00195028249f
                }
            } else {
                if (features[0] < 0.934181929f) {
                    margin += 0.0844822451f
                } else {
                    margin += 0.120849252f
                }
            }
        }
        // Tree 9
        if (features[0] < 0.646425784f) {
            if (features[0] < 0.107302226f) {
                if (features[2] < 0.0829163566f) {
                    margin += -0.0768242329f
                } else {
                    margin += -0.116380326f
                }
            } else {
                if (features[0] < 0.284795046f) {
                    margin += -0.0539943874f
                } else {
                    margin += -0.00579370465f
                }
            }
        } else {
            if (features[0] < 0.934181929f) {
                if (features[2] < 0.122996591f) {
                    margin += 0.00576872332f
                } else {
                    margin += 0.0626297519f
                }
            } else {
                if (features[0] < 0.983518898f) {
                    margin += 0.101833954f
                } else {
                    margin += 0.12373548f
                }
            }
        }
        // Tree 10
        if (features[7] < 0.899999976f) {
            if (features[2] < 0.103342541f) {
                if (features[3] < 5.73996704e-06f) {
                    margin += 0.090452224f
                } else {
                    margin += -0.0432874337f
                }
            } else {
                if (features[3] < 3.35379736e-05f) {
                    margin += -0.0769966468f
                } else {
                    margin += -0.118572406f
                }
            }
        } else {
            if (features[3] < 3.98942211e-06f) {
                margin += 0.115014456f
            } else {
                if (features[2] < 0.12631461f) {
                    margin += 0.0470914468f
                } else {
                    margin += 0.0777293295f
                }
            }
        }
        // Tree 11
        if (features[0] < 0.284795046f) {
            if (features[0] < 0.0850145668f) {
                margin += -0.106914118f
            } else {
                if (features[1] < 0.989600837f) {
                    margin += -0.000621707644f
                } else {
                    margin += -0.0528386235f
                }
            }
        } else {
            if (features[0] < 0.849259198f) {
                if (features[0] < 0.71246922f) {
                    margin += 0.00342487893f
                } else {
                    margin += 0.042300608f
                }
            } else {
                if (features[0] < 0.949537039f) {
                    margin += 0.0699113831f
                } else {
                    margin += 0.111545518f
                }
            }
        }
        // Tree 12
        if (features[0] < 0.65911746f) {
            if (features[0] < 0.0850145668f) {
                margin += -0.105620228f
            } else {
                if (features[0] < 0.259732395f) {
                    margin += -0.0403439254f
                } else {
                    margin += -0.00265548378f
                }
            }
        } else {
            if (features[0] < 0.849259198f) {
                if (features[2] < 0.575717449f) {
                    margin += 0.0411973298f
                } else {
                    margin += -0.0253902413f
                }
            } else {
                if (features[0] < 0.983518898f) {
                    margin += 0.0834633335f
                } else {
                    margin += 0.118379705f
                }
            }
        }
        // Tree 13
        if (features[0] < 0.65911746f) {
            if (features[0] < 0.133482397f) {
                if (features[0] < 0.0204040129f) {
                    margin += -0.11075151f
                } else {
                    margin += -0.0790175274f
                }
            } else {
                if (features[2] < 0.366494983f) {
                    margin += -0.0242913347f
                } else {
                    margin += 0.0150543125f
                }
            }
        } else {
            if (features[0] < 0.934181929f) {
                if (features[2] < 0.0500017852f) {
                    margin += -0.0385340527f
                } else {
                    margin += 0.047074195f
                }
            } else {
                if (features[2] < 0.044093132f) {
                    margin += 0.0341989063f
                } else {
                    margin += 0.105955385f
                }
            }
        }
        // Tree 14
        if (features[0] < 0.259732395f) {
            if (features[0] < 0.0941247493f) {
                if (features[3] < 6.14223563e-06f) {
                    margin += -0.0742234513f
                } else {
                    margin += -0.0998596027f
                }
            } else {
                if (features[0] < 0.219922051f) {
                    margin += -0.0189209878f
                } else {
                    margin += -0.0775937736f
                }
            }
        } else {
            if (features[0] < 0.925641537f) {
                if (features[0] < 0.770036399f) {
                    margin += 0.00941603631f
                } else {
                    margin += 0.0504831672f
                }
            } else {
                if (features[0] < 0.983518898f) {
                    margin += 0.0841571614f
                } else {
                    margin += 0.111202233f
                }
            }
        }
        // Tree 15
        if (features[0] < 0.786524832f) {
            if (features[0] < 0.120803773f) {
                if (features[0] < 0.0850145668f) {
                    margin += -0.0960780755f
                } else {
                    margin += -0.0544679463f
                }
            } else {
                if (features[0] < 0.284795046f) {
                    margin += -0.033612933f
                } else {
                    margin += 0.000698435819f
                }
            }
        } else {
            if (features[0] < 0.949537039f) {
                if (features[1] < 0.925970435f) {
                    margin += 0.0890313536f
                } else {
                    margin += 0.0496684015f
                }
            } else {
                if (features[2] < 0.044093132f) {
                    margin += 0.00251341774f
                } else {
                    margin += 0.107521631f
                }
            }
        }
        // Tree 16
        if (features[0] < 0.65911746f) {
            if (features[0] < 0.0941247493f) {
                if (features[0] < 0.0204040129f) {
                    margin += -0.105271272f
                } else {
                    margin += -0.0748277307f
                }
            } else {
                if (features[3] < 3.45297631e-05f) {
                    margin += -0.0204515364f
                } else {
                    margin += -0.0773538873f
                }
            }
        } else {
            if (features[0] < 0.958618104f) {
                if (features[0] < 0.849259198f) {
                    margin += 0.0261626337f
                } else {
                    margin += 0.0580367856f
                }
            } else {
                if (features[0] < 0.982896388f) {
                    margin += 0.0789956972f
                } else {
                    margin += 0.109075665f
                }
            }
        }
        // Tree 17
        if (features[0] < 0.849259198f) {
            if (features[0] < 0.176155031f) {
                if (features[0] < 0.0611099303f) {
                    margin += -0.090348281f
                } else {
                    margin += -0.0471015088f
                }
            } else {
                if (features[3] < 3.14893396e-06f) {
                    margin += 0.0904668942f
                } else {
                    margin += -0.00558787724f
                }
            }
        } else {
            if (features[0] < 0.949537039f) {
                if (features[0] < 0.917432964f) {
                    margin += 0.0653448924f
                } else {
                    margin += 0.0321797393f
                }
            } else {
                if (features[0] < 0.983518898f) {
                    margin += 0.0789679065f
                } else {
                    margin += 0.107296824f
                }
            }
        }
        // Tree 18
        if (features[0] < 0.133482397f) {
            if (features[0] < 0.0204040129f) {
                if (features[2] < 0.0526733696f) {
                    margin += -0.00809871126f
                } else {
                    margin += -0.0996350273f
                }
            } else {
                if (features[2] < 0.167252839f) {
                    margin += -0.0307704043f
                } else {
                    margin += -0.0662803799f
                }
            }
        } else {
            if (features[0] < 0.925641537f) {
                if (features[0] < 0.138031974f) {
                    margin += 0.14350836f
                } else {
                    margin += 0.0028398342f
                }
            } else {
                if (features[0] < 0.983518898f) {
                    margin += 0.0644107163f
                } else {
                    margin += 0.1039415f
                }
            }
        }

        return sigmoid(margin)
    }

    private fun sigmoid(x: Float): Float =
        (1.0 / (1.0 + Math.exp(-x.toDouble()))).toFloat()
}
