package com.onndetect.detection.fusion

/**
 * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY.
 * Transpiled XGBoost meta-classifier (25 trees, 6 features).
 * Feature order:
 *   0 srm_score, 1 fft_score, 2 kl_score, 3 flatness_score,
 *   4 resolution_norm, 5 is_compressed
 */
class XGBoostMetaClassifier {

    /** @return P(AI-generated) in [0,1]. */
    fun predict(features: FloatArray): Float {
        var margin = 0.002999937643588302f
        // Tree 0
        if (features[3] < 7.38901508e-06f) {
            if (features[2] < 0.581763208f) {
                if (features[2] < 0.241235539f) {
                    margin += 0.128426895f
                } else {
                    margin += 0.0408390984f
                }
            } else {
                if (features[3] < 2.38302027e-06f) {
                    margin += 0.0968773365f
                } else {
                    margin += -0.127896667f
                }
            }
        } else {
            if (features[2] < 0.299476147f) {
                if (features[3] < 8.65685433e-06f) {
                    margin += 0.0334146805f
                } else {
                    margin += -0.0371631384f
                }
            } else {
                if (features[3] < 8.54434802e-06f) {
                    margin += -0.0588220768f
                } else {
                    margin += 0.00997624453f
                }
            }
        }
        // Tree 1
        if (features[0] < 0.496535867f) {
            if (features[0] < 0.120803773f) {
                if (features[0] < 0.0850145668f) {
                    margin += -0.186520576f
                } else {
                    margin += -0.14292185f
                }
            } else {
                if (features[0] < 0.319193482f) {
                    margin += -0.114452533f
                } else {
                    margin += -0.0405017547f
                }
            }
        } else {
            if (features[0] < 0.849259198f) {
                if (features[0] < 0.646425784f) {
                    margin += 0.0343818143f
                } else {
                    margin += 0.100277923f
                }
            } else {
                if (features[0] < 0.934181929f) {
                    margin += 0.151187688f
                } else {
                    margin += 0.189751178f
                }
            }
        }
        // Tree 2
        if (features[0] < 0.420579582f) {
            if (features[0] < 0.107302226f) {
                margin += -0.168441042f
            } else {
                if (features[0] < 0.284795046f) {
                    margin += -0.108108357f
                } else {
                    margin += -0.0522056222f
                }
            }
        } else {
            if (features[0] < 0.813860178f) {
                if (features[0] < 0.537656784f) {
                    margin += 0.0155847771f
                } else {
                    margin += 0.0776052326f
                }
            } else {
                if (features[0] < 0.947903574f) {
                    margin += 0.132542893f
                } else {
                    margin += 0.172899798f
                }
            }
        }
        // Tree 3
        if (features[0] < 0.496535867f) {
            if (features[0] < 0.107302226f) {
                margin += -0.157892048f
            } else {
                if (features[0] < 0.293696761f) {
                    margin += -0.0963602662f
                } else {
                    margin += -0.0385016799f
                }
            }
        } else {
            if (features[0] < 0.822823405f) {
                if (features[2] < 0.124666646f) {
                    margin += 0.011822219f
                } else {
                    margin += 0.0676823556f
                }
            } else {
                if (features[0] < 0.934181929f) {
                    margin += 0.128592581f
                } else {
                    margin += 0.159850881f
                }
            }
        }
        // Tree 4
        if (features[0] < 0.431152046f) {
            if (features[0] < 0.0850145668f) {
                if (features[2] < 0.038688302f) {
                    margin += -0.0513198487f
                } else {
                    margin += -0.148271948f
                }
            } else {
                if (features[0] < 0.319193482f) {
                    margin += -0.0930234566f
                } else {
                    margin += -0.0244481787f
                }
            }
        } else {
            if (features[0] < 0.816632628f) {
                if (features[0] < 0.646425784f) {
                    margin += 0.0195863489f
                } else {
                    margin += 0.0718861595f
                }
            } else {
                if (features[0] < 0.931741357f) {
                    margin += 0.106092356f
                } else {
                    margin += 0.148263171f
                }
            }
        }
        // Tree 5
        if (features[0] < 0.537656784f) {
            if (features[0] < 0.133482397f) {
                if (features[0] < 0.0850145668f) {
                    margin += -0.139003247f
                } else {
                    margin += -0.0986348018f
                }
            } else {
                if (features[0] < 0.259732395f) {
                    margin += -0.0823057964f
                } else {
                    margin += -0.0291430932f
                }
            }
        } else {
            if (features[0] < 0.813860178f) {
                if (features[0] < 0.71246922f) {
                    margin += 0.0312621705f
                } else {
                    margin += 0.0788538307f
                }
            } else {
                if (features[0] < 0.949537039f) {
                    margin += 0.10978023f
                } else {
                    margin += 0.143012017f
                }
            }
        }
        // Tree 6
        if (features[0] < 0.496535867f) {
            if (features[0] < 0.133482397f) {
                if (features[0] < 0.0941247493f) {
                    margin += -0.131233275f
                } else {
                    margin += -0.0880899951f
                }
            } else {
                if (features[3] < 1.5167815e-05f) {
                    margin += -0.033066418f
                } else {
                    margin += -0.0738310814f
                }
            }
        } else {
            if (features[0] < 0.816632628f) {
                if (features[3] < 1.05389008e-05f) {
                    margin += 0.0651137084f
                } else {
                    margin += 0.019064283f
                }
            } else {
                if (features[0] < 0.958618104f) {
                    margin += 0.108406805f
                } else {
                    margin += 0.136079058f
                }
            }
        }
        // Tree 7
        if (features[0] < 0.420579582f) {
            if (features[0] < 0.133482397f) {
                if (features[0] < 0.0850145668f) {
                    margin += -0.127782837f
                } else {
                    margin += -0.0958506018f
                }
            } else {
                if (features[0] < 0.293696761f) {
                    margin += -0.0634210631f
                } else {
                    margin += -0.0293072462f
                }
            }
        } else {
            if (features[0] < 0.83374542f) {
                if (features[0] < 0.71246922f) {
                    margin += 0.0253987443f
                } else {
                    margin += 0.062823832f
                }
            } else {
                if (features[0] < 0.958618104f) {
                    margin += 0.102623008f
                } else {
                    margin += 0.131618679f
                }
            }
        }
        // Tree 8
        if (features[0] < 0.420579582f) {
            if (features[0] < 0.107302226f) {
                if (features[2] < 0.0846588239f) {
                    margin += -0.0764836818f
                } else {
                    margin += -0.122422092f
                }
            } else {
                if (features[1] < 0.989600837f) {
                    margin += 0.0119117266f
                } else {
                    margin += -0.0482688621f
                }
            }
        } else {
            if (features[0] < 0.813860178f) {
                if (features[2] < 0.214328751f) {
                    margin += 0.0523873754f
                } else {
                    margin += 0.0073277764f
                }
            } else {
                if (features[0] < 0.958618104f) {
                    margin += 0.095411621f
                } else {
                    margin += 0.125871256f
                }
            }
        }
        // Tree 9
        if (features[0] < 0.420579582f) {
            if (features[0] < 0.0941247493f) {
                margin += -0.11728289f
            } else {
                if (features[0] < 0.284795046f) {
                    margin += -0.0634054393f
                } else {
                    margin += -0.0201477092f
                }
            }
        } else {
            if (features[0] < 0.849259198f) {
                if (features[0] < 0.646425784f) {
                    margin += 0.00307483505f
                } else {
                    margin += 0.0455485582f
                }
            } else {
                if (features[0] < 0.934181929f) {
                    margin += 0.0771021172f
                } else {
                    margin += 0.117140584f
                }
            }
        }
        // Tree 10
        if (features[0] < 0.496535867f) {
            if (features[0] < 0.0850145668f) {
                if (features[0] < 0.0204040129f) {
                    margin += -0.119051434f
                } else {
                    margin += -0.10023246f
                }
            } else {
                if (features[0] < 0.259732395f) {
                    margin += -0.0611049496f
                } else {
                    margin += -0.0144877452f
                }
            }
        } else {
            if (features[0] < 0.849259198f) {
                if (features[0] < 0.646425784f) {
                    margin += 0.00554594724f
                } else {
                    margin += 0.049926348f
                }
            } else {
                if (features[0] < 0.940416157f) {
                    margin += 0.0776923001f
                } else {
                    margin += 0.113828711f
                }
            }
        }
        // Tree 11
        if (features[3] < 3.68161659e-06f) {
            if (features[2] < 0.889103353f) {
                if (features[2] < 0.455948442f) {
                    margin += 0.115346923f
                } else {
                    margin += 0.0532895103f
                }
            } else {
                if (features[3] < 2.23328084e-06f) {
                    margin += 0.0799891874f
                } else {
                    margin += -0.0430025756f
                }
            }
        } else {
            if (features[2] < 0.637013197f) {
                if (features[3] < 5.53663313e-06f) {
                    margin += 0.0544446595f
                } else {
                    margin += -0.00517686177f
                }
            } else {
                if (features[3] < 9.67513824e-06f) {
                    margin += -0.0813046992f
                } else {
                    margin += 0.0272859931f
                }
            }
        }
        // Tree 12
        if (features[3] < 3.68161659e-06f) {
            if (features[2] < 0.490698338f) {
                margin += 0.10853181f
            } else {
                if (features[3] < 2.38302027e-06f) {
                    margin += 0.0545003414f
                } else {
                    margin += -0.026940031f
                }
            }
        } else {
            if (features[2] < 0.613244176f) {
                if (features[3] < 8.49120443e-06f) {
                    margin += 0.0265916418f
                } else {
                    margin += -0.00691775093f
                }
            } else {
                if (features[3] < 9.67513824e-06f) {
                    margin += -0.0681167617f
                } else {
                    margin += 0.021564791f
                }
            }
        }
        // Tree 13
        if (features[0] < 0.646425784f) {
            if (features[0] < 0.133482397f) {
                if (features[0] < 0.0204040129f) {
                    margin += -0.116418079f
                } else {
                    margin += -0.0885829031f
                }
            } else {
                if (features[2] < 0.366494983f) {
                    margin += -0.0320644416f
                } else {
                    margin += 0.00849054195f
                }
            }
        } else {
            if (features[0] < 0.934181929f) {
                if (features[2] < 0.0500017852f) {
                    margin += -0.0351089984f
                } else {
                    margin += 0.0598183051f
                }
            } else {
                if (features[2] < 0.044093132f) {
                    margin += 0.0385770388f
                } else {
                    margin += 0.111954451f
                }
            }
        }
        // Tree 14
        if (features[1] < 0.990830123f) {
            if (features[3] < 2.07085923e-05f) {
                if (features[1] < 0.90307343f) {
                    margin += 0.0759261921f
                } else {
                    margin += 0.0171416309f
                }
            } else {
                if (features[2] < 0.11207059f) {
                    margin += 0.0108973961f
                } else {
                    margin += 0.0910727829f
                }
            }
        } else {
            if (features[2] < 0.922628403f) {
                if (features[3] < 3.39467579e-06f) {
                    margin += 0.100403868f
                } else {
                    margin += -0.000952414353f
                }
            } else {
                if (features[3] < 1.33299845e-05f) {
                    margin += -0.077054657f
                } else {
                    margin += 0.0180867221f
                }
            }
        }
        // Tree 15
        if (features[0] < 0.646425784f) {
            if (features[0] < 0.120803773f) {
                if (features[0] < 0.0850145668f) {
                    margin += -0.105569862f
                } else {
                    margin += -0.0636003539f
                }
            } else {
                if (features[0] < 0.284795046f) {
                    margin += -0.0502954237f
                } else {
                    margin += -0.00326841394f
                }
            }
        } else {
            if (features[0] < 0.934181929f) {
                if (features[0] < 0.786524832f) {
                    margin += 0.0259443559f
                } else {
                    margin += 0.0665718764f
                }
            } else {
                if (features[2] < 0.044093132f) {
                    margin += 0.0242959f
                } else {
                    margin += 0.111086503f
                }
            }
        }
        // Tree 16
        if (features[0] < 0.65911746f) {
            if (features[0] < 0.0941247493f) {
                if (features[0] < 0.0204040129f) {
                    margin += -0.111796312f
                } else {
                    margin += -0.087160565f
                }
            } else {
                if (features[0] < 0.259732395f) {
                    margin += -0.0493812114f
                } else {
                    margin += -0.0163681079f
                }
            }
        } else {
            if (features[0] < 0.958618104f) {
                if (features[0] < 0.849259198f) {
                    margin += 0.0409804173f
                } else {
                    margin += 0.0706628934f
                }
            } else {
                if (features[2] < 0.044093132f) {
                    margin += 0.0268814098f
                } else {
                    margin += 0.108316444f
                }
            }
        }
        // Tree 17
        if (features[0] < 0.496535867f) {
            if (features[0] < 0.0850145668f) {
                if (features[2] < 0.19725354f) {
                    margin += -0.0784630999f
                } else {
                    margin += -0.105834059f
                }
            } else {
                if (features[2] < 0.0576046333f) {
                    margin += 0.0936435983f
                } else {
                    margin += -0.0415578596f
                }
            }
        } else {
            if (features[0] < 0.849259198f) {
                if (features[3] < 5.53663313e-06f) {
                    margin += 0.0671630129f
                } else {
                    margin += 0.0129219992f
                }
            } else {
                if (features[0] < 0.949537039f) {
                    margin += 0.0643075779f
                } else {
                    margin += 0.104007877f
                }
            }
        }
        // Tree 18
        if (features[0] < 0.420579582f) {
            if (features[0] < 0.101974532f) {
                if (features[0] < 0.0204040129f) {
                    margin += -0.104376726f
                } else {
                    margin += -0.0744625777f
                }
            } else {
                if (features[2] < 0.0858865604f) {
                    margin += 0.0473569632f
                } else {
                    margin += -0.0293532014f
                }
            }
        } else {
            if (features[0] < 0.925641537f) {
                if (features[2] < 0.229605138f) {
                    margin += 0.0400108434f
                } else {
                    margin += 0.00946796779f
                }
            } else {
                if (features[0] < 0.983518898f) {
                    margin += 0.0738227367f
                } else {
                    margin += 0.109100536f
                }
            }
        }
        // Tree 19
        if (features[0] < 0.259732395f) {
            if (features[0] < 0.0850145668f) {
                if (features[0] < 0.0204040129f) {
                    margin += -0.106600679f
                } else {
                    margin += -0.0796269402f
                }
            } else {
                if (features[0] < 0.219922051f) {
                    margin += -0.0217014682f
                } else {
                    margin += -0.0891151577f
                }
            }
        } else {
            if (features[0] < 0.934181929f) {
                if (features[0] < 0.71246922f) {
                    margin += 0.00549929542f
                } else {
                    margin += 0.0367073715f
                }
            } else {
                if (features[0] < 0.983518898f) {
                    margin += 0.0785099417f
                } else {
                    margin += 0.108846642f
                }
            }
        }
        // Tree 20
        if (features[0] < 0.319193482f) {
            if (features[0] < 0.0850145668f) {
                if (features[3] < 1.13259584e-05f) {
                    margin += -0.0754856691f
                } else {
                    margin += -0.101503268f
                }
            } else {
                if (features[3] < 7.03555179e-06f) {
                    margin += 0.000319691724f
                } else {
                    margin += -0.0357843265f
                }
            }
        } else {
            if (features[0] < 0.849259198f) {
                if (features[2] < 0.217452049f) {
                    margin += 0.037146125f
                } else {
                    margin += -0.00797282625f
                }
            } else {
                if (features[0] < 0.983518898f) {
                    margin += 0.06629657f
                } else {
                    margin += 0.10915935f
                }
            }
        }
        // Tree 21
        if (features[0] < 0.496535867f) {
            if (features[0] < 0.0850145668f) {
                if (features[2] < 0.0829163566f) {
                    margin += -0.0361382551f
                } else {
                    margin += -0.0897291079f
                }
            } else {
                if (features[2] < 0.103342541f) {
                    margin += 0.0248298049f
                } else {
                    margin += -0.0349722914f
                }
            }
        } else {
            if (features[0] < 0.940416157f) {
                if (features[2] < 0.099193871f) {
                    margin += -0.0233693626f
                } else {
                    margin += 0.0332781859f
                }
            } else {
                if (features[0] < 0.983518898f) {
                    margin += 0.0665401891f
                } else {
                    margin += 0.103228427f
                }
            }
        }
        // Tree 22
        if (features[0] < 0.420579582f) {
            if (features[0] < 0.0850145668f) {
                if (features[3] < 4.42993223e-05f) {
                    margin += -0.0901839286f
                } else {
                    margin += -0.0148850707f
                }
            } else {
                if (features[3] < 5.78980007e-06f) {
                    margin += 0.0342426933f
                } else {
                    margin += -0.0326661989f
                }
            }
        } else {
            if (features[0] < 0.925641537f) {
                if (features[3] < 3.68161659e-06f) {
                    margin += 0.108266294f
                } else {
                    margin += 0.0147273643f
                }
            } else {
                if (features[0] < 0.983518898f) {
                    margin += 0.0634727478f
                } else {
                    margin += 0.101716064f
                }
            }
        }
        // Tree 23
        if (features[0] < 0.347786874f) {
            if (features[0] < 0.0850145668f) {
                if (features[0] < 0.0221102536f) {
                    margin += -0.096691668f
                } else {
                    margin += -0.062340375f
                }
            } else {
                if (features[2] < 0.366494983f) {
                    margin += -0.0322484858f
                } else {
                    margin += 0.00756094232f
                }
            }
        } else {
            if (features[0] < 0.940416157f) {
                if (features[2] < 0.229605138f) {
                    margin += 0.0339852162f
                } else {
                    margin += 0.0017872086f
                }
            } else {
                if (features[2] < 0.044093132f) {
                    margin += -0.00678118458f
                } else {
                    margin += 0.090404816f
                }
            }
        }
        // Tree 24
        if (features[0] < 0.259732395f) {
            if (features[0] < 0.0850145668f) {
                if (features[2] < 0.106917351f) {
                    margin += -0.0386873968f
                } else {
                    margin += -0.0835464969f
                }
            } else {
                if (features[0] < 0.219922051f) {
                    margin += -0.0253187511f
                } else {
                    margin += -0.0823388323f
                }
            }
        } else {
            if (features[0] < 0.958618104f) {
                if (features[0] < 0.71246922f) {
                    margin += 0.00123519707f
                } else {
                    margin += 0.0282409936f
                }
            } else {
                if (features[0] < 0.983518898f) {
                    margin += 0.0661245137f
                } else {
                    margin += 0.101852074f
                }
            }
        }

        return sigmoid(margin)
    }

    private fun sigmoid(x: Float): Float =
        (1.0 / (1.0 + Math.exp(-x.toDouble()))).toFloat()
}
