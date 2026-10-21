package com.onndetect.domain.pipeline

/**
 * The single source of truth for the fusion feature vector.
 *
 * The contract used to be duplicated in six places:
 *
 *  - `ml/src/meta_classifier/feature_extractor.py` -> `FEATURE_NAMES`
 *  - `ml/src/meta_classifier/feature_extractor.py` -> `SEMANTIC_PROBE_INDEX`
 *  - `ml/scripts/train_xgboost_meta.py` -> `SHORTCUT_FEATURE_INDICES`
 *  - `android/.../domain/pipeline/DetectionPipeline.kt` -> `floatArrayOf(...)`
 *  - `android/.../detection/video/VideoForensicAnalyzer.kt` -> `floatArrayOf(...)`
 *  - `scripts/check_android_consistency.py` -> `expected_features`
 *
 * Six copies of one list is six chances to drift, and it has already
 * drifted: the video route builds a ten-element array by hand, so any
 * change to the contract silently crashes video scans with
 * `IllegalArgumentException` instead of failing a build.
 *
 * `scripts/check_android_consistency.py` now verifies this object against
 * the Python schema, so a divergence fails the build rather than a scan.
 */
object FusionFeatureSchema {

    /** Number of slots in the fusion feature vector. */
    const val SLOT_COUNT = 10

    // Slot indices. Named so a reorder cannot silently change meaning.
    const val SRM_SCORE = 0
    const val FFT_SCORE = 1
    const val KL_SCORE = 2
    const val FLATNESS_SCORE = 3
    const val RESOLUTION_NORM = 4
    const val IS_COMPRESSED = 5
    const val SEMANTIC_PROBE_SCORE = 6
    const val C2PA_PRESENT = 7
    const val LIP_SYNC_CORR = 8
    const val LANDMARK_JITTER = 9

    /**
     * The only slot permitted to be non-finite.
     *
     * Slot 6 carries the Engine C measurement and is `NaN` when the probe
     * did not run. It must be UNAVAILABLE, never a neutral constant: a
     * tree that splits on `0.1` cannot tell "low anomaly" from "not
     * measured", and the booster only learns a real missing branch if the
     * training rows actually contain NaN there.
     */
    const val OPTIONAL_SLOT = SEMANTIC_PROBE_SCORE

    /** Slots that may be non-finite. */
    val OPTIONAL_SLOTS = intArrayOf(OPTIONAL_SLOT)

    /**
     * Build the feature vector, rejecting a wrong-arity call at the point
     * of construction instead of inside the transpiled tree.
     */
    fun build(vararg values: Float): FloatArray {
        require(values.size == SLOT_COUNT) {
            "Fusion feature vector needs $SLOT_COUNT slots, got ${values.size}"
        }
        return FloatArray(values.size) { values[it] }
    }

    /** True when the vector may legally carry a non-finite value. */
    fun isOptionalSlot(index: Int): Boolean = index == OPTIONAL_SLOT
}
