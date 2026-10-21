package com.onndetect.detection.fusion

import kotlin.math.ln

/**
 * The fusion's base margin: the slot-6 probe score's logit.
 *
 * ## Why this exists
 *
 * The fusion used to be a plain gradient-boosted tree over ten features, so
 * its output was a STEP FUNCTION: every sample that landed in the same leaf
 * got the same score, and the ranking inside that leaf was destroyed.
 *
 * That was harmless while slot 6 held the small distilled student. It was
 * not harmless once slot 6 held the frozen SigLIP-B/16 tower, because the
 * probe became so much stronger than the other nine slots that the tree
 * diluted it. Measured on the sealed test split:
 *
 * | scorer | AUC | TPR@1%FPR |
 * |---|---|---|
 * | probe alone | 0.8910 | 0.2850 |
 * | plain fusion (depth 3) | 0.8697 | **0.0000** (34 samples tied at the top) |
 * | fusion with the probe as the base margin | **0.8941** | **0.2800** |
 *
 * The fusion's score is what the app compares against its threshold, so the
 * dilution reached the verdict. With the probe as the BASE MARGIN the
 * booster predicts `sigmoid(logit(probe) + tree_correction)`: the tree can
 * only ADD to the probe, so it can never reorder the probe's confident
 * samples against each other.
 *
 * ## The contract with Python
 *
 * `ml/scripts/train_xgboost_meta.py::probe_base_margin` computes exactly
 * this, in float64, and clips the probability to `[1e-6, 1 - 1e-6]` before
 * taking the logit. This object mirrors it, and
 * `FusionBaseMarginTest` pins the two together against reference values
 * computed in Python.
 *
 * ## The unavailable case
 *
 * Slot 6 is NaN when the probe did not run. The margin is then **0**, which
 * is exactly what the training rows with a masked slot 6 carried, so the
 * tree's learned missing branch stays consistent.
 */
object FusionBaseMargin {

    /** The probability clipping applied before the logit, matching Python. */
    const val EPSILON: Float = 1e-6f

    /**
     * The logit of the slot-6 probe score.
     *
     * @param probe the probe's P(AI), or NaN when the probe did not run.
     * @return `log(p / (1 - p))`, or `0f` when the probe is unavailable or
     *   its probability is exactly 0 or 1 (where the logit is not finite).
     */
    fun logit(probe: Float): Float {
        if (!probe.isFinite()) return 0f
        if (probe <= 0f || probe >= 1f) return 0f
        val q = probe.coerceIn(EPSILON, 1f - EPSILON).toDouble()
        return ln(q / (1.0 - q)).toFloat()
    }
}
