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

    /**
     * The probability clipping applied before the logit.
     *
     * A DOUBLE, and exactly the trainer's `1e-6`: the clip is applied in
     * double precision so that the saturated cases agree with
     * `probe_base_margin` to the last bit. In float32, `1f - 1e-6f` is
     * 0.9999989867 and the p = 1 case still missed the trainer's margin by
     * 1.3e-2.
     */
    const val EPSILON: Double = 1e-6

    /**
     * The logit of the slot-6 probe score.
     *
     * @param probe the probe's P(AI), or NaN when the probe did not run.
     * @return `log(q / (1 - q))` where `q` is the probability clipped to
     *   `[EPSILON, 1 - EPSILON]`, or `0f` when the probe is unavailable.
     *
     * The clip is the SAME clip the Python trainer applies
     * (`train_xgboost_meta.py::probe_base_margin`), and it is not
     * optional. A float32 sigmoid saturates to exactly 1.0f once the logit
     * exceeds about 17, and the probe's linear head can produce that on a
     * confidently synthetic image. The trainer's margin for such a sample is
     * +13.8155; returning 0 here would make the device's margin 13.8 lower
     * than the one the fusion was fitted with, which is a different verdict
     * for exactly the samples the detector is most confident about.
     *
     * An earlier version short-circuited `p <= 0f || p >= 1f` to `0f`,
     * which meant the device and the trainer disagreed only for saturated
     * probabilities - the one case that matters most.
     */
    fun logit(probe: Float): Float {
        if (!probe.isFinite()) return 0f
        val q = probe.toDouble().coerceIn(EPSILON, 1.0 - EPSILON)
        return ln(q / (1.0 - q)).toFloat()
    }
}
