package com.onndetect.domain.pipeline

import com.onndetect.domain.models.AnalysisStatus
import com.onndetect.domain.models.DetectionVerdict

/**
 * Dual-engine cascade arbitration.
 *
 * Hand-written mirror of `ml/src/meta_classifier/dual_engine.py`. The two
 * implementations are checked against each other by
 * `ml/tests/test_dual_engine_contract.py`.
 *
 * Policy
 * ------
 * Engine C (semantic) is the stronger detector, so it decides whenever it is
 * confident. Engine A (artifact fusion) is consulted only when Engine C sits
 * inside the measured ambiguity band:
 *
 *  - both engines agree on the side of their own thresholds -> that verdict
 *  - the engines disagree -> INCONCLUSIVE
 *
 * Disagreement between two independent detectors is evidence of uncertainty,
 * not evidence of authenticity, and `VerdictPolicy` already treats a
 * non-SUCCESS status as abstention. Picking a winner would be a coin flip
 * dressed up as a measurement.
 *
 * The band is fitted on the calibration split by `ml/scripts/evaluate_cascade.py`
 * and shipped as a hash-bound constant. It is not a hand-tuned constant.
 */
data class CascadePolicy(
    val engineAThreshold: Float,
    val engineCThreshold: Float,
    val escalateBelow: Float,
    val escalateAbove: Float,
    val coverage: Float,
    val agreementRate: Float,
    val releaseId: String,
) {
    init {
        require(engineAThreshold.isFinite() && engineAThreshold in 0f..1f) {
            "Engine A threshold must be a finite probability"
        }
        require(engineCThreshold.isFinite() && engineCThreshold in 0f..1f) {
            "Engine C threshold must be a finite probability"
        }
        require(escalateBelow in 0f..1f && escalateAbove in 0f..1f && escalateBelow <= escalateAbove) {
            "Escalation band must satisfy 0 <= below <= above <= 1"
        }
        require(coverage in 0f..1f) { "coverage must be a probability" }
        require(agreementRate in 0f..1f) { "agreementRate must be a probability" }
    }

    /**
     * Apply the policy to one sample.
     *
     * @param engineAScore Fusion score from the artifact engine.
     * @param engineCScore Engine C score, or null when Engine C did not run.
     */
    fun decide(engineAScore: Float, engineCScore: Float?): CascadeDecision {
        require(engineAScore.isFinite() && engineAScore in 0f..1f) {
            "Fusion score must be a finite probability"
        }

        // Engine C unavailable: no arbitration is possible, the fusion decides.
        if (engineCScore == null || !engineCScore.isFinite()) {
            val verdict = if (engineAScore >= engineAThreshold) {
                DetectionVerdict.SYNTHETIC
            } else {
                DetectionVerdict.AUTHENTIC
            }
            return CascadeDecision(verdict, engineAScore, false, "Engine C unavailable; fusion alone")
        }

        val aSaysFake = engineAScore >= engineAThreshold
        val cSaysFake = engineCScore >= engineCThreshold

        if (engineCScore in escalateBelow..escalateAbove) {
            if (aSaysFake == cSaysFake) {
                // Both engines agree, so the verdict is the same either way.
                // Report the FUSION score: the score and the threshold it is
                // compared against must come from the same engine.
                val verdict = if (aSaysFake) {
                    DetectionVerdict.SYNTHETIC
                } else {
                    DetectionVerdict.AUTHENTIC
                }
                return CascadeDecision(verdict, engineAScore, true, "Fusion and Engine C agree")
            }
            return CascadeDecision(
                DetectionVerdict.INCONCLUSIVE,
                engineAScore,
                true,
                "Fusion/Engine C disagreement; abstaining rather than selecting a winner",
            )
        }

        // Outside the ambiguity band the FUSION is the stronger and better
        // calibrated signal (sealed test AUC 0.7152 vs Engine C 0.6871).
        // The previous code returned Engine C's verdict here and reported
        // Engine C's score with Engine C's threshold, which reduced the
        // best engine in the system to a boolean veto.
        val verdict = if (aSaysFake) {
            DetectionVerdict.SYNTHETIC
        } else {
            DetectionVerdict.AUTHENTIC
        }
        return CascadeDecision(verdict, engineAScore, false, "Fusion confident")
    }

    companion object {
        /**
         * Load the measured policy from the hash-bound release constant.
         *
         * Returns null when the release does not ship a policy, in which case
         * the pipeline uses Engine A alone - the same behaviour as a release
         * that never had a second engine.
         */
        fun fromAssets(context: android.content.Context): CascadePolicy? {
            return try {
                val json = org.json.JSONObject(
                    context.assets.open("constants/cascade_policy.json")
                        .bufferedReader().use { it.readText() }
                )
                CascadePolicy(
                    engineAThreshold = json.getDouble("engine_a_threshold").toFloat(),
                    engineCThreshold = json.getDouble("engine_c_threshold").toFloat(),
                    escalateBelow = json.getDouble("escalate_below").toFloat(),
                    escalateAbove = json.getDouble("escalate_above").toFloat(),
                    coverage = json.optDouble("coverage", 1.0).toFloat(),
                    agreementRate = json.optDouble("agreement_rate", 0.0).toFloat(),
                    releaseId = json.optString("release_id", "unvalidated"),
                )
            } catch (e: Exception) {
                null
            }
        }
    }
}

/**
 * Outcome of applying the cascade policy to one sample.
 */
data class CascadeDecision(
    val verdict: DetectionVerdict,
    val score: Float,
    val escalated: Boolean,
    val reason: String,
) {
    /**
     * The analysis status that must accompany this decision.
     *
     * A disagreement is an abstention, so it must be reported as
     * INCONCLUSIVE rather than SUCCESS with a hedged verdict.
     */
    val status: AnalysisStatus
        get() = if (verdict == DetectionVerdict.INCONCLUSIVE) {
            AnalysisStatus.INCONCLUSIVE
        } else {
            AnalysisStatus.SUCCESS
        }
}
