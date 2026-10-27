package com.onndetect.detection.provenance

/**
 * Provenance evidence is CATEGORICAL, not probabilistic.
 *
 * A verified C2PA claim, or a detected watermark mark, is evidence that a
 * *specific tool* produced or edited the asset. It is not a probability and
 * it must never be averaged into a score: averaging would let a
 * statistical detector's confidence dilute a cryptographic fact.
 *
 * Absence is NOT evidence of authenticity. Metadata is stripped by
 * re-encoding; marks are stripped by adversaries. The absence of provenance
 * therefore lowers nothing, and the detector's score is unchanged by it.
 */
sealed interface ProvenanceEvidence {

    /** No provenance signal was found. This is not evidence of authenticity. */
    data object Absent : ProvenanceEvidence

    /**
     * A C2PA manifest was located in the asset.
     *
     * [signed] is TRUE only when the signature chain and the binding to
     * the asset's bytes have been verified. A manifest that is merely
     * PRESENT is not a claim: the specification's "exclusion range" allows
     * parts of an asset to sit outside the signed region, and an
     * unverified manifest can be forged.
     */
    data class C2PAClaim(
        val manifestPresent: Boolean,
        val signed: Boolean,
        /** The IPTC digital source type, e.g. `trainedAlgorithmicMedia`. */
        val digitalSourceType: String? = null,
        val ingredientCount: Int = 0,
        val regionsOfInterest: List<String> = emptyList(),
    ) : ProvenanceEvidence {
        /** True only for a VERIFIED claim. */
        val confirmsAiInvolvement: Boolean
            get() = signed && digitalSourceType in AI_SOURCE_TYPES
    }

    /**
     * A watermark mark was detected by a neural watermark detector.
     *
     * [pValue] is the detector's probability-value-style confidence; a
     * detection is only treated as positive evidence below
     * [SIGNIFICANT_P_VALUE].
     */
    data class WatermarkMark(
        val detected: Boolean,
        val kind: String,
        val pValue: Double = 1.0,
    ) : ProvenanceEvidence {
        val confirmsAiInvolvement: Boolean
            get() = detected && pValue < SIGNIFICANT_P_VALUE
    }

    companion object {
        /**
         * Digital source types that mean a generative model was involved.
         * These are the IPTC Photo Metadata standard values.
         */
        val AI_SOURCE_TYPES = setOf(
            "trainedAlgorithmicMedia",
            "compositeWithTrainedAlgorithmicMedia",
            "algorithmicMedia",
            "compositeAlgorithmicMedia",
        )

        /** A mark must be at least this significant to count as evidence. */
        const val SIGNIFICANT_P_VALUE = 1e-4
    }
}

/**
 * How much the verdict rests on.
 *
 * This is separate from [com.onndetect.domain.models.DetectionVerdict]
 * because a verdict says WHAT and this says HOW SURELY - and because a
 * confirmed provenance claim is a different KIND of answer from a
 * statistical score.
 */
enum class VerdictConfidence {
    /** No validated detector applied; diagnostics only. */
    UNVALIDATED,

    /** A statistical detector decided at a calibrated operating point. */
    STATISTICAL,

    /**
     * Cryptographic provenance or a watermark confirmed AI involvement.
     *
     * Reaching this state requires a VERIFIED signature chain or a
     * significant watermark detection. A manifest that is merely present
     * does not qualify.
     */
    PROVENANCE_CONFIRMED,
}
