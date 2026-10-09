package com.onndetect.domain.pipeline

import com.onndetect.domain.models.ProvenanceInfo

/** Content scores and unverified metadata are independent observations. */
class SignalFusion {
    @Suppress("UNUSED_PARAMETER")
    fun fuse(rawScore: Float, provenance: ProvenanceInfo?): Float {
        require(rawScore.isFinite() && rawScore in 0f..1f) { "Invalid model score" }
        // No deployed signature or watermark verifier exists. Presence of a
        // string, metadata, or a caller-supplied flag must not change a score.
        return rawScore
    }
}
