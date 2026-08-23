package com.onndetect.domain.pipeline

import com.onndetect.domain.models.ProvenanceInfo

class SignalFusion {
    fun fuse(rawScore: Float, provenance: ProvenanceInfo?): Float {
        return applyProvenanceAdjustment(rawScore, provenance)
    }

    private fun applyProvenanceAdjustment(rawScore: Float, provenance: ProvenanceInfo?): Float {
        if (provenance == null) return rawScore
        
        return when {
            // Valid SynthID watermark = definitive AI-generated
            provenance.hasSynthID -> 1.0f
            
            // Valid C2PA from trusted hardware = strong authentic signal
            provenance.hasC2PA -> rawScore * 0.1f
            
            // No provenance metadata = trust the ML signals
            else -> rawScore
        }
    }
}
