package com.onndetect.domain.models

/** Only SUCCESS with a validated release may produce a classified verdict. */
enum class AnalysisStatus { SUCCESS, INCONCLUSIVE, UNSUPPORTED_INPUT, MODEL_FAILURE, CANCELLED }
enum class DetectionVerdict { SYNTHETIC, AUTHENTIC, INCONCLUSIVE }

data class ScanProgress(
    val stage: String,
    val percent: Float,
    val indeterminate: Boolean = false,
    val result: ScanResult? = null
)

data class ScanResult(
    val id: String,
    val scanMode: ScanMode,
    val timestamp: Long,
    val overallScore: Float,
    val isAiGenerated: Boolean,
    val confidence: Float,
    val signals: SignalBreakdown,
    val heatmap: FloatArray?,
    val provenance: ProvenanceInfo?,
    val qualityInfo: QualityInfo,
    val scanDurationMs: Long,
    val status: AnalysisStatus = AnalysisStatus.INCONCLUSIVE,
    val verdict: DetectionVerdict = DetectionVerdict.INCONCLUSIVE,
    val reason: String? = "Detector calibration has not been validated",
    val modelValidated: Boolean = false,
    val modelVersion: String = "unknown",
    val preprocessingVersion: String = "unknown",
    val calibrationVersion: String = "unvalidated",
    val decisionThreshold: Float? = null,
    val analyzedDurationMs: Long = 0,
    val totalDurationMs: Long = 0,
    val analyzedTimestampsMs: List<Long> = emptyList(),
) {
    init {
        require(overallScore.isFinite() && (overallScore == -1f || overallScore in 0f..1f))
        require(confidence.isFinite() && confidence in 0f..1f)
        if (verdict != DetectionVerdict.INCONCLUSIVE) {
            require(status == AnalysisStatus.SUCCESS && modelValidated && decisionThreshold != null)
            require(decisionThreshold.isFinite() && decisionThreshold in 0f..1f && overallScore in 0f..1f)
            require(isAiGenerated == (verdict == DetectionVerdict.SYNTHETIC))
        }
    }
}

/** A failure, missing score, or unvalidated model must always abstain. */
object VerdictPolicy {
    fun decide(score: Float, status: AnalysisStatus, validated: Boolean, threshold: Float?): DetectionVerdict {
        if (status != AnalysisStatus.SUCCESS || !validated || !score.isFinite() || score !in 0f..1f ||
            threshold == null || !threshold.isFinite() || threshold !in 0f..1f) return DetectionVerdict.INCONCLUSIVE
        return if (score >= threshold) DetectionVerdict.SYNTHETIC else DetectionVerdict.AUTHENTIC
    }
}

data class SignalBreakdown(
    val mobileClipScore: Float,
    val srmForensicScore: Float,
    val fftSpectralScore: Float,
    val klDivergenceScore: Float,
    val perturbationScore: Float,
    val audioDspScore: Float,
    val audioModelScore: Float,
    val temporalScore: Float,
    val provenanceScore: Float,
    /** Residual texture proxy, NOT camera PRNU attribution; -1 = unavailable. */
    val prnuScore: Float = -1f
)

data class ProvenanceInfo(
    val hasC2PA: Boolean,
    val hasSynthID: Boolean,
    val exifData: Map<String, String>,
    /** Presence booleans are unverified observations and must not affect verdicts. */
    val verificationStatus: String = "unverified",
)

data class QualityInfo(
    val resolution: Int,
    val isCompressed: Boolean,
    val jpegQuality: Int?,
    val isBlurry: Boolean,
    val isDark: Boolean
)
