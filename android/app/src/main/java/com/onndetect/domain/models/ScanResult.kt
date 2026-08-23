package com.onndetect.domain.models

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
    val scanDurationMs: Long
)

data class SignalBreakdown(
    val mobileClipScore: Float,
    val srmForensicScore: Float,
    val fftSpectralScore: Float,
    val klDivergenceScore: Float,
    val perturbationScore: Float,
    val audioDspScore: Float,
    val audioModelScore: Float,
    val temporalScore: Float,
    val provenanceScore: Float
)

data class ProvenanceInfo(
    val hasC2PA: Boolean,
    val hasSynthID: Boolean,
    val exifData: Map<String, String>
)

data class QualityInfo(
    val resolution: Int,
    val isCompressed: Boolean,
    val jpegQuality: Int?,
    val isBlurry: Boolean,
    val isDark: Boolean
)
