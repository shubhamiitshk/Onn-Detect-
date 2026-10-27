package com.onndetect.domain.models

import com.onndetect.detection.provenance.ProvenanceEvidence
import com.onndetect.detection.provenance.VerdictConfidence

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
    /**
     * How the verdict was reached.
     *
     * [VerdictConfidence.PROVENANCE_CONFIRMED] requires a VERIFIED
     * signature chain or a significant watermark detection. A C2PA
     * manifest that is merely present does not qualify.
     */
    val verdictConfidence: VerdictConfidence = VerdictConfidence.UNVALIDATED,
    /**
     * Provenance evidence, if any was collected.
     *
     * Absence is [ProvenanceEvidence.Absent], which is NOT evidence of
     * authenticity: it simply means nothing was found.
     */
    val provenanceEvidence: ProvenanceEvidence = ProvenanceEvidence.Absent,
    /**
     * The SHA-256 of the MEDIA BYTES that were analysed, or null.
     *
     * It is computed over the imported file, before the import is deleted,
     * and it is what binds a forensic receipt to the content it describes.
     * A receipt without it is not produced: hashing anything else (a scan
     * id, say) would bind the receipt to nothing.
     */
    val mediaSha256: String? = null,
    /**
     * True when an independent ON-DEVICE evaluation verified this release.
     *
     * [modelValidated] means the release is USABLE: its packaged bytes are
     * hash-bound, measured and calibrated. This field distinguishes HOW it
     * was verified. A release that is desktop-verified but not device
     * verified is usable, and must be presented as such rather than as a
     * fully verified release.
     */
    val releaseDeviceValidated: Boolean = false,
    /** Forensic analysis of document/screenshot splicing, ELA, and layout. */
    val documentForgeryReport: com.onndetect.detection.document.DocumentForgeryReport? = null,
    /** Video-KYC (V-CIP) camera injection, virtual camera drivers, and live stream tampering report. */
    val kycInjectionReport: com.onndetect.detection.kyc.KycInjectionReport? = null,
    /** Family Safe-Voice Vault speaker verification and identity matching report. */
    val voiceMatchReport: com.onndetect.detection.audio.VoiceMatchReport? = null,
) {
    init {
        require(overallScore.isNaN() || overallScore == -1f || overallScore in 0f..1f)
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
    /**
     * Slot 6: the frozen SigLIP-B/16 tower's P(AI).
     *
     * It was called `mobileClipScore`, but slot 6 has never carried a
     * MobileCLIP score - MobileCLIP2-S0 measured 0.5145 AUC (chance), and
     * the slot now carries the frozen tower. The persisted JSON key was
     * migrated too; the decoder still reads the old key so earlier history
     * rows decode.
     */
    val vfmProbeScore: Float = Float.NaN,
    val srmForensicScore: Float = Float.NaN,
    val fftSpectralScore: Float = Float.NaN,
    val klDivergenceScore: Float = Float.NaN,
    val perturbationScore: Float = Float.NaN,
    val audioDspScore: Float = Float.NaN,
    val audioModelScore: Float = Float.NaN,
    val temporalScore: Float = Float.NaN,
    val provenanceScore: Float = Float.NaN,
    /** Residual texture proxy, NOT camera PRNU attribution; NaN = unavailable. */
    val prnuScore: Float = Float.NaN,
    /**
     * The XGBoost FUSION score (all measured signals combined).
     *
     * This is the score the cascade policy actually receives as its
     * "Engine A" input, and it is NOT the same thing as
     * [srmForensicScore], which is only the raw SRM CNN output. The
     * cascade telemetry card previously displayed the raw SRM score
     * while the pipeline arbitrated on the fusion score, so the card
     * could contradict the headline verdict. NaN = unavailable.
     */
    val fusionScore: Float = Float.NaN,
    /** BioLip perioral facial landmark acceleration jitter (slot 9); NaN = unavailable/faceless. */
    val landmarkJitter: Float = Float.NaN,
    /** PatchCraft rich-poor log-odds texture contrast (slot 1); NaN = unavailable. */
    val richPoorContrast: Float = Float.NaN,
    /** Localized 8x8 residual hotspot ROI score (slot 2); NaN = unavailable. */
    val roiHotspotScore: Float = Float.NaN,
    /** PatchHead 14x14 spatial response variance (slot 3); NaN = unavailable. */
    val vfmSpatialVariance: Float = Float.NaN,
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
