package com.onndetect.detection.visual

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.tanh

/**
 * Engine D: On-Demand Second-Stage Forensic VLM Router, Pack Contract & Calibration Layer.
 *
 * Designed in accordance with the 28 October 2026 ONN-DETECT VLM Research Audit:
 * 1. **Explicit Separation of Evidence**:
 *    - Provenance (C2PA/EXIF cryptographic binding) is categorical and never enters statistical scores.
 *    - Specialist detectors (`SigLIP/TAP` + `SRM/PatchCraft`) remain the primary baseline.
 *    - The on-device VLM runs strictly as an optional second-stage deep scan when escalated,
 *      defaulting to `diagnosticOnly = true` (shadow mode) until device validation and
 *      held-out recall gains at fixed 1% / 0.1% FPR are established.
 * 2. **Complete Multimodal Pack Accounting (`language GGUF + mmproj GGUF`)**:
 *    - Accounts for both the language backbone and the required vision encoder/projector (`mmproj`)
 *      component across the prioritized candidate shortlist:
 *        (a) `Qwen/Qwen3.5-2B` (`1,280 MB` Q4_K_M + `668 MB` FP16 mmproj = `1,948 MB`) — First quality candidate
 *        (b) `Qwen/Qwen3.5-0.8B` (`533 MB` Q4_K_M + `205 MB` FP16 mmproj = `738 MB`) — Smaller deployment candidate
 *        (c) `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (`437 MB` Q8_0 + `109 MB` Q8_0 mmproj = `546 MB`) — Compact baseline
 *        (d) `google/gemma-4-E2B-it` — Google LiteRT-LM multimodal alternative
 * 3. **Masked Observable Artifact Axes (`NOT_APPLICABLE` / `NOT_OBSERVABLE`)**:
 *    - Prevents mechanical label inversion (`1 - y`) when scoring auxiliary artifact axes;
 *      axes that do not apply to the image's content domain are explicitly masked (`Float.NaN`).
 */
object OnDemandForensicVlmAnalyzer {

    enum class EscalationMode {
        FAST_EXIT_CONFIDENT,
        MICRO_MOE_ONLY,
        VLM_SINGLE_PASS_LOGIT,
        VLM_FULL_DIALECTICAL_COT,
    }

    enum class AxisObservationStatus {
        OBSERVED_ARTIFACT,
        WITHIN_AUTHENTIC_BOUNDS,
        NOT_APPLICABLE,
        NOT_OBSERVABLE,
    }

    /**
     * Explicit label taxonomy distinguishing computational photography and conventional
     * retouching (Label 0 = Authentic) from generative synthesis and generative region
     * editing (Label 1 = Generative AI).
     */
    enum class LabelTaxonomyCategory(val binaryLabel: Int) {
        /** Raw camera capture or standard phone ISP processing (neural denoising, HDR merge, portrait bokeh). */
        AUTHENTIC_CAMERA_OR_ISP(0),
        /** Non-generative editing (crop, exposure, color grade, resize, recompression) or screenshot of real media. */
        AUTHENTIC_CONVENTIONAL_EDIT_OR_SCREENSHOT(0),
        /** Localized generative manipulation (diffusion/autoregressive inpainting, object insertion/removal, face swap). */
        GENERATIVE_LOCAL_EDIT(1),
        /** Full-image generative synthesis (text-to-image, image-to-image diffusion/autoregressive generation). */
        GENERATIVE_FULL_SYNTHESIS(1),
    }

    /**
     * Measurable crop geometry from the original unresized source dimensions (`sourceWidth x sourceHeight`
     * via `sourceUri` / `BitmapRegionDecoder`). Reports `downscaleRatio` so tests can verify whether
     * small text or local edit regions retain sufficient pixel density before VLM encoding.
     */
    data class OriginalSourceCropSpec(
        val left: Int,
        val top: Int,
        val width: Int,
        val height: Int,
        val targetEncodedEdgePx: Int,
        val downscaleRatio: Float,
    )

    fun computeOriginalSourceCropSpec(
        sourceWidth: Int,
        sourceHeight: Int,
        normX0: Float,
        normY0: Float,
        normX1: Float,
        normY1: Float,
        budget: ImageDetailBudget = ImageDetailBudget(),
    ): OriginalSourceCropSpec {
        require(sourceWidth > 0 && sourceHeight > 0) { "Source dimensions must be positive" }
        val x0 = (normX0.coerceIn(0f, 1f) * sourceWidth).toInt().coerceIn(0, max(0, sourceWidth - 1))
        val y0 = (normY0.coerceIn(0f, 1f) * sourceHeight).toInt().coerceIn(0, max(0, sourceHeight - 1))
        val x1 = (normX1.coerceIn(0f, 1f) * sourceWidth).toInt().coerceIn(x0 + 1, sourceWidth)
        val y1 = (normY1.coerceIn(0f, 1f) * sourceHeight).toInt().coerceIn(y0 + 1, sourceHeight)
        val w = x1 - x0
        val h = y1 - y0
        val maxEdge = max(w, h)
        val targetEdge = min(maxEdge, budget.originalSourceCropMaxEdgePx)
        val ratio = if (targetEdge > 0) maxEdge.toFloat() / targetEdge.toFloat() else 1f
        return OriginalSourceCropSpec(
            left = x0,
            top = y0,
            width = w,
            height = h,
            targetEncodedEdgePx = targetEdge,
            downscaleRatio = ratio,
        )
    }

    /**
     * Published reference artifact configurations (decimal MB, including BOTH language weights
     * and the required vision/projector `mmproj` file for `llama.cpp` `mtmd` / LiteRT-LM).
     * Note: Phone compatibility must be validated by profiling full-process PSS/RSS, native
     * allocations, visual-token buffers, and KV state on target devices rather than Java heap alone.
     */
    enum class Tier2ModelPack(
        val packId: String,
        val hfModelId: String,
        val languageFileMb: Int,
        val visionMmprojFileMb: Int,
        val combinedDownloadMb: Int,
        val initialTargetDeviceRamMb: Int,
    ) {
        NONE_SPECIALISTS_ONLY(
            packId = "specialists_only",
            hfModelId = "none",
            languageFileMb = 0,
            visionMmprojFileMb = 0,
            combinedDownloadMb = 0,
            initialTargetDeviceRamMb = 2048,
        ),
        SMOLVLM2_500M_VIDEO_Q8(
            packId = "smolvlm2_500m_video_q8_0",
            hfModelId = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            languageFileMb = 437,
            visionMmprojFileMb = 109,
            combinedDownloadMb = 546,
            initialTargetDeviceRamMb = 4096,
        ),
        QWEN3_5_0_8B_Q4_K_M(
            packId = "qwen3_5_0_8b_q4_k_m",
            hfModelId = "Qwen/Qwen3.5-0.8B",
            languageFileMb = 533,
            visionMmprojFileMb = 205,
            combinedDownloadMb = 738,
            initialTargetDeviceRamMb = 6144,
        ),
        QWEN3_5_2B_Q4_K_M(
            packId = "qwen3_5_2b_q4_k_m",
            hfModelId = "Qwen/Qwen3.5-2B",
            languageFileMb = 1280,
            visionMmprojFileMb = 668,
            combinedDownloadMb = 1948,
            initialTargetDeviceRamMb = 8192,
        ),
        LFM2_5_VL_1_6B_GGUF(
            packId = "lfm2_5_vl_1_6b_gguf",
            hfModelId = "LiquidAI/LFM2.5-VL-1.6B",
            languageFileMb = 880,
            visionMmprojFileMb = 270,
            combinedDownloadMb = 1150,
            initialTargetDeviceRamMb = 6144,
        ),
        GEMMA_4_E2B_LITERT(
            packId = "gemma_4_e2b_it_litert",
            hfModelId = "google/gemma-4-E2B-it",
            languageFileMb = 1450,
            visionMmprojFileMb = 350,
            combinedDownloadMb = 1800,
            initialTargetDeviceRamMb = 8192,
        );

        /** Backwards-compatible alias for download size in decimal MB. */
        val downloadSizeMb: Int get() = combinedDownloadMb
    }

    /**
     * Separate image detail, crop, and visual-token budgets so small receipt/document text
     * and subtle local edits are extracted from the ORIGINAL source resolution before any
     * global 384-512 px resize.
     */
    data class ImageDetailBudget(
        val globalMaxEdgePx: Int = 512,
        val originalSourceCropMaxEdgePx: Int = 512,
        val maxCropCount: Int = 1,
        val maxVisualTokensTotal: Int = 512,
    )

    /**
     * Learned routing thresholds fitted on held-out calibrated specialist posteriors
     * (rather than unvalidated fixed constants).
     */
    data class LearnedRoutingThresholds(
        val llrUncertaintyHalfwidth: Float = 2.20f,
        val calibratedConflictLogitThreshold: Float = 1.80f,
        val hotspotZThreshold: Float = 1.85f,
        val docTamperThreshold: Float = 0.30f,
        val voiEscalationThreshold: Float = 0.38f,
        val voiFullExplanationThreshold: Float = 0.68f,
    )

    /**
     * Signed local model pack contract for offline user-imported VLM bundles.
     * Because ONN-DETECT forbids `INTERNET` and `ACCESS_NETWORK_STATE`, packs are imported
     * via the system document picker and must validate both language AND vision (`mmproj`)
     * hashes before staging. Text-only packs without `visionMmprojSha256` are rejected.
     */
    data class SignedVlmPackManifest(
        val packId: String,
        val hfModelId: String,
        val languageWeightsSha256: String,
        val visionMmprojSha256: String,
        val languageBytes: Long,
        val visionMmprojBytes: Long,
        val processorVersion: String,
        val chatTemplateId: String,
        val imageResolutionCapPx: Int,
        val maxVisualTokens: Int,
        val publisherSignatureValid: Boolean,
        val diagnosticOnly: Boolean = true,
    )

    fun verifySignedPackContract(manifest: SignedVlmPackManifest): Boolean {
        val hex64 = Regex("^[0-9a-fA-F]{64}$")
        if (!manifest.publisherSignatureValid) return false
        if (!hex64.matches(manifest.languageWeightsSha256)) return false
        // Vision/projector component is mandatory: a text-only GGUF cannot inspect an image
        if (!hex64.matches(manifest.visionMmprojSha256)) return false
        if (manifest.languageBytes < 50L * 1024 * 1024 || manifest.visionMmprojBytes < 10L * 1024 * 1024) return false
        if (manifest.imageResolutionCapPx !in 224..1024) return false
        if (manifest.maxVisualTokens !in 64..2048) return false
        if (manifest.processorVersion.isBlank() || manifest.chatTemplateId.isBlank()) return false
        return true
    }

    /**
     * Selects the recommended candidate pack based on initial product targets:
     * - `Qwen/Qwen3.5-2B` (~1.95 GB combined pack) when a ~2 GB deep-scan pack is allowed (8 GB+ target)
     * - `Qwen/Qwen3.5-0.8B` (~738 MB combined pack) for tighter storage / 6 GB+ target
     * - `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (~546 MB combined pack) as compact baseline
     */
    fun selectOptimalTier2ModelPack(
        totalDeviceRamMb: Int,
        freeStorageMb: Int,
        allowLarge2GbPack: Boolean = false,
    ): Tier2ModelPack = when {
        allowLarge2GbPack && totalDeviceRamMb >= 7800 && freeStorageMb >= 2600 ->
            Tier2ModelPack.QWEN3_5_2B_Q4_K_M
        totalDeviceRamMb >= 5800 && freeStorageMb >= 1100 ->
            Tier2ModelPack.QWEN3_5_0_8B_Q4_K_M
        totalDeviceRamMb >= 3800 && freeStorageMb >= 800 ->
            Tier2ModelPack.SMOLVLM2_500M_VIDEO_Q8
        else ->
            Tier2ModelPack.NONE_SPECIALISTS_ONLY
    }

    /**
     * Projects the VLM's final prefill token hidden state `hLast` (`d_model`) through a
     * supervised `d_model x 6` auxiliary head (`wFakeFlat`, `bFake6d`) and optional authentic
     * reference head (`wRealFlat`, `bReal6d`), masking axes listed in [inapplicableAxes].
     */
    fun projectSinglePassRubricHead(
        hLast: FloatArray,
        wFakeFlat: FloatArray,
        bFake6d: FloatArray,
        wRealFlat: FloatArray,
        bReal6d: FloatArray,
        inapplicableAxes: BooleanArray? = null,
    ): Pair<FloatArray, FloatArray> {
        val d = hLast.size
        require(wFakeFlat.size == d * 6 && wRealFlat.size == d * 6) {
            "Expected rubric projection weights of size ${d * 6}"
        }
        val fakeLogits = FloatArray(6)
        val realLogits = FloatArray(6)
        for (axis in 0 until 6) {
            if (inapplicableAxes != null && axis < inapplicableAxes.size && inapplicableAxes[axis]) {
                fakeLogits[axis] = Float.NaN
                realLogits[axis] = Float.NaN
                continue
            }
            var sumF = bFake6d[axis]
            var sumR = bReal6d[axis]
            var idx = axis
            for (k in 0 until d) {
                val hVal = hLast[k]
                sumF += hVal * wFakeFlat[idx]
                sumR += hVal * wRealFlat[idx]
                idx += 6
            }
            fakeLogits[axis] = sumF
            realLogits[axis] = sumR
        }
        return fakeLogits to realLogits
    }

    data class EscalationRoute(
        val mode: EscalationMode,
        val invokeOnDeviceVlm: Boolean,
        val expectedVoiScore: Float,
        val targetDomain: String,
        val triggerReason: String,
    )

    data class DialecticalVlmOutcome(
        val vlmDialecticalProb: Float,
        val posteriorProb: Float,
        val dominantFlawAxis: String,
        val dialecticalExplanation: String,
        val diagnosticOnly: Boolean = false,
    )

    private const val EPS = 1e-6f

    private fun logit(p: Float): Float {
        val c = p.coerceIn(EPS, 1f - EPS)
        return ln((c / (1f - c)).toDouble()).toFloat()
    }

    private fun sigmoid(x: Float): Float {
        val c = x.coerceIn(-30f, 30f)
        return (1.0 / (1.0 + exp(-c.toDouble()))).toFloat()
    }

    /**
     * Infers the active content domain so inapplicable artifact axes (e.g., typography on a
     * natural face portrait, or anatomical geometry on a text screenshot) can be masked.
     */
    fun inferForensicDomain(
        docTamperScore: Float = 0f,
        hasFaceLandmarks: Boolean = false,
        richPoorContrast: Float = 0.5f,
    ): String = when {
        docTamperScore >= 0.30f -> "document_text"
        hasFaceLandmarks -> "human_face"
        richPoorContrast >= 0.68f -> "object_product"
        richPoorContrast <= 0.35f -> "scene_arch"
        else -> "human_face"
    }

    /**
     * Evaluates whether the scan warrants second-stage VLM escalation based on specialist
     * disagreement, uncertainty near the operating threshold, localized ROI hotspot, or
     * document/screenshot routing.
     */
    fun routeAsPerNeed(
        fastFusionProb: Float,
        decisionThreshold: Float,
        srmCalibratedScore: Float,
        vfmProbeScore: Float?,
        hotspotZScore: Float = 0f,
        activeDomain: String = "human_face",
        docTamperScore: Float = 0f,
        avConflictScore: Float = 0f,
        availableHeapMb: Int = 256,
        requireExplanation: Boolean = false,
    ): EscalationRoute {
        val llrDist = abs(logit(fastFusionProb) - logit(decisionThreshold.coerceIn(0.05f, 0.95f)))
        val boundaryUncertainty = max(0f, 1f - llrDist / 2.20f)

        val zA = logit(srmCalibratedScore)
        val engineConflict = if (vfmProbeScore != null && vfmProbeScore.isFinite()) {
            val zC = logit(vfmProbeScore)
            if (zA * zC < 0f || abs(zA - zC) > 1.80f) {
                (abs(zA - zC) / 3.50f).coerceIn(0f, 1f)
            } else {
                0f
            }
        } else {
            0.45f
        }

        val hotspotNeed = if (hotspotZScore >= 1.85f) {
            ((hotspotZScore - 1.50f) / 2.0f).coerceIn(0f, 1f)
        } else {
            0f
        }

        var domainNeed = 0f
        if (activeDomain == "document_text" || docTamperScore >= 0.35f) {
            domainNeed = max(0.55f, docTamperScore.coerceIn(0f, 1f))
        }
        if (avConflictScore >= 0.35f) {
            domainNeed = max(domainNeed, avConflictScore.coerceIn(0f, 1f))
        }

        val voi = (
            0.42f * boundaryUncertainty +
                0.28f * engineConflict +
                0.18f * hotspotNeed +
                0.22f * domainNeed +
                (if (requireExplanation) 0.35f else 0f)
            ).coerceIn(0f, 1f)

        val reason = when {
            requireExplanation -> "forensic_explanation_requested"
            boundaryUncertainty > 0.45f -> "ambiguous_fast_margin"
            engineConflict > 0.45f -> "cross_engine_disagreement"
            hotspotNeed > 0.15f -> "localized_roi_hotspot"
            domainNeed > 0.30f -> "specialized_domain_$activeDomain"
            else -> "fast_tier_confident"
        }

        if (availableHeapMb < 96) {
            val mode = if (voi >= 0.38f) EscalationMode.MICRO_MOE_ONLY else EscalationMode.FAST_EXIT_CONFIDENT
            return EscalationRoute(mode, false, voi, activeDomain, reason)
        }

        return when {
            requireExplanation || voi >= 0.68f ->
                EscalationRoute(EscalationMode.VLM_FULL_DIALECTICAL_COT, true, voi, activeDomain, reason)
            voi >= 0.38f ->
                EscalationRoute(EscalationMode.VLM_SINGLE_PASS_LOGIT, true, voi, activeDomain, reason)
            else ->
                EscalationRoute(EscalationMode.FAST_EXIT_CONFIDENT, false, voi, activeDomain, "fast_tier_confident")
        }
    }

    /**
     * Combines observable artifact axis logits (`Float.NaN` when `NOT_APPLICABLE` or `NOT_OBSERVABLE`)
     * with the specialist baseline probability. When [diagnosticOnly] is true (shadow evaluation mode),
     * `posteriorProb` preserves `fastFusionProb` so the public detector's operating point is not
     * altered before held-out calibration is verified.
     */
    fun fuseDialecticalLogits(
        fastFusionProb: Float,
        forgeryLogits6d: FloatArray,
        authenticCounterLogits6d: FloatArray,
        roiForgeryLogits6d: FloatArray? = null,
        nullPriorShift6d: FloatArray? = null,
        hotspotZScore: Float = 0f,
        activeDomain: String = "human_face",
        decisionThreshold: Float = 0.5782f,
        diagnosticOnly: Boolean = false,
    ): DialecticalVlmOutcome {
        require(forgeryLogits6d.size == 6 && authenticCounterLogits6d.size == 6) {
            "Rubric requires 6 axis slots (use Float.NaN for inapplicable axes)"
        }
        val rawWeights = when (activeDomain) {
            "document_text" -> floatArrayOf(0.04f, 0.12f, 0.10f, 0.16f, 0.54f, 0.04f)
            "scene_arch" -> floatArrayOf(0.06f, 0.20f, 0.30f, 0.34f, 0.06f, 0.04f)
            "object_product" -> floatArrayOf(0.14f, 0.24f, 0.22f, 0.26f, 0.10f, 0.04f)
            "animal_art" -> floatArrayOf(0.28f, 0.32f, 0.18f, 0.16f, 0.02f, 0.04f)
            else -> floatArrayOf(0.32f, 0.26f, 0.18f, 0.10f, 0.02f, 0.12f)
        }
        val axisNames = arrayOf(
            "anatomical_geometry",
            "surface_texture_fidelity",
            "lighting_shadow_physics",
            "perspective_boundary_seams",
            "typography_document_layout",
            "temporal_audiovisual_sync",
        )

        val roiW = if (hotspotZScore >= 1.85f) {
            min(0.65f, 0.35f + 0.12f * (hotspotZScore - 1.85f))
        } else {
            0.20f
        }

        // Normalize weights strictly over applicable/observable (finite) axes
        var activeWeightSum = 0f
        for (i in 0 until 6) {
            if (forgeryLogits6d[i].isFinite() && authenticCounterLogits6d[i].isFinite()) {
                activeWeightSum += rawWeights[i]
            }
        }
        if (activeWeightSum <= 0f) {
            return DialecticalVlmOutcome(
                vlmDialecticalProb = Float.NaN,
                posteriorProb = fastFusionProb,
                dominantFlawAxis = "not_observable",
                dialecticalExplanation = "No observable VLM axes applicable; specialist baseline retained.",
                diagnosticOnly = diagnosticOnly,
            )
        }

        var weightedSum = 0f
        var bestIdx = 0
        var bestWeightedVal = Float.NEGATIVE_INFINITY
        var peakCalibratedDiff = 0f

        for (i in 0 until 6) {
            if (!forgeryLogits6d[i].isFinite() || !authenticCounterLogits6d[i].isFinite()) continue
            val wNorm = rawWeights[i] / activeWeightSum
            val roiTerm = if (roiForgeryLogits6d != null && i < roiForgeryLogits6d.size && roiForgeryLogits6d[i].isFinite()) {
                roiW * roiForgeryLogits6d[i]
            } else {
                0f
            }
            val prior = if (nullPriorShift6d != null && i < nullPriorShift6d.size && nullPriorShift6d[i].isFinite()) {
                nullPriorShift6d[i]
            } else {
                0f
            }
            val diff = (forgeryLogits6d[i] - authenticCounterLogits6d[i]) + roiTerm - prior
            weightedSum += wNorm * diff
            if (wNorm * diff > bestWeightedVal) {
                bestWeightedVal = wNorm * diff
                bestIdx = i
                peakCalibratedDiff = diff
            }
        }

        val vlmLogit = 0.65f * weightedSum + 0.35f * peakCalibratedDiff
        val vlmProb = sigmoid(1.5f * vlmLogit)

        val zFast = 1.5f * tanh((logit(fastFusionProb) / 1.5f).toDouble()).toFloat()
        val zVlm = 1.8f * tanh((vlmLogit / 1.8f).toDouble()).toFloat()
        val fusedProb = sigmoid(0.62f * zFast + 0.38f * zVlm)
        val postProb = if (diagnosticOnly) fastFusionProb else fusedProb

        val isFake = fusedProb >= decisionThreshold
        val flawAxis = if (isFake) axisNames[bestIdx] else "none"
        val explanation = if (isFake) {
            "<think>[Clue] Domain='$activeDomain', primary anomaly on '$flawAxis' (Z=${"%.2f".format(peakCalibratedDiff)}). " +
                "[Why Fake] Observable regional/structural inconsistency on '$flawAxis'. " +
                "[If Real] Checked against natural optical bokeh, sensor grain, and JPEG quantization.</think><answer>Fake</answer>"
        } else {
            "<think>[Clue] Domain='$activeDomain', observable axes within authentic bounds. " +
                "[Why Real] Lighting, geometry, and surface details are consistent with optical capture. " +
                "[If Fake] High-frequency variations consistent with camera ISP or compression.</think><answer>Real</answer>"
        }

        return DialecticalVlmOutcome(
            vlmDialecticalProb = vlmProb,
            posteriorProb = postProb,
            dominantFlawAxis = flawAxis,
            dialecticalExplanation = explanation,
            diagnosticOnly = diagnosticOnly,
        )
    }
}
