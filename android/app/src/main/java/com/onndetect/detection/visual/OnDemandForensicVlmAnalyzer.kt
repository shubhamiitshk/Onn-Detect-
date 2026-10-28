package com.onndetect.detection.visual

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapRegionDecoder
import android.graphics.Rect
import android.net.Uri
import com.onndetect.inference.LiteRTInferenceEngine
import org.json.JSONObject
import java.io.File
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.tanh

/**
 * Second-Stage On-Device Forensic VLM Escalation & Unified Policy/Head Engine (`Engine D`).
 *
 * Implements the unified training/serving contract paired with `ml/scripts/train_forensic_vlm_colab_t4.py`:
 *
 * 1. **Versioned Policy & 13-Output Head Contract (`PolicyBundle`)**:
 *    - Consumes `onn-forensic-vlm-policy-v1` JSON exported by `train_forensic_vlm_colab_t4.py`.
 *    - Shares the exact 6 observable artifact axes (`RUBRIC_AXES`), routing equations, and
 *      13-output linear projection (`classLogit` + 6 `forgeryLogits6d` + 6 `authenticCounterLogits6d`).
 *
 * 2. **Feasibility-Checked Routing (`routeWithPolicy` / `routeAsPerNeed`)**:
 *    - Uses the same log-odds uncertainty, calibrated cross-engine disagreement, and ROI hotspot
 *      equations as Python's `evaluate_frozen_routing_on_test`.
 *    - Honors `policy.feasible == false` by never escalating when the routing budget was infeasible.
 *
 * 3. **Exclusive Memory Lease & Original-Source ROI Cropping (`executeEscalatedDeepScan`)**:
 *    - Acquires `LiteRTInferenceEngine.acquireExclusiveVlmSession()` so `siglip_probe_int8.tflite`
 *      is evicted and blocked from reloading for the entire VLM session lifetime.
 *    - Decodes the high-resolution ROI crop directly from `sourceUri` via `BitmapRegionDecoder`
 *      using `computeOriginalSourceCropSpec(...)` and records `downscaleRatio`.
 *
 * 4. **Evidence-Grounded Explanations**:
 *    - Reports only the axes that were actually evaluated (finite in `forgeryLogits6d` and
 *      `authenticCounterLogits6d`) and their measured numeric log-odds differentials, never
 *      emitting hardcoded claims for unperformed checks.
 */
object OnDemandForensicVlmAnalyzer {

    const val POLICY_SCHEMA_VERSION = "onn-forensic-vlm-policy-v1"

    /**
     * Explicit experimental feature flag for invoking the Tier-2 on-device VLM cascade
     * inside `DetectionPipeline`. Disabled by default until a device-validated VLM pack
     * is installed and promoted.
     */
    @Volatile
    var experimentalVlmCascadeEnabled: Boolean = false

    /**
     * When true, `DetectionPipeline` records the escalated VLM outcome as telemetry on
     * `ScanResult.forensicVlmOutcome` without overriding the calibrated specialist verdict.
     */
    @Volatile
    var experimentalDiagnosticOnly: Boolean = true

    /**
     * Optional runtime prefill backend (`llama.cpp` `mtmd` JNI, LiteRT-LM session, or test fixture).
     */
    @Volatile
    var activePrefillBackend: VlmPrefillBackend? = null

    /**
     * Optional in-memory override for the learned `PolicyBundle` (used by tests or dynamic pack loaders).
     */
    @Volatile
    var overridePolicyBundle: PolicyBundle? = null

    /**
     * Official independently trusted publisher Ed25519 public key hex.
     * Production packs must be signed by this key; self-signed packs are strictly rejected.
     */
    const val OFFICIAL_PUBLISHER_PUBLIC_KEY_HEX =
        "c0776fc7a2844b40915c13550a1dc428a36ced6cd5b27f4d5cdbe8d19cca8a3c"

    /**
     * Pinned publisher Ed25519 public key hex for validating VLM model packs.
     * Defaults to [OFFICIAL_PUBLISHER_PUBLIC_KEY_HEX] for production integrity.
     */
    @Volatile
    var pinnedPublisherPublicKeyHex: String? = OFFICIAL_PUBLISHER_PUBLIC_KEY_HEX

    val RUBRIC_AXES: List<String> = listOf(
        "anatomical_geometry",
        "physics_illumination",
        "surface_micro_texture",
        "boundary_halo_seam",
        "typography_document_layout",
        "temporal_kinematic_consistency",
    )

    enum class EscalationMode {
        FAST_EXIT_CONFIDENT,
        MICRO_MOE_ONLY,
        VLM_SINGLE_PASS_LOGIT,
        VLM_FULL_DIALECTICAL_COT,
    }

    /**
     * Explicit binary label taxonomy so ordinary phone camera ISP processing is never
     * conflated with generative AI synthesis or generative editing.
     */
    enum class LabelTaxonomyCategory(val binaryLabel: Int) {
        AUTHENTIC_CAMERA_OR_ISP(0),
        AUTHENTIC_CONVENTIONAL_EDIT_OR_SCREENSHOT(0),
        GENERATIVE_LOCAL_EDIT(1),
        GENERATIVE_FULL_SYNTHESIS(1),
    }

    /**
     * Measurable crop geometry from the original unresized source dimensions (`sourceWidth x sourceHeight`
     * via `sourceUri` / `BitmapRegionDecoder`).
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
            initialTargetDeviceRamMb = 0,
        ),
        SMOLVLM2_500M_VIDEO_Q8(
            packId = "smolvlm2_500m_video_instruct_q8_0",
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
        LFM2_5_VL_1_6B_Q4(
            packId = "lfm2_5_vl_1_6b_q4",
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

        val downloadSizeMb: Int get() = combinedDownloadMb
    }

    data class ImageDetailBudget(
        val globalMaxEdgePx: Int = 512,
        val originalSourceCropMaxEdgePx: Int = 512,
        val maxCropCount: Int = 1,
        val maxVisualTokensTotal: Int = 512,
    )

    /**
     * Versioned routing policy matching Python's `learn_routing_thresholds_on_held_out` and
     * `evaluate_frozen_routing_on_test`.
     */
    data class RoutingPolicyConfig(
        val feasible: Boolean = true,
        val decisionThresholdDev: Float = 0.5782f,
        val llrUncertaintyHalfwidth: Float = 0.65f,
        val calibratedConflictLogitThreshold: Float = 1.80f,
        val hotspotZThreshold: Float = 1.85f,
        val docTamperThreshold: Float = 0.48f,
    )

    /**
     * Unified 13-output head contract matching Python's `MaskedAuxiliaryForensicHead.to_contract_dict()`.
     */
    data class HeadContractWeights(
        val hiddenDim: Int,
        val axes: List<String>,
        val axesSupervised: List<Boolean> = List(6) { true },
        val wClass: FloatArray,
        val bClass: Float,
        val wForgeryAxes: Array<FloatArray>,
        val bForgeryAxes: FloatArray,
        val wCounterAxes: Array<FloatArray>,
        val bCounterAxes: FloatArray,
    ) {
        init {
            require(hiddenDim > 0) { "hiddenDim must be positive" }
            require(axes.size == 6) { "Expected 6 rubric axes, got ${axes.size}" }
            require(wClass.size == hiddenDim) { "wClass size mismatch" }
            require(wForgeryAxes.size == 6 && bForgeryAxes.size == 6) { "forgery axes size mismatch" }
            require(wCounterAxes.size == 6 && bCounterAxes.size == 6) { "counter axes size mismatch" }
        }
    }

    data class HeadPrediction(
        val classLogit: Float,
        val classProb: Float,
        val forgeryLogits6d: FloatArray,
        val authenticCounterLogits6d: FloatArray,
    )

    data class VlmCalibrationConfig(
        val decisionThresholdDev: Float = 0.50f,
        val thresholdAt1PctFpr: Float = 0.50f,
        val thresholdAt5PctFpr: Float = 0.50f,
        val thresholdBalanced: Float = 0.50f,
    )

    data class PolicyBundle(
        val schemaVersion: String,
        val modelId: String,
        val routingPolicy: RoutingPolicyConfig,
        val headContract: HeadContractWeights?,
        val vlmCalibration: VlmCalibrationConfig = VlmCalibrationConfig(),
        val diagnosticOnly: Boolean = false,
    ) {
        companion object {
            fun fromJson(jsonString: String): PolicyBundle {
                val root = JSONObject(jsonString)
                val schema = root.getString("schema_version")
                require(schema == POLICY_SCHEMA_VERSION) {
                    "Unsupported forensic VLM policy schema '$schema' (expected '$POLICY_SCHEMA_VERSION')"
                }
                val modelId = root.optString("model_id", "unknown")
                val rp = root.getJSONObject("routing_policy")
                val routing = RoutingPolicyConfig(
                    feasible = rp.optBoolean("feasible", true),
                    decisionThresholdDev = rp.getDouble("decision_threshold_dev").toFloat(),
                    llrUncertaintyHalfwidth = rp.getDouble("llr_uncertainty_halfwidth").toFloat(),
                    calibratedConflictLogitThreshold = rp.getDouble("calibrated_conflict_logit_threshold").toFloat(),
                    hotspotZThreshold = rp.getDouble("hotspot_z_threshold").toFloat(),
                    docTamperThreshold = rp.optDouble("doc_tamper_threshold", 0.48).toFloat(),
                )
                val hcObj = root.optJSONObject("head_contract")
                val head = if (hcObj != null) {
                    val hiddenDim = hcObj.getInt("hidden_dim")
                    val axesArr = hcObj.getJSONArray("axes")
                    val axes = List(axesArr.length()) { axesArr.getString(it) }
                    val wClassArr = hcObj.getJSONArray("w_class")
                    val wClass = FloatArray(wClassArr.length()) { wClassArr.getDouble(it).toFloat() }
                    val bClass = hcObj.getDouble("b_class").toFloat()

                    val wForgeryOuter = hcObj.getJSONArray("w_forgery_axes")
                    val wForgery = Array(6) { axisIdx ->
                        val row = wForgeryOuter.getJSONArray(axisIdx)
                        FloatArray(row.length()) { row.getDouble(it).toFloat() }
                    }
                    val bForgeryArr = hcObj.getJSONArray("b_forgery_axes")
                    val bForgery = FloatArray(6) { bForgeryArr.getDouble(it).toFloat() }

                    val wCounterOuter = hcObj.getJSONArray("w_counter_axes")
                    val wCounter = Array(6) { axisIdx ->
                        val row = wCounterOuter.getJSONArray(axisIdx)
                        FloatArray(row.length()) { row.getDouble(it).toFloat() }
                    }
                    val bCounterArr = hcObj.getJSONArray("b_counter_axes")
                    val bCounter = FloatArray(6) { bCounterArr.getDouble(it).toFloat() }

                    val axesSupArr = hcObj.optJSONArray("axes_supervised")
                    val axesSupervised = if (axesSupArr != null) {
                        List(axesSupArr.length()) { axesSupArr.getBoolean(it) }
                    } else {
                        List(6) { i ->
                            wForgery[i].any { abs(it) > 1e-6f } || wCounter[i].any { abs(it) > 1e-6f }
                        }
                    }

                    HeadContractWeights(
                        hiddenDim = hiddenDim,
                        axes = axes,
                        axesSupervised = axesSupervised,
                        wClass = wClass,
                        bClass = bClass,
                        wForgeryAxes = wForgery,
                        bForgeryAxes = bForgery,
                        wCounterAxes = wCounter,
                        bCounterAxes = bCounter,
                    )
                } else {
                    null
                }
                val vlmCalObj = root.optJSONObject("vlm_calibration")
                val vlmCalibration = if (vlmCalObj != null) {
                    VlmCalibrationConfig(
                        decisionThresholdDev = vlmCalObj.optDouble("decision_threshold_dev", 0.50).toFloat(),
                        thresholdAt1PctFpr = vlmCalObj.optDouble("threshold_at_1pct_fpr", 0.50).toFloat(),
                        thresholdAt5PctFpr = vlmCalObj.optDouble("threshold_at_5pct_fpr", 0.50).toFloat(),
                        thresholdBalanced = vlmCalObj.optDouble("threshold_balanced", 0.50).toFloat(),
                    )
                } else {
                    VlmCalibrationConfig()
                }
                val diagnosticOnly = root.optBoolean("diagnostic_only", true)
                return PolicyBundle(
                    schemaVersion = schema,
                    modelId = modelId,
                    routingPolicy = routing,
                    headContract = head,
                    vlmCalibration = vlmCalibration,
                    diagnosticOnly = diagnosticOnly,
                )
            }
        }
    }

    /**
     * Projects the VLM's final prefill hidden state `hLast` through [head], matching Python's
     * `MaskedAuxiliaryForensicHead.predict` to within floating-point precision.
     */
    fun projectWithHeadContract(
        hLast: FloatArray,
        head: HeadContractWeights,
        applicableMask6d: BooleanArray? = null,
    ): HeadPrediction {
        require(hLast.size == head.hiddenDim) {
            "Hidden state dimension ${hLast.size} does not match head contract ${head.hiddenDim}"
        }
        var cLogit = head.bClass
        for (k in 0 until head.hiddenDim) {
            cLogit += hLast[k] * head.wClass[k]
        }
        val cProb = sigmoid(cLogit)
        val forgeryLogits = FloatArray(6)
        val counterLogits = FloatArray(6)
        for (axis in 0 until 6) {
            val isSupervised = axis < head.axesSupervised.size && head.axesSupervised[axis]
            val isApplicable = applicableMask6d == null || (axis < applicableMask6d.size && applicableMask6d[axis])
            if (!isSupervised || !isApplicable) {
                forgeryLogits[axis] = Float.NaN
                counterLogits[axis] = Float.NaN
                continue
            }
            var sumF = head.bForgeryAxes[axis]
            var sumC = head.bCounterAxes[axis]
            val wf = head.wForgeryAxes[axis]
            val wc = head.wCounterAxes[axis]
            for (k in 0 until head.hiddenDim) {
                val hv = hLast[k]
                sumF += hv * wf[k]
                sumC += hv * wc[k]
            }
            forgeryLogits[axis] = sumF
            counterLogits[axis] = sumC
        }
        return HeadPrediction(
            classLogit = cLogit,
            classProb = cProb,
            forgeryLogits6d = forgeryLogits,
            authenticCounterLogits6d = counterLogits,
        )
    }

    /**
     * Runtime interface for an on-device VLM prefill engine (e.g., `llama.cpp` `mtmd` JNI
     * or LiteRT-LM multimodal session). Separating the prefill runtime behind this interface
     * allows `DetectionPipeline` and unit tests to execute the full session-lease, original-crop,
     * projection, and dialectical fusion flow deterministically.
     */
    fun interface VlmPrefillBackend {
        fun extractFinalHiddenState(
            fullBitmap: Bitmap,
            roiCropBitmap: Bitmap?,
            cropSpec: OriginalSourceCropSpec?,
            activeDomain: String,
        ): FloatArray
    }

    data class SignedVlmPackManifest(
        val packId: String,
        val hfModelId: String,
        val languageWeightsSha256: String,
        val visionMmprojSha256: String,
        val policySha256: String = "",
        val languageBytes: Long,
        val visionMmprojBytes: Long,
        val processorVersion: String,
        val chatTemplateId: String,
        val imageResolutionCapPx: Int,
        val maxVisualTokens: Int,
        val publisherPublicKeyHex: String = "",
        val publisherSignatureHex: String = "",
        val publisherSignatureValid: Boolean = false,
        val diagnosticOnly: Boolean = true,
    ) {
        companion object {
            fun fromJson(jsonStr: String): SignedVlmPackManifest {
                val obj = JSONObject(jsonStr)
                return SignedVlmPackManifest(
                    packId = obj.optString("pack_id", ""),
                    hfModelId = obj.optString("hf_model_id", ""),
                    languageWeightsSha256 = obj.optString("language_weights_sha256", ""),
                    visionMmprojSha256 = obj.optString("vision_mmproj_sha256", ""),
                    policySha256 = obj.optString("policy_sha256", ""),
                    languageBytes = obj.optLong("language_bytes", 0L),
                    visionMmprojBytes = obj.optLong("vision_mmproj_bytes", 0L),
                    processorVersion = obj.optString("processor_version", ""),
                    chatTemplateId = obj.optString("chat_template_id", ""),
                    imageResolutionCapPx = obj.optInt("image_resolution_cap_px", 512),
                    maxVisualTokens = obj.optInt("max_visual_tokens", 512),
                    publisherPublicKeyHex = obj.optString("publisher_public_key_hex", ""),
                    publisherSignatureHex = obj.optString("publisher_signature_hex", ""),
                    publisherSignatureValid = obj.optBoolean("publisher_signature_valid", false),
                    diagnosticOnly = obj.optBoolean("diagnostic_only", true),
                )
            }
        }
    }

    const val MIN_VLM_SUPPORTED_SDK_INT = 33 // Android 13+ (Tiramisu) required for platform Ed25519 & multimodal leases

    fun isVlmRuntimeSupportedOnDevice(): Boolean {
        val sdkInt = runCatching { android.os.Build.VERSION.SDK_INT }.getOrDefault(0)
        if (sdkInt in 1 until MIN_VLM_SUPPORTED_SDK_INT) {
            return false
        }
        return isPlatformEd25519Supported()
    }

    fun isPlatformEd25519Supported(): Boolean {
        return try {
            java.security.Signature.getInstance("Ed25519")
            true
        } catch (_: Throwable) {
            false
        }
    }

    fun computeFileSha256(file: File): String {
        val digest = java.security.MessageDigest.getInstance("SHA-256")
        file.inputStream().use { stream ->
            val buf = ByteArray(1024 * 1024)
            var bytesRead: Int
            while (stream.read(buf).also { bytesRead = it } != -1) {
                digest.update(buf, 0, bytesRead)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    fun computeCanonicalManifestPayload(manifest: SignedVlmPackManifest): ByteArray {
        val canonicalStr = "${manifest.packId}|${manifest.hfModelId}|${manifest.languageWeightsSha256}|" +
            "${manifest.visionMmprojSha256}|${manifest.policySha256}|${manifest.languageBytes}|" +
            "${manifest.visionMmprojBytes}|${manifest.processorVersion}|${manifest.chatTemplateId}|" +
            "${manifest.imageResolutionCapPx}|${manifest.maxVisualTokens}|${manifest.diagnosticOnly}"
        return canonicalStr.toByteArray(Charsets.UTF_8)
    }

    fun verifyEd25519Signature(publicKeyHex: String, signatureHex: String, payload: ByteArray): Boolean {
        return try {
            val hex2bin: (String) -> ByteArray = { s ->
                ByteArray(s.length / 2) { i ->
                    s.substring(i * 2, i * 2 + 2).toInt(16).toByte()
                }
            }
            val pubBytes = hex2bin(publicKeyHex)
            val sigBytes = hex2bin(signatureHex)
            if (pubBytes.size != 32 || sigBytes.size != 64) return false

            val x509Prefix = byteArrayOf(
                0x30.toByte(), 0x2a.toByte(), 0x30.toByte(), 0x05.toByte(),
                0x06.toByte(), 0x03.toByte(), 0x2b.toByte(), 0x65.toByte(),
                0x70.toByte(), 0x03.toByte(), 0x21.toByte(), 0x00.toByte()
            )
            val x509Key = ByteArray(x509Prefix.size + pubBytes.size)
            System.arraycopy(x509Prefix, 0, x509Key, 0, x509Prefix.size)
            System.arraycopy(pubBytes, 0, x509Key, x509Prefix.size, pubBytes.size)

            val kf = java.security.KeyFactory.getInstance("Ed25519")
            val pk = kf.generatePublic(java.security.spec.X509EncodedKeySpec(x509Key))

            val verifier = java.security.Signature.getInstance("Ed25519")
            verifier.initVerify(pk)
            verifier.update(payload)
            verifier.verify(sigBytes)
        } catch (_: Throwable) {
            false
        }
    }

    fun verifySignedPackContract(
        manifest: SignedVlmPackManifest,
        expectedPublisherPublicKeyHex: String? = pinnedPublisherPublicKeyHex ?: OFFICIAL_PUBLISHER_PUBLIC_KEY_HEX,
        allowUntrustedPublisherForTesting: Boolean = false,
    ): Boolean {
        if (!isVlmRuntimeSupportedOnDevice()) return false

        // Require an independently trusted publisher key for production packs
        val trustedKey = if (allowUntrustedPublisherForTesting) {
            if (expectedPublisherPublicKeyHex != null && expectedPublisherPublicKeyHex != OFFICIAL_PUBLISHER_PUBLIC_KEY_HEX) {
                expectedPublisherPublicKeyHex
            } else {
                manifest.publisherPublicKeyHex
            }
        } else {
            expectedPublisherPublicKeyHex
                ?: return false // Optional / null trusted key strictly rejected in production
        }

        val hex64 = Regex("^[0-9a-fA-F]{64}$")
        val hex128 = Regex("^[0-9a-fA-F]{128}$")

        if (!hex64.matches(manifest.languageWeightsSha256)) return false
        if (!hex64.matches(manifest.visionMmprojSha256)) return false
        if (!hex64.matches(manifest.policySha256)) return false
        if (manifest.languageBytes < 50L * 1024 * 1024 || manifest.visionMmprojBytes < 10L * 1024 * 1024) return false
        if (manifest.imageResolutionCapPx !in 224..1024) return false
        if (manifest.maxVisualTokens !in 64..2048) return false
        if (manifest.processorVersion.isBlank() || manifest.chatTemplateId.isBlank()) return false

        // Strict digital signature verification: require matching trusted publisher key
        if (!hex64.matches(manifest.publisherPublicKeyHex)) return false
        if (!hex128.matches(manifest.publisherSignatureHex)) return false

        if (!manifest.publisherPublicKeyHex.equals(trustedKey, ignoreCase = true)) {
            return false
        }
        val payload = computeCanonicalManifestPayload(manifest)
        return verifyEd25519Signature(manifest.publisherPublicKeyHex, manifest.publisherSignatureHex, payload)
    }

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

    data class EscalationRoute(
        val mode: EscalationMode,
        val invokeOnDeviceVlm: Boolean,
        val expectedVoiScore: Float,
        val targetDomain: String,
        val triggerReason: String,
    )

    data class DialecticalVlmOutcome(
        val vlmClassProb: Float = Float.NaN,
        val vlmClassLogit: Float = Float.NaN,
        val vlmDialecticalProb: Float,
        val posteriorProb: Float,
        val dominantFlawAxis: String,
        val dialecticalExplanation: String,
        val evaluatedAxes: List<String> = emptyList(),
        val cropDownscaleRatio: Float = Float.NaN,
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
     * Evaluates the learned routing policy using the exact same equations as Python's
     * `evaluate_frozen_routing_on_test`:
     *   `escalate = (|z_f - z_thr| < llr_hw) || (|z_a - z_c| > conflict_thr) || (hz >= hz_thr)`
     * When `policy.feasible` is false, never escalates (`invokeOnDeviceVlm = false`).
     */
    fun routeWithPolicy(
        policy: RoutingPolicyConfig,
        fastFusionProb: Float,
        srmCalibratedScore: Float,
        vfmProbeScore: Float?,
        hotspotZScore: Float = 0f,
        activeDomain: String = "human_face",
        docTamperScore: Float = 0f,
        availableHeapMb: Int = 256,
        requireExplanation: Boolean = false,
    ): EscalationRoute {
        if (!policy.feasible) {
            return EscalationRoute(
                mode = EscalationMode.FAST_EXIT_CONFIDENT,
                invokeOnDeviceVlm = false,
                expectedVoiScore = 0f,
                targetDomain = activeDomain,
                triggerReason = "routing_policy_infeasible",
            )
        }
        val zF = logit(fastFusionProb)
        val zThr = logit(policy.decisionThresholdDev)
        val zA = logit(srmCalibratedScore)
        val llrDiff = abs(zF - zThr)
        val nearBoundary = llrDiff < policy.llrUncertaintyHalfwidth

        val conflictDiff = if (vfmProbeScore != null && vfmProbeScore.isFinite()) {
            abs(zA - logit(vfmProbeScore))
        } else {
            0f
        }
        val crossEngineConflict = conflictDiff > policy.calibratedConflictLogitThreshold
        val localizedHotspot = hotspotZScore.isFinite() && hotspotZScore >= policy.hotspotZThreshold
        val docTamper = activeDomain == "document_text" &&
            docTamperScore.isFinite() &&
            docTamperScore >= policy.docTamperThreshold

        val shouldEscalate = requireExplanation || nearBoundary || crossEngineConflict || localizedHotspot || docTamper

        val boundaryScore = if (policy.llrUncertaintyHalfwidth > 0f) {
            max(0f, 1f - llrDiff / max(0.25f, policy.llrUncertaintyHalfwidth))
        } else 0f
        val conflictScore = if (policy.calibratedConflictLogitThreshold < 1e6f) {
            (conflictDiff / max(0.5f, policy.calibratedConflictLogitThreshold)).coerceIn(0f, 1f)
        } else 0f
        val hotspotScore = if (localizedHotspot) {
            ((hotspotZScore - policy.hotspotZThreshold) / 1.5f + 0.5f).coerceIn(0f, 1f)
        } else 0f
        val voi = if (!shouldEscalate) {
            0.10f * boundaryScore
        } else {
            max(0.42f, (0.40f * boundaryScore + 0.35f * conflictScore + 0.25f * hotspotScore).coerceIn(0f, 1f))
        }

        val reason = when {
            requireExplanation -> "forensic_explanation_requested"
            nearBoundary -> "ambiguous_fast_margin"
            crossEngineConflict -> "cross_engine_disagreement"
            localizedHotspot -> "localized_roi_hotspot"
            docTamper -> "specialized_domain_$activeDomain"
            else -> "fast_tier_confident"
        }

        if (!shouldEscalate) {
            return EscalationRoute(EscalationMode.FAST_EXIT_CONFIDENT, false, voi, activeDomain, "fast_tier_confident")
        }
        if (availableHeapMb < 96) {
            return EscalationRoute(EscalationMode.MICRO_MOE_ONLY, false, voi, activeDomain, reason)
        }
        val mode = if (requireExplanation || voi >= 0.68f) {
            EscalationMode.VLM_FULL_DIALECTICAL_COT
        } else {
            EscalationMode.VLM_SINGLE_PASS_LOGIT
        }
        return EscalationRoute(mode, true, voi, activeDomain, reason)
    }

    /**
     * Convenience entry point using default `RoutingPolicyConfig` parameterized by [decisionThreshold].
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
        policy: RoutingPolicyConfig = RoutingPolicyConfig(decisionThresholdDev = decisionThreshold),
    ): EscalationRoute = routeWithPolicy(
        policy = policy,
        fastFusionProb = fastFusionProb,
        srmCalibratedScore = srmCalibratedScore,
        vfmProbeScore = vfmProbeScore,
        hotspotZScore = hotspotZScore,
        activeDomain = activeDomain,
        docTamperScore = max(docTamperScore, avConflictScore),
        availableHeapMb = availableHeapMb,
        requireExplanation = requireExplanation,
    )

    /**
     * Combines observable artifact axis logits (`Float.NaN` when `NOT_APPLICABLE` or `NOT_OBSERVABLE`)
     * with the specialist baseline probability. Explanations cite ONLY the axes that were actually
     * evaluated (`evaluatedAxes`) and their measured log-odds differentials.
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
        vlmDecisionThreshold: Float = 0.50f,
        cropDownscaleRatio: Float = Float.NaN,
        diagnosticOnly: Boolean = false,
        vlmClassProb: Float = Float.NaN,
        vlmClassLogit: Float = Float.NaN,
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

        val roiW = if (hotspotZScore.isFinite() && hotspotZScore >= 1.85f) {
            min(0.65f, 0.35f + 0.12f * (hotspotZScore - 1.85f))
        } else {
            0.20f
        }

        var activeWeightSum = 0f
        val evaluated = ArrayList<String>(6)
        for (i in 0 until 6) {
            if (forgeryLogits6d[i].isFinite() && authenticCounterLogits6d[i].isFinite()) {
                activeWeightSum += rawWeights[i]
                evaluated += RUBRIC_AXES[i]
            }
        }
        if (activeWeightSum <= 0f) {
            val primaryVlmProb = if (vlmClassProb.isFinite()) vlmClassProb else Float.NaN
            val fusedProb = if (vlmClassProb.isFinite()) {
                val zFast = 1.5f * tanh((logit(fastFusionProb) / 1.5f).toDouble()).toFloat()
                val logitVal = if (vlmClassLogit.isFinite()) vlmClassLogit else logit(vlmClassProb)
                val zVlm = 1.8f * tanh((logitVal / 1.8f).toDouble()).toFloat()
                sigmoid(0.62f * zFast + 0.38f * zVlm)
            } else {
                fastFusionProb
            }
            val postProb = if (diagnosticOnly) fastFusionProb else fusedProb
            val isFake = if (vlmClassProb.isFinite()) {
                vlmClassProb >= vlmDecisionThreshold
            } else {
                (if (fusedProb.isFinite()) fusedProb else fastFusionProb) >= decisionThreshold
            }
            val diagPrefix = if (diagnosticOnly) "[DiagnosticOnly] true | " else ""
            val explanation = if (vlmClassProb.isFinite()) {
                "<think>$diagPrefix[Domain] $activeDomain | [SupervisedRubric] none | [VlmClassProb] ${"%.4f".format(vlmClassProb)} " +
                    "(fastProb=${"%.3f".format(fastFusionProb)})</think><answer>${if (isFake) "Fake" else "Real"}</answer>"
            } else {
                "No observable VLM axes applicable; specialist baseline retained."
            }
            return DialecticalVlmOutcome(
                vlmClassProb = vlmClassProb,
                vlmClassLogit = vlmClassLogit,
                vlmDialecticalProb = primaryVlmProb,
                posteriorProb = postProb,
                dominantFlawAxis = if (isFake && vlmClassProb.isFinite()) "global_vlm_detection" else "not_observable",
                dialecticalExplanation = explanation,
                evaluatedAxes = emptyList(),
                cropDownscaleRatio = cropDownscaleRatio,
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

        val rubricLogit = 0.65f * weightedSum + 0.35f * peakCalibratedDiff
        val vlmProb = if (vlmClassProb.isFinite()) {
            vlmClassProb
        } else {
            sigmoid(1.5f * rubricLogit)
        }
        val vlmLogit = if (vlmClassLogit.isFinite()) {
            vlmClassLogit
        } else if (vlmClassProb.isFinite()) {
            logit(vlmClassProb)
        } else {
            rubricLogit
        }

        val zFast = 1.5f * tanh((logit(fastFusionProb) / 1.5f).toDouble()).toFloat()
        val zVlm = 1.8f * tanh((vlmLogit / 1.8f).toDouble()).toFloat()
        val fusedProb = sigmoid(0.62f * zFast + 0.38f * zVlm)
        val postProb = if (diagnosticOnly) fastFusionProb else fusedProb

        val isFake = if (vlmClassProb.isFinite()) {
            vlmClassProb >= vlmDecisionThreshold
        } else {
            fusedProb >= decisionThreshold
        }
        val flawAxis = if (isFake) RUBRIC_AXES[bestIdx] else "none"
        val axesSummary = evaluated.joinToString(",")
        val diagPrefix = if (diagnosticOnly) "[DiagnosticOnly] true | " else ""
        val explanation = if (isFake) {
            "<think>$diagPrefix[Domain] $activeDomain | [EvaluatedAxes] $axesSummary | " +
                "[PeakAxis] $flawAxis (deltaLogit=${"%.2f".format(peakCalibratedDiff)}, " +
                "fastProb=${"%.3f".format(fastFusionProb)}, vlmProb=${"%.3f".format(vlmProb)})</think><answer>Fake</answer>"
        } else {
            "<think>$diagPrefix[Domain] $activeDomain | [EvaluatedAxes] $axesSummary | " +
                "[PeakDeltaLogit] ${"%.2f".format(peakCalibratedDiff)} " +
                "(fastProb=${"%.3f".format(fastFusionProb)}, vlmProb=${"%.3f".format(vlmProb)})</think><answer>Real</answer>"
        }

        return DialecticalVlmOutcome(
            vlmClassProb = if (vlmClassProb.isFinite()) vlmClassProb else vlmProb,
            vlmClassLogit = if (vlmClassLogit.isFinite()) vlmClassLogit else vlmLogit,
            vlmDialecticalProb = vlmProb,
            posteriorProb = postProb,
            dominantFlawAxis = flawAxis,
            dialecticalExplanation = explanation,
            evaluatedAxes = evaluated,
            cropDownscaleRatio = cropDownscaleRatio,
            diagnosticOnly = diagnosticOnly,
        )
    }

    /**
     * End-to-end execution of an escalated Tier-2 VLM deep scan:
     * 1. Acquires an exclusive session lease on [inferenceEngine] so `siglip_probe_int8.tflite`
     *    is evicted and blocked from reloading while the VLM runs.
     * 2. Decodes the original-source ROI crop from [sourceUri] via [BitmapRegionDecoder] when
     *    available, computing `OriginalSourceCropSpec.downscaleRatio`.
     * 3. Runs [backend] and projects `hLast` through [policyBundle]'s `headContract`.
     * 4. Fuses the resulting 6-axis forgery and counter-evidence logits with [fastFusionProb].
     */
    fun executeEscalatedDeepScan(
        context: Context?,
        inferenceEngine: LiteRTInferenceEngine,
        policyBundle: PolicyBundle,
        backend: VlmPrefillBackend,
        fullBitmap: Bitmap,
        sourceUri: Uri? = null,
        hotspotNormBox: FloatArray? = null,
        fastFusionProb: Float,
        hotspotZScore: Float = 0f,
        activeDomain: String = "human_face",
        applicableMask6d: BooleanArray? = null,
        diagnosticOnly: Boolean = true,
    ): DialecticalVlmOutcome {
        requireNotNull(policyBundle.headContract) {
            "PolicyBundle '${policyBundle.modelId}' does not contain a head_contract"
        }
        var roiBitmap: Bitmap? = null
        var cropSpec: OriginalSourceCropSpec? = null
        try {
            return executeLeasedProjection(
                inferenceEngine = inferenceEngine,
                policyBundle = policyBundle,
                fastFusionProb = fastFusionProb,
                hotspotZScore = hotspotZScore,
                activeDomain = activeDomain,
                applicableMask6d = applicableMask6d,
                diagnosticOnly = diagnosticOnly,
                cropDownscaleRatioProvider = { cropSpec?.downscaleRatio ?: 1.0f },
            ) {
                if (context != null && sourceUri != null && hotspotNormBox != null && hotspotNormBox.size == 4) {
                    runCatching {
                        context.contentResolver.openInputStream(sourceUri)?.use { stream ->
                            @Suppress("DEPRECATION")
                            val decoder = BitmapRegionDecoder.newInstance(stream, false)
                            if (decoder != null) {
                                try {
                                    val spec = computeOriginalSourceCropSpec(
                                        sourceWidth = decoder.width,
                                        sourceHeight = decoder.height,
                                        normX0 = hotspotNormBox[0],
                                        normY0 = hotspotNormBox[1],
                                        normX1 = hotspotNormBox[2],
                                        normY1 = hotspotNormBox[3],
                                    )
                                    cropSpec = spec
                                    val rect = Rect(spec.left, spec.top, spec.left + spec.width, spec.top + spec.height)
                                    roiBitmap = decoder.decodeRegion(rect, BitmapFactory.Options())
                                } finally {
                                    decoder.recycle()
                                }
                            }
                        }
                    }
                }
                if (backend is LlamaCppVlmPrefillBackend) {
                    backend.loadNativeContext()
                }
                try {
                    backend.extractFinalHiddenState(fullBitmap, roiBitmap, cropSpec, activeDomain)
                } finally {
                    if (backend is LlamaCppVlmPrefillBackend) {
                        backend.unloadNativeContext()
                    }
                }
            }
        } finally {
            val roi = roiBitmap
            if (roi != null && roi !== fullBitmap && !roi.isRecycled) {
                roi.recycle()
            }
        }
    }

    /**
     * Bitmap-independent core of [executeEscalatedDeepScan]. Holds an exclusive session lease on
     * [inferenceEngine] for the whole duration of [extractHiddenState], projects the returned
     * final hidden state through the policy's 13-output head contract, and fuses the axes.
     * The lease is always released, including when [extractHiddenState] throws.
     */
    fun executeLeasedProjection(
        inferenceEngine: LiteRTInferenceEngine,
        policyBundle: PolicyBundle,
        fastFusionProb: Float,
        hotspotZScore: Float = 0f,
        activeDomain: String = "human_face",
        applicableMask6d: BooleanArray? = null,
        diagnosticOnly: Boolean = true,
        cropDownscaleRatioProvider: () -> Float = { 1.0f },
        extractHiddenState: () -> FloatArray,
    ): DialecticalVlmOutcome {
        val head = requireNotNull(policyBundle.headContract) {
            "PolicyBundle '${policyBundle.modelId}' does not contain a head_contract"
        }
        return inferenceEngine.acquireExclusiveVlmSession().use { _ ->
            val hLast = extractHiddenState()
            val projected = projectWithHeadContract(hLast, head, applicableMask6d)
            fuseDialecticalLogits(
                fastFusionProb = fastFusionProb,
                forgeryLogits6d = projected.forgeryLogits6d,
                authenticCounterLogits6d = projected.authenticCounterLogits6d,
                hotspotZScore = hotspotZScore,
                activeDomain = activeDomain,
                decisionThreshold = policyBundle.routingPolicy.decisionThresholdDev,
                vlmDecisionThreshold = policyBundle.vlmCalibration.decisionThresholdDev,
                cropDownscaleRatio = cropDownscaleRatioProvider(),
                diagnosticOnly = policyBundle.diagnosticOnly || diagnosticOnly,
                vlmClassProb = projected.classProb,
                vlmClassLogit = projected.classLogit,
            )
        }
    }

    /**
     * Loads an optional user-imported local policy bundle from `forensic_vlm_policy.json`
     * inside [packDir] when present and hash-verified.
     */
    fun loadInstalledPolicyBundleOrNull(
        packDir: File,
        expectedPublisherPublicKeyHex: String? = pinnedPublisherPublicKeyHex ?: OFFICIAL_PUBLISHER_PUBLIC_KEY_HEX,
        allowUntrustedPublisherForTesting: Boolean = false,
    ): PolicyBundle? {
        overridePolicyBundle?.let { return it }
        return runCatching {
            val policyFile = File(packDir, "forensic_vlm_policy.json")
            if (!policyFile.isFile) return null

            val manifestFile = File(packDir, "signed_manifest.json").takeIf { it.isFile }
                ?: File(packDir, "manifest.json").takeIf { it.isFile }
            if (manifestFile != null) {
                val manifest = SignedVlmPackManifest.fromJson(manifestFile.readText(Charsets.UTF_8))
                if (!verifySignedPackContract(manifest, expectedPublisherPublicKeyHex, allowUntrustedPublisherForTesting)) {
                    return null
                }
                if (manifest.policySha256.isNotEmpty()) {
                    val actualSha = computeFileSha256(policyFile)
                    if (!actualSha.equals(manifest.policySha256, ignoreCase = true)) {
                        return null
                    }
                }
            }

            PolicyBundle.fromJson(policyFile.readText(Charsets.UTF_8))
        }.getOrNull()
    }

    /**
     * Loads an optional user-imported local policy bundle from `<filesDir>/vlm_packs/forensic_vlm_policy.json`
     * when present on device.
     */
    fun loadInstalledPolicyBundleOrNull(context: Context): PolicyBundle? =
        loadInstalledPolicyBundleOrNull(File(context.filesDir, "vlm_packs"))

    /**
     * Discovers, cryptographically verifies, and initializes the native prefill backend
     * from [packDir] when a valid signed pack with complete verified weights is resident.
     */
    fun initializeActiveBackendIfAvailable(
        packDir: File,
        expectedPublisherPublicKeyHex: String? = pinnedPublisherPublicKeyHex ?: OFFICIAL_PUBLISHER_PUBLIC_KEY_HEX,
        allowUntrustedPublisherForTesting: Boolean = false,
    ): VlmPrefillBackend? {
        activePrefillBackend?.let { return it }
        if (!isVlmRuntimeSupportedOnDevice()) return null

        return runCatching {
            if (!packDir.isDirectory) return null

            val manifestFile = File(packDir, "signed_manifest.json").takeIf { it.isFile }
                ?: File(packDir, "manifest.json").takeIf { it.isFile }
                ?: return null

            val manifest = SignedVlmPackManifest.fromJson(manifestFile.readText(Charsets.UTF_8))
            if (!verifySignedPackContract(manifest, expectedPublisherPublicKeyHex, allowUntrustedPublisherForTesting)) {
                return null
            }

            // 1. Verify policy file exists and matches signed hash
            val policyFile = File(packDir, "forensic_vlm_policy.json")
            if (!policyFile.isFile) return null
            val actualPolicySha = computeFileSha256(policyFile)
            if (!actualPolicySha.equals(manifest.policySha256, ignoreCase = true)) {
                return null
            }

            val policyBundle = PolicyBundle.fromJson(policyFile.readText(Charsets.UTF_8))
            val head = policyBundle.headContract ?: return null
            val expectedHiddenDim = head.hiddenDim
            if (expectedHiddenDim <= 0) return null

            // 2. Discover and hash-verify language model file
            val modelFile = listOf(
                File(packDir, "${manifest.packId}-model.gguf"),
                File(packDir, "${manifest.packId}-language.gguf"),
                File(packDir, "model.gguf"),
                File(packDir, "language.gguf"),
            ).firstOrNull { it.isFile && it.length() == manifest.languageBytes }
                ?: packDir.listFiles()?.firstOrNull {
                    it.isFile && it.extension == "gguf" && !it.name.contains("mmproj") && it.length() == manifest.languageBytes
                }
                ?: return null

            val actualLangSha = computeFileSha256(modelFile)
            if (!actualLangSha.equals(manifest.languageWeightsSha256, ignoreCase = true)) {
                return null
            }

            // 3. Discover and hash-verify vision mmproj file
            val mmprojFile = listOf(
                File(packDir, "${manifest.packId}-mmproj.gguf"),
                File(packDir, "${manifest.packId}-vision.gguf"),
                File(packDir, "mmproj.gguf"),
                File(packDir, "vision.gguf"),
            ).firstOrNull { it.isFile && it.length() == manifest.visionMmprojBytes }
                ?: packDir.listFiles()?.firstOrNull {
                    it.isFile && it.extension == "gguf" && it.name.contains("mmproj") && it.length() == manifest.visionMmprojBytes
                }
                ?: return null

            val actualVisSha = computeFileSha256(mmprojFile)
            if (!actualVisSha.equals(manifest.visionMmprojSha256, ignoreCase = true)) {
                return null
            }

            val backend = LlamaCppVlmPrefillBackend(
                modelPath = modelFile.absolutePath,
                mmprojPath = mmprojFile.absolutePath,
                hiddenDim = expectedHiddenDim,
                imageResolutionCapPx = manifest.imageResolutionCapPx,
            )
            activePrefillBackend = backend
            backend
        }.getOrNull()
    }

    /**
     * Discovers, cryptographically verifies, and initializes the native prefill backend
     * from `<filesDir>/vlm_packs/` when a valid signed pack is resident.
     */
    fun initializeActiveBackendIfAvailable(
        context: Context,
        expectedPublisherPublicKeyHex: String? = pinnedPublisherPublicKeyHex,
    ): VlmPrefillBackend? =
        initializeActiveBackendIfAvailable(File(context.filesDir, "vlm_packs"), expectedPublisherPublicKeyHex)

    /**
     * Resets mutable singleton state for test isolation.
     */
    fun resetForTesting() {
        activePrefillBackend = null
        overridePolicyBundle = null
        experimentalVlmCascadeEnabled = false
        experimentalDiagnosticOnly = true
        pinnedPublisherPublicKeyHex = null
    }
}
