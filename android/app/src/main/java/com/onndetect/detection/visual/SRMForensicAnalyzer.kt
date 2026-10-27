package com.onndetect.detection.visual

import android.content.Context
import android.graphics.Bitmap
import com.onndetect.inference.LiteRTInferenceEngine
import com.onndetect.inference.requireModelContract
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONObject

/** Square RGB float32 preprocessing sized and routed from the model's declared frontend_id contract. */
class SRMForensicAnalyzer(private val context: Context, private val engine: LiteRTInferenceEngine) {
    companion object {
        const val MODEL_FILE = "srm_forensic_cnn_int8.tflite"

        /**
         * Combines rich-texture and poor-texture PatchCraft mosaic SRM scores:
         * 1. Portrait-Mode Bokeh Guard: When `poorScore < 0.20` (authentic camera thermal noise floor in
         *    smooth background) while `richScore` is moderately elevated (`0.40..0.72` due to ISP edge
         *    sharpening) and `contrast > 0.35`, dampens the sharpening spike by 15% of the contrast.
         * 2. Co-Synthetic Diffusion Reinforcement: When both `richScore > 0.48` and `poorScore > 0.48`
         *    (VAE upsampler periodicity present in both textured and flat regions), reinforces the score.
         */
        fun computeContrastCalibratedScore(richScore: Float, poorScore: Float): Float {
            if (!richScore.isFinite()) return richScore
            if (!poorScore.isFinite() || poorScore !in 0f..1f) return richScore.coerceIn(0f, 1f)
            val contrast = richScore - poorScore
            return when {
                poorScore < 0.20f && richScore in 0.40f..0.72f && contrast > 0.35f ->
                    (richScore - 0.15f * contrast).coerceIn(0f, 1f)
                richScore > 0.48f && poorScore > 0.48f ->
                    (0.75f * richScore + 0.25f * maxOf(richScore, poorScore)).coerceIn(0f, 1f)
                else -> richScore.coerceIn(0f, 1f)
            }
        }

        /**
         * v11 PRISM Saliency-Guided Localized ROI Splice & Contrast Calibrator.
         *
         * When the 32x32 Laplacian residual heatmap identifies a concentrated spatial anomaly
         * (`hotspotZScore >= 1.85f`, e.g. localized face-swap, Magic Eraser, or Generative Fill)
         * and the focused second-pass `roiScore` exceeds `baseScore + 0.12f`, blends the
         * localized splice evidence into the effective SRM score in log-odds space so it
         * actively drives `XGBoostMetaClassifier` rather than sitting unused in UI telemetry.
         */
        fun computeContrastAndRoiCalibratedScore(
            richScore: Float,
            poorScore: Float = Float.NaN,
            roiScore: Float = Float.NaN,
            hotspotZScore: Float = 0f,
        ): Float {
            val base = computeContrastCalibratedScore(richScore, poorScore)
            if (!base.isFinite()) return richScore
            if (!roiScore.isFinite() || roiScore !in 0f..1f || hotspotZScore < 1.85f) {
                return base
            }
            if (roiScore <= base + 0.12f) return base
            val eps = 1e-5
            val bClamped = base.toDouble().coerceIn(eps, 1.0 - eps)
            val rClamped = roiScore.toDouble().coerceIn(eps, 1.0 - eps)
            val zBase = kotlin.math.ln(bClamped / (1.0 - bClamped))
            val zRoi = kotlin.math.ln(rClamped / (1.0 - rClamped))
            val alphaRoi = (0.25 + 0.10 * (hotspotZScore - 1.85).toDouble()).coerceIn(0.25, 0.65)
            val zEff = (1.0 - alphaRoi) * zBase + alphaRoi * zRoi
            return (1.0 / (1.0 + kotlin.math.exp(-zEff))).toFloat().coerceIn(0f, 1f)
        }
    }

    data class SrmDetailedResult(
        val effectiveScore: Float,
        val richScore: Float,
        val poorScore: Float = Float.NaN,
        val richPoorContrast: Float = Float.NaN,
    )

    /**
     * Primary SRM score routed into the fusion vector (slot 0).
     *
     * Returns the contrast-calibrated effective score (rich-poor PatchCraft mosaics, bokeh guard,
     * co-synthetic reinforcement) computed with `includePoorMosaic = true`, so the poor-texture
     * mosaic and `computeContrastCalibratedScore` always participate in standard scans.
     */
    fun analyze(bitmap: Bitmap, checkStop: () -> Unit = {}): Float =
        analyzeDetailed(bitmap, checkStop, includePoorMosaic = true).effectiveScore

    fun analyzeDetailed(bitmap: Bitmap, checkStop: () -> Unit = {}, includePoorMosaic: Boolean = true): SrmDetailedResult {
        checkStop()
        require(!bitmap.isRecycled && bitmap.width > 1 && bitmap.height > 1) { "Invalid image" }
        val contract = engine.contract(MODEL_FILE)
        val shape = contract.inputShape
        requireModelContract(shape.size == 4 && shape[0] == 1 && shape[3] == 3 && shape[1] == shape[2] && shape[1] in 16..1024) { "Unsupported image tensor" }
        val modelManifest = JSONObject(context.assets.open("constants/release_manifest.json").bufferedReader().use { it.readText() })
            .getJSONObject("models").getJSONObject(MODEL_FILE)
        val expected = modelManifest.getJSONArray("input_shape")
        requireModelContract((0 until 4).all { expected.getInt(it) == shape[it] }) { "Release/model input mismatch" }
        val frontendId = modelManifest.optString("frontend_id", "")
        val size = shape[1]
        val legacyBilinearId = "rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-size${size}-float32"
        val mosaicId = "${legacyBilinearId}-patchcraft-mosaic-tile32"

        fun scoreScaledBitmap(scaled: Bitmap): Float {
            val pixels = IntArray(size * size)
            try { scaled.getPixels(pixels, 0, size, 0, 0, size, size) }
            finally { if (scaled !== bitmap) scaled.recycle() }
            // Per-request ownership: another scan cannot overwrite this input.
            val input = ByteBuffer.allocateDirect(contract.inputBytes).order(ByteOrder.nativeOrder())
            for (c in pixels) {
                input.putFloat((c shr 16 and 255) / 255f)
                input.putFloat((c shr 8 and 255) / 255f)
                input.putFloat((c and 255) / 255f)
            }
            input.flip()
            return engine.predict(MODEL_FILE, input, checkStop)
        }

        return when (frontendId) {
            mosaicId -> {
                val tile = if (size >= 64 && size % 32 == 0) 32 else maxOf(8, size / 4)
                val pair = PhaseSpectrumAndPatchCraftAndroid.extractPatchCraftMosaics(bitmap, mosaicSize = size, tileSize = tile)
                val richScore = try {
                    scoreScaledBitmap(pair.richMosaic)
                } catch (t: Throwable) {
                    pair.poorMosaic.recycle()
                    throw t
                }
                val poorScore = if (includePoorMosaic) {
                    runCatching { scoreScaledBitmap(pair.poorMosaic) }.getOrDefault(Float.NaN)
                } else {
                    pair.poorMosaic.recycle()
                    Float.NaN
                }
                val contrast = if (poorScore.isFinite()) richScore - poorScore else Float.NaN
                val effective = computeContrastCalibratedScore(richScore, poorScore)
                SrmDetailedResult(
                    effectiveScore = effective,
                    richScore = richScore,
                    poorScore = poorScore,
                    richPoorContrast = contrast,
                )
            }
            legacyBilinearId -> {
                val s = scoreScaledBitmap(Bitmap.createScaledBitmap(bitmap, size, size, true))
                SrmDetailedResult(effectiveScore = s, richScore = s)
            }
            else -> {
                requireModelContract(false) { "Unsupported SRM frontend_id: $frontendId" }
                val s = scoreScaledBitmap(Bitmap.createScaledBitmap(bitmap, size, size, true))
                SrmDetailedResult(effectiveScore = s, richScore = s)
            }
        }
    }
}

/**
 * Track 1 v11 upgrade: Log-Odds Rich-Poor Contrast + Saliency-Guided ROI Injection.
 *
 * Kotlin port of `ContrastivePatchCraftROI.evaluate()` from `onn_prism_v11_algorithms.py`.
 * Fixes the dead `roiScore` telemetry field: the localized 8x8 hotspot ROI score is now
 * exported into `roiHotspotScore` (fusion slot 2) and the rich-poor texture contrast into
 * `richPoorContrast` (fusion slot 1), replacing the chance/inverted `fft_score` and `kl_score`.
 */
object ContrastivePatchCraftROI {
    private const val EPSILON = 1e-6f

    private fun logit(p: Float): Float {
        val pp = p.coerceIn(EPSILON, 1.0f - EPSILON)
        return kotlin.math.ln((pp / (1.0f - pp)).toDouble()).toFloat()
    }

    private fun sigmoid(z: Float): Float = (1.0f / (1.0f + kotlin.math.exp(-z.toDouble()))).toFloat()

    data class Result(
        val calibratedSrmScore: Float,
        val richMosaicScore: Float,
        val poorMosaicScore: Float,
        val richPoorContrast: Float,
        val roiHotspotScore: Float,
        val hotspotZScore: Float,
        val bokehDampened: Boolean,
        val roiBoosted: Boolean,
    )

    /**
     * Evaluates rich/poor PatchCraft mosaic scores in log-odds space and injects the
     * saliency-guided localized ROI hotspot score.
     *
     * @param richScore raw rich-texture mosaic score from SRMForensicAnalyzer (0..1)
     * @param poorScore raw poor-texture mosaic score from SRMForensicAnalyzer (0..1), or null
     * @param roiCropScore SRM score on the localized 8x8 hotspot crop (0..1), or null
     * @param hotspotZScore Z-score from ResidualHeatmapExtractor.extractHotspotBoundingBox
     */
    fun evaluate(
        richScore: Float,
        poorScore: Float?,
        roiCropScore: Float?,
        hotspotZScore: Float,
    ): Result {
        val rProb = richScore.coerceIn(EPSILON, 1.0f - EPSILON)
        val zRich = logit(rProb)

        var bokehDampened = false
        var roiBoosted = false
        var pProb: Float = Float.NaN
        var contrastNorm = 0.5f
        var zEff = zRich

        if (poorScore != null && poorScore.isFinite()) {
            pProb = poorScore.coerceIn(EPSILON, 1.0f - EPSILON)
            val zPoor = logit(pProb)
            // Normalized contrast in [0, 1] centered at 0.5 for fusion slot 1
            contrastNorm = sigmoid(0.5f * (zRich - zPoor))

            if (pProb < 0.18f && rProb in 0.35f..0.78f && (rProb - pProb) > 0.32f) {
                // Portrait bokeh / ISP edge sharpening guard: smooth background is authentic
                zEff = zRich - 0.65f * (rProb - pProb)
                bokehDampened = true
            } else if (rProb > 0.45f && pProb > 0.45f) {
                // Co-synthetic VAE upsampling grid present in both rich and poor regions
                zEff = 0.70f * zRich + 0.30f * kotlin.math.max(zRich, zPoor) + 0.35f
            } else {
                zEff = 0.85f * zRich + 0.15f * zPoor
            }
        }

        val effProb = sigmoid(zEff)
        var roiOut = effProb

        // Saliency-Guided Localized ROI Dual-Pass Injection
        if (roiCropScore != null && roiCropScore.isFinite() && hotspotZScore >= 1.85f) {
            val roiP = roiCropScore.coerceIn(EPSILON, 1.0f - EPSILON)
            if (roiP > effProb + 0.12f) {
                // Weight the localized ROI anomaly by hotspot Z-score concentration
                val alphaRoi = kotlin.math.min(0.65f, 0.25f + 0.10f * (hotspotZScore - 1.85f))
                zEff = (1.0f - alphaRoi) * zEff + alphaRoi * logit(roiP)
                roiBoosted = true
            }
            roiOut = roiP
        }

        return Result(
            calibratedSrmScore = sigmoid(zEff).coerceIn(0f, 1f),
            richMosaicScore = rProb,
            poorMosaicScore = pProb,
            richPoorContrast = contrastNorm,
            roiHotspotScore = roiOut.coerceIn(0f, 1f),
            hotspotZScore = hotspotZScore,
            bokehDampened = bokehDampened,
            roiBoosted = roiBoosted,
        )
    }
}

/** Constants never silently fall back to a verdict-changing value. */
object CalibrationStore {
    fun naturalHistogram(context: Context): FloatArray {
        val json = read(context, "constants/natural_diff_histogram.json")
        val bins = json.getJSONArray("bins")
        require(bins.length() == 21)
        val histogram = FloatArray(21) { bins.getDouble(it).toFloat() }
        require(histogram.all { it.isFinite() && it > 0f }) { "Invalid baseline bins" }
        val sum = histogram.sum()
        require(sum.isFinite() && sum > 0f)
        return histogram.apply { for (i in indices) this[i] /= sum }
    }
    fun metaDecisionThreshold(context: Context): Float = threshold(read(context, "constants/meta_classifier.json"), "decision_threshold")
    fun audioThreshold(context: Context): Float = threshold(read(context, "constants/lcnn_audio_threshold.json"), "threshold_at_eer")
    private fun threshold(json: JSONObject, key: String): Float = json.getDouble(key).toFloat().also {
        require(it.isFinite() && it in 0f..1f) { "Invalid decision threshold" }
    }
    private fun read(context: Context, path: String) = JSONObject(context.assets.open(path).bufferedReader().use { it.readText() })
}
