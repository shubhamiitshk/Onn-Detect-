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
    companion object { const val MODEL_FILE = "srm_forensic_cnn_int8.tflite" }
    fun analyze(bitmap: Bitmap, checkStop: () -> Unit = {}): Float {
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
        val scaled: Bitmap = when (frontendId) {
            mosaicId -> {
                val tile = if (size >= 64 && size % 32 == 0) 32 else maxOf(8, size / 4)
                val pair = PhaseSpectrumAndPatchCraftAndroid.extractPatchCraftMosaics(bitmap, mosaicSize = size, tileSize = tile)
                pair.poorMosaic.recycle()
                pair.richMosaic
            }
            legacyBilinearId -> Bitmap.createScaledBitmap(bitmap, size, size, true)
            else -> {
                requireModelContract(false) { "Unsupported SRM frontend_id: $frontendId" }
                Bitmap.createScaledBitmap(bitmap, size, size, true)
            }
        }
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
