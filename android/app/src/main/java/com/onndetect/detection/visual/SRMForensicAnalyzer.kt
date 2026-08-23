package com.onndetect.detection.visual

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONObject
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Runs the trained SRMForensicCNN (TFLite INT8) over an image.
 *
 * Input contract (see ml/scripts/export_all_models.py):
 *   tensor shape (1, 32, 32, 3), RGB floats in [0,1] - exactly the native-
 *   resolution ToTensor() preprocessing the model was retrained with.
 */
class SRMForensicAnalyzer(
    private val context: Context,
    private val engine: com.onndetect.inference.LiteRTInferenceEngine,
) {

    companion object {
        const val MODEL_FILE = "srm_forensic_cnn_int8.tflite"
        const val INPUT_SIZE = 32

        /** Preallocated input container: [1][32][32][3]. */
        private fun newInput(): Array<Array<Array<FloatArray>>> =
            Array(1) { Array(INPUT_SIZE) { Array(INPUT_SIZE) { FloatArray(3) } } }
    }

    private val input = newInput()

    /** @return P(AI-generated) in [0,1]; 0.5 on any failure. */
    fun analyze(bitmap: Bitmap): Float {
        return try {
            fillInput(bitmap)
            engine.predict(MODEL_FILE, input).coerceIn(0f, 1f)
        } catch (_: Exception) {
            0.5f
        }
    }

    private fun fillInput(src: Bitmap) {
        val scaled = Bitmap.createScaledBitmap(src, INPUT_SIZE, INPUT_SIZE, true)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        if (scaled !== src) scaled.recycle()

        val plane = input[0]
        for (i in pixels.indices) {
            val c = pixels[i]
            plane[i / INPUT_SIZE][i % INPUT_SIZE][0] = ((c shr 16 and 0xFF)) / 255f
            plane[i / INPUT_SIZE][i % INPUT_SIZE][1] = ((c shr 8 and 0xFF)) / 255f
            plane[i / INPUT_SIZE][i % INPUT_SIZE][2] = ((c and 0xFF)) / 255f
        }
    }
}

/**
 * Loads calibration constants (JSON) from assets/constants/ with safe
 * fallbacks so a missing file degrades instead of crashing.
 */
object CalibrationStore {

    fun naturalHistogram(context: Context): FloatArray? {
        return try {
            val json = JSONObject(readAsset(context, "constants/natural_diff_histogram.json"))
            val bins = json.optJSONArray("bins") ?: return null
            FloatArray(bins.length()) { i -> bins.getDouble(i).toFloat() }
        } catch (_: Exception) {
            null
        }
    }

    fun metaDecisionThreshold(context: Context): Float = try {
        val json = JSONObject(readAsset(context, "constants/meta_classifier.json"))
        json.getDouble("decision_threshold").toFloat()
    } catch (_: Exception) {
        0.5f
    }

    fun audioThreshold(context: Context): Float = try {
        val json = JSONObject(readAsset(context, "constants/lcnn_audio_threshold.json"))
        json.getDouble("threshold_at_eer").toFloat()
    } catch (_: Exception) {
        0.5f
    }

    private fun readAsset(context: Context, path: String): String =
        context.assets.open(path).bufferedReader().use { it.readText() }

    @Suppress("unused")
    private fun clamp01(v: Float) = min(1f, v.coerceAtLeast(0f))

    @Suppress("unused")
    private fun round(v: Double) = v.roundToInt()
}
