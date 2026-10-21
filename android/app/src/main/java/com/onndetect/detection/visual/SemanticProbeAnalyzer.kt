package com.onndetect.detection.visual

import android.content.Context
import android.graphics.Bitmap
import com.onndetect.inference.LiteRTInferenceEngine
import com.onndetect.inference.requireModelContract
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt
import org.json.JSONObject

/**
 * Engine C - the "Lite C" appearance probe.
 *
 * A small RGB-appearance CNN with a learned classification head, exported
 * as one INT8 TFLite graph: image -> CNN -> P(AI).
 *
 * The shipped artifact is `SemanticProbeCNN`, a distilled student trained
 * against a frozen CLIP-style vision tower's logits (see
 * `ml/scripts/train_semantic_probe.py`). It is NOT the frozen tower
 * itself: the tower is ~22 MB and the shipped INT8 graph is ~153 KiB.
 * The distillation loses measurable accuracy, which is tracked in the
 * release evaluation as a teacher/student AUC gap.
 *
 * It is the semantic counterpart to [SRMForensicAnalyzer], which reads
 * high-frequency residual structure. Engine C reads appearance and
 * semantics. The two engines fail differently, which is what makes the
 * fusion worth more than its best member.
 *
 * Frontend contract (must match `data.image_frontend.semantic_image_tensor`):
 *  - square BICUBIC resize to 224x224
 *  - per-channel `(x / 255 - mean) / std` with the CLIP constants
 *
 * The resampling filter differs from Engine A's bilinear resize on purpose.
 * Sharing a resampling artifact between the two engines would make the
 * fusion measure the same low-level cue twice and would inflate the gain.
 *
 * The model is OPTIONAL. When it is absent, unvalidated, or fails to run,
 * the caller receives `null` and the fusion slot stays unavailable (NaN) -
 * never a neutral constant that would read as a real low measurement.
 */
class SemanticProbeAnalyzer(
    private val context: Context,
    private val engine: LiteRTInferenceEngine,
) {
    companion object {
        const val MODEL_FILE = "semantic_probe_int8.tflite"
        const val INPUT_SIZE = 224

        // CLIP normalization constants, identical to the Python frontend.
        private val MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        private val STD = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)
    }

    /**
     * Score one image with Engine C.
     *
     * @return P(AI-generated) in [0,1], or `null` when the probe is unavailable.
     */
    fun analyze(bitmap: Bitmap, checkStop: () -> Unit = {}): Float? {
        checkStop()
        if (bitmap.isRecycled || bitmap.width < 2 || bitmap.height < 2) return null

        return try {
            val contract = engine.contract(MODEL_FILE)
            val shape = contract.inputShape
            requireModelContract(
                shape.size == 4 && shape[0] == 1 && shape[3] == 3 &&
                    shape[1] == shape[2] && shape[1] in 64..1024
            ) { "Unsupported semantic probe tensor" }

            val size = shape[1]
            // BICUBIC, not Bitmap.createScaledBitmap(filter = true).
            // The declared frontend contract is
            // "clip-appearance-bicubic224-clipnorm-meanstd-float32" and the
            // Python training/calibration path resizes with
            // Image.Resampling.BICUBIC. createScaledBitmap applies BILINEAR,
            // which made the device feed Engine C a different tensor than
            // every threshold in the release was calibrated from.
            val scaled = BicubicResizer.resize(bitmap, size, size)
            val pixels = IntArray(size * size)
            try {
                scaled.getPixels(pixels, 0, size, 0, 0, size, size)
            } finally {
                if (scaled !== bitmap) scaled.recycle()
            }

            val input = ByteBuffer.allocateDirect(contract.inputBytes)
                .order(ByteOrder.nativeOrder())
            for (c in pixels) {
                val r = (c shr 16 and 255) / 255f
                val g = (c shr 8 and 255) / 255f
                val b = (c and 255) / 255f
                input.putFloat((r - MEAN[0]) / STD[0])
                input.putFloat((g - MEAN[1]) / STD[1])
                input.putFloat((b - MEAN[2]) / STD[2])
            }
            input.flip()
            engine.predict(MODEL_FILE, input, checkStop)
        } catch (e: Exception) {
            // An unavailable second engine is UNAVAILABLE evidence, not a zero
            // and not a failure of the primary detector.
            null
        }
    }

    /**
     * Verify the packaged probe matches the release contract before use.
     *
     * Returns true when the probe's declared tensor shape agrees with the
     * release manifest. A mismatch means the probe belongs to another
     * release and must not contribute to a verdict.
     */
    fun matchesReleaseContract(): Boolean {
        return try {
            val contract = engine.contract(MODEL_FILE)
            val declared = JSONObject(
                context.assets.open("constants/release_manifest.json")
                    .bufferedReader().use { it.readText() }
            ).getJSONObject("models").getJSONObject(MODEL_FILE).getJSONArray("input_shape")
            (0 until 4).all { declared.getInt(it) == contract.inputShape[it] }
        } catch (e: Exception) {
            false
        }
    }
}
