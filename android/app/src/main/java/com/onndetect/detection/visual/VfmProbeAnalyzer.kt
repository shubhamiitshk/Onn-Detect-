package com.onndetect.detection.visual

import android.content.Context
import android.graphics.Bitmap
import com.onndetect.inference.LiteRTInferenceEngine
import com.onndetect.inference.requireModelContract
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.inject.Inject
import org.json.JSONObject

/**
 * The frozen vision-foundation-model probe: image -> tower -> linear head -> P(AI).
 *
 * ## What this replaces
 *
 * This class replaced the distilled CNN student (Engine C, "Lite C"), which
 * reached **0.6871** sealed-test AUC. A frozen SigLIP-B/16 with a
 * 769-parameter linear head reaches **0.8919**, and the shipped INT8 graph
 * keeps **0.8910** after quantization. That is +0.204 AUC over the student.
 *
 * The source-debiased variant of this head was measured and **rejected**: it
 * lowers the LAION false-positive rate (0.2897 -> 0.2573) but costs AUC
 * (0.8919 -> 0.8682), so the plain head is what ships.
 *
 * The probe is a LIKE-FOR-LIKE replacement for Engine C: it answers the
 * same question (P(this image is AI-generated)) for the same input, so it
 * occupies **slot 6** of the fusion vector. It is not a new slot, so the
 * fusion contract does not change and the shipped ten-slot format stands.
 *
 * ## Cost
 *
 * SigLIP-B/16 is 86 M parameters: **89.96 MB** as INT8, against a
 * 153 KiB student and a 21 MB APK today. DINOv2-S/14 is the 21 MB
 * alternative and reaches 0.8358. The size decision is a product
 * decision; this class supports either model.
 *
 * The compute cost is real too: SigLIP-B/16 is ~17.5 GFLOPs per image, so
 * the scan is materially slower than it was with the student. That is a
 * product decision as well, and it is the reason the model is OPTIONAL:
 * see below.
 *
 * **The latency depends on the interpreter configuration.** The graph is
 * dominated by [196, 768] x [768, 3072] INT8 matrix multiplications, so
 * it only reaches XNNPACK's SDOT/UDOT SIMD kernels on ARM64 when XNNPACK is
 * enabled and the interpreter is allowed more than one thread.
 * [DelegateSelector] does exactly that (XNNPACK on, `min(cores, 4)`
 * threads) and [DelegateSelectorTest] pins it, because the thread count is
 * worth about 2x on this graph and nothing else would have noticed it
 * changing. Measured on x86: 540 ms p50 single-threaded, 282 ms p50 with
 * 4 threads. The expectation on a modern Cortex-A78/X-series core is
 * roughly 140-180 ms; that has NOT been measured on a device.
 *
 * ## The model is optional
 *
 * The graph is only present in a release that staged it. When it is
 * absent the analyzer returns null, the fusion slot stays UNAVAILABLE
 * (NaN), and the tree takes its learned missing branch - exactly as it
 * does today for Engine C.
 */
class VfmProbeAnalyzer @Inject constructor(
    private val context: Context,
    private val engine: LiteRTInferenceEngine,
) {

    companion object {
        /** The SigLIP-B/16 probe graph. */
        const val MODEL_FILE = "siglip_probe_int8.tflite"

        /**
         * The frontend contract. It must match the Python side exactly:
         * `ml/src/export/vfm_frontend.py::vfm_frontend_id` and
         * `ml/scripts/evaluate_vfm_probes.py::Tower.frontend_id`.
         */
        const val FRONTEND_ID = "vfm-siglip224-siglip-base-patch16-224-bicubic-clamp-meanstd-float32"

        /** Input resolution. */
        const val INPUT_SIZE = 224

        /** SigLIP's normalisation: (x / 255 - 0.5) / 0.5. */
        private val MEAN = floatArrayOf(0.5f, 0.5f, 0.5f)
        private val STD = floatArrayOf(0.5f, 0.5f, 0.5f)
    }

    /**
     * Score [bitmap] with the VFM probe.
     *
     * @return P(AI) in `[0, 1]`, or null when the model is unavailable or
     *   the input is unusable. Absence is UNAVAILABLE, never a neutral
     *   constant that would read as a real low measurement.
     */
    fun analyze(bitmap: Bitmap, checkStop: () -> Unit = {}): Float? {
        checkStop()
        if (bitmap.isRecycled || bitmap.width < 2 || bitmap.height < 2) return null

        val contract = runCatching { engine.contract(MODEL_FILE) }.getOrNull() ?: return null
        val shape = contract.inputShape
        requireModelContract(
            shape.size == 4 && shape[0] == 1 && shape[3] == 3 &&
                shape[1] == shape[2] && shape[1] in 64..1024
        ) { "Unsupported VFM probe tensor" }
        val size = shape[1]

        // The tower was trained on BICUBIC resizes, and the frontend
        // contract says bicubic. Android's Bitmap scaling is bilinear, so
        // the resampler is explicit.
        val scaled = BicubicResizer.resize(bitmap, size, size)
        val pixels = IntArray(size * size)
        try {
            scaled.getPixels(pixels, 0, size, 0, 0, size, size)
        } finally {
            if (scaled !== bitmap) scaled.recycle()
        }

        val input = ByteBuffer.allocateDirect(contract.inputBytes).order(ByteOrder.nativeOrder())
        for (c in pixels) {
            input.putFloat(((c shr 16 and 255) / 255f - MEAN[0]) / STD[0])
            input.putFloat(((c shr 8 and 255) / 255f - MEAN[1]) / STD[1])
            input.putFloat(((c and 255) / 255f - MEAN[2]) / STD[2])
        }
        input.flip()
        return runCatching { engine.predict(MODEL_FILE, input, checkStop) }.getOrNull()
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
