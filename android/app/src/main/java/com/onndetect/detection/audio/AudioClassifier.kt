package com.onndetect.detection.audio

import android.content.Context
import android.util.Log
import com.onndetect.inference.LiteRTInferenceEngine
import com.onndetect.detection.visual.CalibrationStore

/**
 * Full audio deepfake verdict: decode -> LFCC -> LCNN-MFM (tflite) ->
 * threshold calibrated at the model's EER operating point.
 */
class AudioClassifier(
    private val context: Context,
    private val engine: LiteRTInferenceEngine,
    private val decoder: AudioDecoder = AudioDecoder(context),
    private val lfcc: LFCCExtractor = LFCCExtractor(),
) {

    companion object {
        const val MODEL_FILE = "lcnn_audio_int8.tflite"
        private const val TAG = "AudioClassifier"
        const val TARGET_RATE = 16000
        const val TARGET_SAMPLES = 64000   // 4 s window -> 401 frames
    }

    data class Verdict(
        val score: Float,          // P(synthetic)
        val isAiGenerated: Boolean,
        val threshold: Float,
    )

    /** @param onStage progress callback (called from a worker thread). */
    fun analyze(uri: android.net.Uri, onStage: (String) -> Unit = {}): Verdict {
        onStage("Decoding audio to PCM…")
        val pcm16k = decoder.decodeMono16k(uri, TARGET_RATE, TARGET_SAMPLES)

        onStage("Extracting LFCC features…")
        val features = lfcc.extract(pcm16k)

        onStage("Running LCNN deepfake model…")
        val score = try {
            engine.predict(MODEL_FILE, features)
        } catch (e: Exception) {
            Log.e(TAG, "LCNN inference failed", e)
            0.5f
        }

        val threshold = CalibrationStore.audioThreshold(context)
        val clamped = score.coerceIn(0f, 1f)
        Log.i(TAG, "score=$clamped threshold=$threshold")
        return Verdict(clamped, clamped >= threshold, threshold)
    }
}
