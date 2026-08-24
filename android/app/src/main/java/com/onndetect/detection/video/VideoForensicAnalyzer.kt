package com.onndetect.detection.video

import android.content.Context
import com.onndetect.detection.audio.AudioClassifier
import com.onndetect.detection.fusion.XGBoostMetaClassifier
import com.onndetect.domain.pipeline.DetectionPipeline
import com.onndetect.detection.temporal.FrameCosineDelta
import com.onndetect.detection.temporal.VideoDecoder
import com.onndetect.detection.visual.FFTSpectralAnalyzer
import com.onndetect.detection.visual.KLCooccurrenceAnalyzer
import com.onndetect.detection.visual.SRMForensicAnalyzer
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.abs
import kotlin.math.min

/**
 * Video deepfake analysis:
 *  1. Sample K evenly-spaced frames (VideoDecoder).
 *  2. Run the visual ensemble per frame - SRM-CNN tflite, FFT spectral slope,
 *     KL co-occurrence - and average the signals.
 *  3. Fuse frame-averaged features with the transpiled XGBoost meta model
 *     (same canonical feature contract as images).
 *  4. Temporal consistency via FrameCosineDelta (reported signal).
 *  5. If the container has an audio track, score it with the LCNN detector
 *     and combine: overall = mean(visualMeta, audioScore).
 */
@Singleton
class VideoForensicAnalyzer @Inject constructor(
    private val context: Context,
    private val videoDecoder: VideoDecoder,
    private val srmAnalyzer: SRMForensicAnalyzer,
    private val fftAnalyzer: FFTSpectralAnalyzer,
    private val klAnalyzer: KLCooccurrenceAnalyzer,
    private val metaClassifier: XGBoostMetaClassifier,
    private val audioClassifier: AudioClassifier,
) {

    data class Outcome(
        val overallScore: Float,
        val isAiGenerated: Boolean,
        val confidence: Float,
        val srmScore: Float,
        val fftScore: Float,
        val klScore: Float,
        val flatnessScore: Float,
        val temporalScore: Float,
        val audioScore: Float,   // -1 when no usable audio track
        val framesAnalyzed: Int,
        val width: Int,
        val height: Int,
        val durationMs: Long,
    )

    fun analyze(
        uri: android.net.Uri,
        maxFrames: Int = 10,
        onStage: (String) -> Unit = {},
    ): Outcome {
        onStage("Sampling video frames…")
        val video = videoDecoder.sampleFrames(uri, maxFrames)

        onStage("Per-frame forensics (${video.frames.size} frames)…")
        var srm = 0f
        var fft = 0f
        var flat = 0f
        var kl = 0f
        video.frames.forEachIndexed { i, bmp ->
            srm += srmAnalyzer.analyze(bmp)
            val sp = fftAnalyzer.analyze(bmp)
            fft += sp.score
            flat += sp.spectralFlatness
            kl += klAnalyzer.analyze(bmp)
            onStage("Frame ${i + 1}/${video.frames.size} analyzed…")
        }
        val n = video.frames.size.toFloat()
        srm /= n; fft /= n; flat /= n; kl /= n

        onStage("Temporal consistency check…")
        val sims = FrameCosineDelta.similarities(video.frames)
        val temporal = FrameCosineDelta.anomalyScore(sims)

        onStage("Fusing signals (XGBoost meta-classifier)…")
        val resNorm = (min(video.width, video.height) / 1024f).coerceIn(0f, 1f)
        // Video containers are re-encoded/compressed by nature.
        // 10-feature contract (must match XGBoostMetaClassifier.kt): the six
        // visual features are frame-averaged; provenance/temporal slots use
        // the same neutral authentic-side constants as the image pipeline
        // because per-file EXIF/C2PA and landmark signals are not extracted
        // for decoded frames yet.
        val features = floatArrayOf(
            srm, fft, kl, flat, resNorm, 1f,
            DetectionPipeline.NEUTRAL_EXIF,
            DetectionPipeline.NEUTRAL_C2PA,
            DetectionPipeline.NEUTRAL_LIP_SYNC,
            DetectionPipeline.NEUTRAL_JITTER,
        )
        val metaScore = metaClassifier.predict(features).coerceIn(0f, 1f)

        var audioScore = -1f
        var isAudioSynthetic = false
        if (video.hasAudioTrack) {
            try {
                onStage("Scanning embedded audio…")
                val verdict = audioClassifier.analyze(uri) { }
                audioScore = verdict.score
                isAudioSynthetic = verdict.isAiGenerated
            } catch (_: Exception) {
                audioScore = -1f // silent / unsupported tracks are normal
            }
        }

        val visualThreshold = com.onndetect.detection.visual.CalibrationStore
            .metaDecisionThreshold(context)
        val isVisualSynthetic = metaScore >= visualThreshold

        // Multimodal threat model: if EITHER the embedded audio track (voice clone)
        // or the visual track (generative frames/face swap) is synthetic, the container is flagged.
        val isAiGenerated = isVisualSynthetic || isAudioSynthetic

        // Overall risk score reflects the maximum anomaly across evaluated modalities.
        val overall = if (audioScore >= 0f) maxOf(metaScore, audioScore) else metaScore

        return Outcome(
            overallScore = overall,
            isAiGenerated = isAiGenerated,
            confidence = abs(overall - 0.5f) * 2f,
            srmScore = srm,
            fftScore = fft,
            klScore = kl,
            flatnessScore = flat,
            temporalScore = temporal,
            audioScore = audioScore,
            framesAnalyzed = video.frames.size,
            width = video.width,
            height = video.height,
            durationMs = video.durationMs,
        )
    }
}
