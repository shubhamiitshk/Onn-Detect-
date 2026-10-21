package com.onndetect.detection.video

import android.content.Context
import android.os.SystemClock
import com.onndetect.detection.audio.AudioClassifier
import com.onndetect.detection.fusion.XGBoostMetaClassifier
import com.onndetect.detection.temporal.FrameCosineDelta
import com.onndetect.detection.temporal.VideoDecoder
import com.onndetect.detection.visual.FFTSpectralAnalyzer
import com.onndetect.detection.visual.KLCooccurrenceAnalyzer
import com.onndetect.detection.visual.SRMForensicAnalyzer
import com.onndetect.domain.models.AnalysisStatus
import com.onndetect.domain.pipeline.DetectionPipeline
import com.onndetect.domain.pipeline.FusionFeatureSchema
import com.onndetect.inference.ModelIntegrityChecker
import com.onndetect.inference.ModelIntegrityGate
import com.onndetect.inference.AnalysisFailurePolicy
import java.util.concurrent.CancellationException
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Weights for the video-level decision.
 *
 * A video generator's artifacts are temporal, so the video route needs its
 * OWN operating point rather than the still-image fusion's. These weights
 * and the video threshold are placeholders until a video-level calibration
 * set exists; `TEMPORAL_WEIGHT` is therefore ZERO, because an uncalibrated
 * term in the score is worse than no term at all.
 *
 * To enable the temporal term:
 *   1. assemble a video-level calibration split with real and generated
 *      videos, labelled at the FILE level;
 *   2. fit the weights that maximise the video-level operating point on
 *      that split only;
 *   3. write the fitted weights and the threshold into the release's
 *      calibration asset and re-stage the bundle;
 *   4. set TEMPORAL_WEIGHT to the fitted value here.
 *
 * Until then the temporal signal is computed and REPORTED (it is visible in
 * the signal breakdown) but does not move the verdict.
 */
object VideoScorePolicy {
    const val FRAME_WEIGHT = 0.75f
    const val AUDIO_WEIGHT = 0.25f
    const val TEMPORAL_WEIGHT = 0f
}

/** Sampled-frame and sampled-audio diagnostics, without an untrained video verdict. */
@Singleton
class VideoForensicAnalyzer @Inject constructor(private val context: Context,
    private val videoDecoder: VideoDecoder, private val srmAnalyzer: SRMForensicAnalyzer,
    private val fftAnalyzer: FFTSpectralAnalyzer, private val klAnalyzer: KLCooccurrenceAnalyzer,
    private val metaClassifier: XGBoostMetaClassifier, private val audioClassifier: AudioClassifier,
    private val semanticProbe: com.onndetect.detection.visual.SemanticProbeAnalyzer) {
    data class Outcome(val overallScore: Float, val isAiGenerated: Boolean, val confidence: Float,
        val srmScore: Float, val fftScore: Float, val klScore: Float, val flatnessScore: Float,
        val temporalScore: Float, val audioScore: Float, val framesAnalyzed: Int,
        val width: Int, val height: Int, val durationMs: Long,
        val status: AnalysisStatus = AnalysisStatus.INCONCLUSIVE, val reason: String? = null,
        val modelValidated: Boolean = false, val timestampsMs: List<Long> = emptyList(),
        val audioAnalyzedDurationMs: Long = 0, val decisionThreshold: Float? = null,
        val modelVersion: String = "unknown", val preprocessingVersion: String = "unknown",
        val calibrationVersion: String = "unvalidated",
        /**
         * Mean Engine C ("Lite C") score across the analysed frames, or -1
         * when the probe did not run.
         */
        val semanticScore: Float = -1f)
    fun analyze(uri: android.net.Uri, maxFrames: Int = 10, checkStop: () -> Unit = {}, onStage: (String) -> Unit = {}): Outcome {
        val deadline = SystemClock.elapsedRealtime() + 60_000
        fun checkProgress() {
            if (Thread.currentThread().isInterrupted) throw CancellationException("Video analysis cancelled")
            checkStop(); check(SystemClock.elapsedRealtime() <= deadline) { "Video analysis deadline exceeded" }
        }
        fun failed(status: AnalysisStatus, reason: String?) = Outcome(-1f, false, 0f, -1f, -1f, -1f, -1f, -1f, -1f, 0, 0, 0, 0, status, reason)
        val report = ModelIntegrityGate.report(context, setOf(ModelIntegrityChecker.IMAGE_MODEL))
        if (!report.allOk) return failed(AnalysisStatus.MODEL_FAILURE, report.failures.joinToString())
        try {
            checkProgress(); onStage("Sampling video frames...")
            val video = videoDecoder.sampleFrames(uri, maxFrames, checkStop = ::checkProgress)
            try {
                val srmScores = ArrayList<Float>()
                val fftScores = ArrayList<Float>()
                val flatScores = ArrayList<Float>()
                val klScores = ArrayList<Float>()
                val frameFusionScores = ArrayList<Float>()
                val semanticScores = ArrayList<Float>()
                val issues = ArrayList<String>()
                val resNorm = (minOf(video.width, video.height) / 1024f).coerceIn(0f, 1f)
                video.frames.forEachIndexed { index, bitmap ->
                    checkProgress()
                    val srm = srmAnalyzer.analyze(bitmap, ::checkProgress)
                    srmScores += srm
                    val kl = klAnalyzer.analyze(bitmap)
                    klScores += kl
                    try {
                        val spectral = fftAnalyzer.analyze(bitmap, ::checkProgress)
                        fftScores += spectral.score; flatScores += spectral.spectralFlatness
                        if (report.validated && srm in 0f..1f && kl in 0f..1f) {
                            // Engine C ("Lite C") on THIS frame.
                            //
                            // Slot 6 must be a PER-FRAME measurement: the
                            // fusion is computed per frame, so a video-level
                            // average would be the wrong quantity in the
                            // wrong place. Previously the video route hard-coded
                            // NaN here, which meant every video was scored by
                            // the weakest engine alone - the fusion tree took
                            // its learned "probe unavailable" branch and the
                            // strongest semantic signal was discarded.
                            //
                            // The probe is only run when the release
                            // validated it; otherwise slot 6 stays
                            // UNAVAILABLE (NaN) - never a neutral constant.
                            val probe = if (report.cascadeValidated) {
                                semanticProbe.analyze(bitmap, ::checkProgress)
                            } else null
                            if (probe != null) semanticScores += probe
                            val features = FusionFeatureSchema.build(
                                srm, spectral.score, kl, spectral.spectralFlatness, resNorm, 1f,
                                probe ?: Float.NaN,
                                DetectionPipeline.NEUTRAL_C2PA,
                                DetectionPipeline.NEUTRAL_LIP_SYNC,
                                DetectionPipeline.NEUTRAL_JITTER
                            )
                            frameFusionScores += metaClassifier.predict(features)
                        }
                    } catch (e: CancellationException) { throw e }
                    catch (_: IllegalArgumentException) { issues += "Insufficient spectrum in frame ${index + 1}" }
                    onStage("Frame ${index + 1}/${video.frames.size} analyzed...")
                }
                checkProgress()
                val temporal = if (video.frames.size > 1) FrameCosineDelta.anomalyScore(FrameCosineDelta.similarities(video.frames)) else -1f
                var audioScore = -1f
                var audioDuration = 0L
                var audioValidatedSuccess = false
                if (video.hasAudioTrack) {
                    // Named argument: `windows` sits before the lambda parameters, so the
                    // positional form would try to pass checkProgress as a
                    // window count.
                    val audio = audioClassifier.analyze(uri, checkStop = ::checkProgress) { onStage(it) }
                    audioScore = audio.score; audioDuration = audio.analyzedDurationMs
                    audioValidatedSuccess = audio.modelValidated && audio.status == AnalysisStatus.SUCCESS && audio.score in 0f..1f
                    if (audio.status != AnalysisStatus.SUCCESS) issues += audio.reason ?: "Audio inconclusive"
                }
                fun avg(values: List<Float>) = if (values.isEmpty()) -1f else values.average().toFloat()
                val threshold = report.threshold
                if (report.validated && threshold != null && frameFusionScores.isNotEmpty()) {
                    val visualScore = avg(frameFusionScores)
                    // The video-level blend is policy-driven so the weights can
                    // be fitted on a video-level calibration split and
                    // re-staged, rather than being hard-coded here. TEMPORAL_WEIGHT
                    // is 0 until that calibration exists; the temporal signal
                    // is still computed and reported.
                    val overall = if (audioValidatedSuccess) {
                        ((VideoScorePolicy.FRAME_WEIGHT * visualScore +
                            VideoScorePolicy.AUDIO_WEIGHT * audioScore +
                            VideoScorePolicy.TEMPORAL_WEIGHT * kotlin.math.max(temporal, 0f)).coerceIn(0f, 1f))
                    } else visualScore
                    val note = if (issues.isEmpty()) null else issues.distinct().joinToString()
                    return Outcome(overall, overall >= threshold, 0f, avg(srmScores), avg(fftScores), avg(klScores),
                        avg(flatScores), temporal, audioScore, video.frames.size, video.width, video.height,
                        video.durationMs, AnalysisStatus.SUCCESS, note, true, video.timestampsMs, audioDuration,
                        threshold, report.releaseId, report.frontendId, report.releaseId,
                        avg(semanticScores))
                }
                return Outcome(-1f, false, 0f, avg(srmScores), avg(fftScores), avg(klScores), avg(flatScores), temporal,
                    audioScore, video.frames.size, video.width, video.height, video.durationMs,
                    AnalysisStatus.INCONCLUSIVE, (if (!report.validated) report.validationReason else "Insufficient usable video frames") +
                        if (issues.isEmpty()) "" else "; " + issues.distinct().joinToString(), false, video.timestampsMs, audioDuration,
                    avg(semanticScores))
            } finally { video.frames.forEach { if (!it.isRecycled) it.recycle() } }
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { return failed(AnalysisFailurePolicy.status(e), e.message ?: "Video analysis failed") }
    }
}

