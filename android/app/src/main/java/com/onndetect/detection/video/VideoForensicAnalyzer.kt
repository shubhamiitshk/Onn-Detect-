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
    private val vfmProbe: com.onndetect.detection.visual.VfmProbeAnalyzer,
    private val bioLipAnalyzer: com.onndetect.detection.visual.BioLipLandmarkAnalyzer = com.onndetect.detection.visual.BioLipLandmarkAnalyzer()) {
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
         * Mean slot-6 (frozen tower) score across the analysed frames, or -1
         * when the probe did not run.
         */
        val semanticScore: Float = -1f,
        /** BioLip perioral facial landmark temporal micro-jitter. */
        val landmarkJitter: Float = DetectionPipeline.NEUTRAL_JITTER)
    companion object {
        /**
         * Minimum adjacent-frame cosine similarity for a frame to inherit a
         * neighbour keyframe's tower score.
         */
        const val KEYFRAME_SIMILARITY_THRESHOLD = 0.98f

        /** The maximum number of keyframes the tower is run on. */
        const val MAX_PROBE_KEYFRAMES = 4

        /**
         * Selects up to [maxKeyframes] indices across [frameCount] frames:
         * the first, the middle, the last, and the frame immediately after
         * the lowest adjacent cosine similarity (the scene cut or the peak
         * motion).
         */
        internal fun selectProbeKeyframes(
            frameCount: Int,
            adjacentCosines: FloatArray,
            maxKeyframes: Int = MAX_PROBE_KEYFRAMES,
        ): Set<Int> {
            if (frameCount <= 0) return emptySet()
            if (frameCount <= maxKeyframes) return (0 until frameCount).toSet()
            val selected = linkedSetOf(0, frameCount / 2, frameCount - 1)
            if (adjacentCosines.isNotEmpty()) {
                val minIdx = adjacentCosines.indices.minByOrNull { adjacentCosines[it] } ?: 0
                val transitionFrame = (minIdx + 1).coerceIn(0, frameCount - 1)
                selected += transitionFrame
            }
            // If the transition frame collided with the first, the middle or
            // the last, fill the remaining budget with the quarter points.
            if (selected.size < maxKeyframes) selected += (frameCount / 4).coerceIn(0, frameCount - 1)
            if (selected.size < maxKeyframes) selected += ((3 * frameCount) / 4).coerceIn(0, frameCount - 1)
            return selected
        }

        /**
         * Evaluates [scoreFrame] on the keyframes, and propagates a
         * keyframe's score to any frame whose path to it crosses no
         * adjacent similarity below [similarityThreshold].
         *
         * A frame that no keyframe reaches without crossing such a
         * transition is scored directly: a scene cut means the frame
         * genuinely looks different.
         */
        internal inline fun evaluateWithKeyframeReuse(
            frameCount: Int,
            adjacentCosines: FloatArray,
            similarityThreshold: Float = KEYFRAME_SIMILARITY_THRESHOLD,
            scoreFrame: (Int) -> Float?,
        ): FloatArray {
            val scores = FloatArray(frameCount) { Float.NaN }
            if (frameCount == 0) return scores
            val keyframes = selectProbeKeyframes(frameCount, adjacentCosines)
            for (idx in keyframes) {
                scores[idx] = scoreFrame(idx) ?: Float.NaN
            }
            for (i in 0 until frameCount) {
                if (!scores[i].isNaN()) continue
                // The nearest EVALUATED keyframe reachable without crossing
                // a transition.
                var bestDist = Int.MAX_VALUE
                var inherited = Float.NaN
                for (k in keyframes) {
                    val kScore = scores[k]
                    if (kScore.isNaN()) continue
                    val lo = minOf(i, k)
                    val hi = maxOf(i, k)
                    var connected = true
                    for (step in lo until hi) {
                        if (step >= adjacentCosines.size || adjacentCosines[step] < similarityThreshold) {
                            connected = false
                            break
                        }
                    }
                    val dist = hi - lo
                    if (connected && dist < bestDist) {
                        bestDist = dist
                        inherited = kScore
                    }
                }
                scores[i] = if (!inherited.isNaN()) inherited else (scoreFrame(i) ?: Float.NaN)
            }
            return scores
        }
    }

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
                // The adjacent-frame similarities are computed ONCE: the
                // keyframe selector needs them, and the temporal score uses
                // them. Previously they were computed after the frame loop.
                val adjacentCosines = if (video.frames.size > 1) {
                    FrameCosineDelta.similarities(video.frames).toFloatArray()
                } else {
                    FloatArray(0)
                }
                // The tower on the KEYFRAMES, with the score propagated to
                // the near-identical frames. A frame that a scene cut
                // separates from every keyframe is scored directly, because
                // it genuinely looks different.
                val probeScores = if (report.cascadeValidated) {
                    evaluateWithKeyframeReuse(video.frames.size, adjacentCosines) { idx ->
                        checkProgress()
                        vfmProbe.analyze(video.frames[idx], ::checkProgress)
                    }
                } else {
                    FloatArray(video.frames.size) { Float.NaN }
                }
                val bioLip = bioLipAnalyzer.analyzeFrames(video.frames)
                val landmarkJitter = bioLip.landmarkJitter
                if (bioLip.hasFace && bioLip.notes != null) {
                    issues += bioLip.notes
                }
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
                            // The frozen tower on THIS frame.
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
                            //
                            // The score for THIS frame comes from the
                            // keyframe evaluation: either it was measured,
                            // or it was inherited from a near-identical
                            // neighbour.
                            val probe = probeScores.getOrNull(index)?.takeIf { it.isFinite() }
                            if (probe != null) semanticScores += probe
                            val features = FusionFeatureSchema.build(
                                srm, spectral.score, kl, spectral.spectralFlatness, resNorm, 1f,
                                probe ?: Float.NaN,
                                DetectionPipeline.NEUTRAL_C2PA,
                                DetectionPipeline.NEUTRAL_LIP_SYNC,
                                landmarkJitter
                            )
                            frameFusionScores += metaClassifier.predict(features)
                        }
                    } catch (e: CancellationException) { throw e }
                    catch (_: IllegalArgumentException) { issues += "Insufficient spectrum in frame ${index + 1}" }
                    onStage("Frame ${index + 1}/${video.frames.size} analyzed...")
                }
                checkProgress()
                val temporal = if (adjacentCosines.isNotEmpty()) FrameCosineDelta.anomalyScore(adjacentCosines.toList()) else -1f
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
                        avg(semanticScores), landmarkJitter)
                }
                return Outcome(-1f, false, 0f, avg(srmScores), avg(fftScores), avg(klScores), avg(flatScores), temporal,
                    audioScore, video.frames.size, video.width, video.height, video.durationMs,
                    AnalysisStatus.INCONCLUSIVE, (if (!report.validated) report.validationReason else "Insufficient usable video frames") +
                        if (issues.isEmpty()) "" else "; " + issues.distinct().joinToString(), false, video.timestampsMs, audioDuration,
                    semanticScore = avg(semanticScores), landmarkJitter = landmarkJitter)
            } finally { video.frames.forEach { if (!it.isRecycled) it.recycle() } }
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { return failed(AnalysisFailurePolicy.status(e), e.message ?: "Video analysis failed") }
    }
}

