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
 * Weights for the provisional multimodal video-level decision.
 *
 * Visual frame scores come from the calibrated `XGBoostMetaClassifier` (`threshold = 0.5782`).
 * Audio scores come from `LCNN-MFM` (`audioThreshold = 0.3555`).
 * Temporal cosine discontinuity and BioLip mouth acceleration act as one-sided positive
 * anomaly boosters so that smooth motion (`temporal == 0f`) or real background music never
 * penalize a visually synthetic video below the decision threshold.
 */
object VideoScorePolicy {
    const val FRAME_WEIGHT = 0.65f
    const val AUDIO_WEIGHT = 0.20f
    const val TEMPORAL_WEIGHT = 0.15f

    /**
     * Computes the multimodal video synthetic score in `[0, 1]` without allowing an authentic
     * secondary track (e.g. real background music on a Sora/Runway AI video, or smooth frame-to-frame
     * motion) to drag `visualScore` below `visualThreshold`.
     */
    fun combineMultimodalScore(
        visualScore: Float,
        visualThreshold: Float = 0.578125f,
        audioScore: Float = Float.NaN,
        audioValidatedSuccess: Boolean = false,
        audioThreshold: Float = 0.35546875f,
        temporalScore: Float = Float.NaN,
        lipSyncAnomaly: Float = 0f,
        rppgPhaseAnomaly: Float = Float.NaN,
    ): Float {
        if (!visualScore.isFinite() || visualScore !in 0f..1f) return Float.NaN
        var base = visualScore.coerceIn(0f, 1f)

        // 1. Audio escalation: if the audio track is synthetic (above its calibrated EER threshold),
        // map it into the visual threshold space so a real video with an AI-cloned voice crosses
        // `visualThreshold`. If the audio track is authentic, do NOT penalize `visualScore`.
        // Guard against IEEE 754 Math.max NaN contamination: both visualThreshold and audioThreshold
        // must be valid finite probabilities in [0, 1] before threshold-space mapping is attempted.
        if (visualThreshold.isFinite() && visualThreshold in 0f..1f &&
            audioThreshold.isFinite() && audioThreshold in 0f..1f &&
            audioScore.isFinite() && audioScore in 0f..1f &&
            audioScore >= audioThreshold
        ) {
            val mappedAudio = if (audioThreshold < 0.99f) {
                visualThreshold + ((audioScore - audioThreshold) / (1f - audioThreshold)) * (1f - visualThreshold)
            } else {
                audioScore
            }
            if (mappedAudio.isFinite()) {
                val weight = if (audioValidatedSuccess) 0.70f else 0.35f
                val candidate = (1f - weight) * base + weight * mappedAudio.coerceIn(0f, 1f)
                if (candidate.isFinite() && candidate > base) {
                    base = candidate
                }
            }
        }

        // 2. One-sided temporal & BioLip anomaly boost: high frame-to-frame discontinuity (> 0.30)
        // or high perioral landmark acceleration jitter boosts suspicion toward 1.0, while 0.0
        // (smooth video / no jitter) leaves `base` untouched.
        val tempAnomaly = if (temporalScore.isFinite() && temporalScore > 0.30f) {
            ((temporalScore - 0.30f) / 0.70f).coerceIn(0f, 1f)
        } else 0f
        val lipAnomaly = if (lipSyncAnomaly.isFinite() && lipSyncAnomaly > 0f) {
            lipSyncAnomaly.coerceIn(0f, 1f)
        } else 0f

        val rppgAnom = if (rppgPhaseAnomaly.isFinite() && rppgPhaseAnomaly > 0.35f) {
            ((rppgPhaseAnomaly - 0.35f) / 0.65f).coerceIn(0f, 1f)
        } else 0f

        val boost = (TEMPORAL_WEIGHT * tempAnomaly + 0.15f * lipAnomaly + 0.10f * rppgAnom).coerceIn(0f, 0.35f)
        return (base + (1f - base) * boost).coerceIn(0f, 1f)
    }

    /**
     * Computes Dimensionless Trajectory Tortuosity (2nd-order jerk / 1st-order path length)
     * over unit VFM frame embeddings in `[0f, 1f]` (v11 PRISM, CalibratedVideoMultimodalFusion).
     */
    fun computeVfmTemporalAccelerationAnomaly(embeddings: List<FloatArray>): Float {
        if (embeddings.size < 3) return Float.NaN
        val d = embeddings[0].size
        if (d == 0 || embeddings.any { it.size != d }) return Float.NaN

        val unit = Array(embeddings.size) { i ->
            val v = embeddings[i]
            var normSq = 0.0
            for (x in v) normSq += x.toDouble() * x.toDouble()
            val norm = kotlin.math.sqrt(normSq)
            if (norm > 1e-12) FloatArray(d) { (v[it] / norm).toFloat() } else v.clone()
        }

        var totalKappa = 0.0
        var validCount = 0

        for (t in 1 until embeddings.size - 1) {
            val prev = unit[t - 1]
            val curr = unit[t]
            val next = unit[t + 1]

            var stepPrevSq = 0.0
            var stepNextSq = 0.0
            var jerkSq = 0.0

            for (i in 0 until d) {
                val dp = (curr[i] - prev[i]).toDouble()
                val dn = (next[i] - curr[i]).toDouble()
                val jerk = dn - dp
                stepPrevSq += dp * dp
                stepNextSq += dn * dn
                jerkSq += jerk * jerk
            }

            val stepPrev = kotlin.math.sqrt(stepPrevSq)
            val stepNext = kotlin.math.sqrt(stepNextSq)

            if (stepPrev < 0.75 && stepNext < 0.75 && (stepPrev + stepNext) > 1e-5) {
                val jerkMag = kotlin.math.sqrt(jerkSq)
                val pathLen = stepNext + stepPrev + 1e-8
                val kappa = (jerkMag / pathLen).coerceIn(0.0, 1.0)
                totalKappa += kappa
                validCount++
            }
        }

        if (validCount == 0) return 0f
        val meanKappa = (totalKappa / validCount).toFloat()
        return ((meanKappa - 0.25f) / 0.55f).coerceIn(0f, 1f)
    }
}

/** Sampled-frame and sampled-audio diagnostics, without an untrained video verdict. */
@Singleton
class VideoForensicAnalyzer @Inject constructor(private val context: Context,
    private val videoDecoder: VideoDecoder, private val srmAnalyzer: SRMForensicAnalyzer,
    private val fftAnalyzer: FFTSpectralAnalyzer, private val klAnalyzer: KLCooccurrenceAnalyzer,
    private val metaClassifier: XGBoostMetaClassifier, private val audioClassifier: AudioClassifier,
    private val vfmProbe: com.onndetect.detection.visual.VfmProbeAnalyzer,
    private val bioLipAnalyzer: com.onndetect.detection.visual.BioLipLandmarkAnalyzer) {
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
         * Mean slot-6 (frozen tower) score across the analysed frames, or Float.NaN
         * when the probe did not run.
         */
        val semanticScore: Float = Float.NaN,
        /** BioLip perioral facial landmark temporal micro-jitter. */
        val landmarkJitter: Float = Float.NaN)
    companion object {
        /**
         * Minimum adjacent-frame cosine similarity for a frame to inherit a
         * neighbour keyframe's tower score.
         */
        const val KEYFRAME_SIMILARITY_THRESHOLD = 0.60f

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
        fun failed(status: AnalysisStatus, reason: String?) = Outcome(
            Float.NaN, false, 0f, Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN,
            0, 0, 0, 0, status, reason, semanticScore = Float.NaN, landmarkJitter = Float.NaN
        )
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
                val measuredJitter = bioLip.landmarkJitter
                val landmarkJitter = if (measuredJitter.isFinite()) measuredJitter else DetectionPipeline.NEUTRAL_JITTER
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
                val temporal = if (adjacentCosines.isNotEmpty()) FrameCosineDelta.anomalyScore(adjacentCosines.toList()) else Float.NaN
                var audioScore = Float.NaN
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
                fun avg(values: List<Float>) = if (values.isEmpty()) Float.NaN else values.average().toFloat()
                val threshold = report.threshold
                if (report.validated && threshold != null && frameFusionScores.isNotEmpty()) {
                    // Hybrid pooling: 50% mean across all frames + 50% top-k peak anomaly
                    // In video deepfakes, artifacts frequently cluster in specific burst frames.
                    val meanScore = frameFusionScores.average().toFloat()
                    val k = maxOf(1, frameFusionScores.size / 4)
                    val topKMean = frameFusionScores.sortedDescending().take(k).average().toFloat()
                    val visualScore = (0.50f * meanScore + 0.50f * topKMean).coerceIn(0f, 1f)

                    val lipSyncAnomaly = if (bioLip.hasFace && measuredJitter.isFinite() && measuredJitter > 0.15f) {
                        ((measuredJitter - 0.15f) / 0.85f).coerceIn(0f, 1f)
                    } else 0f

                    // Multimodal blending across visual, audio, temporal, and biomechanical lip sync
                    val audioThr = runCatching {
                        com.onndetect.detection.visual.CalibrationStore.audioThreshold(context)
                    }.getOrDefault(0.35546875f)
                    val overall = VideoScorePolicy.combineMultimodalScore(
                        visualScore = visualScore,
                        visualThreshold = threshold,
                        audioScore = audioScore,
                        audioValidatedSuccess = audioValidatedSuccess,
                        audioThreshold = audioThr,
                        temporalScore = temporal,
                        lipSyncAnomaly = lipSyncAnomaly,
                        rppgPhaseAnomaly = bioLip.rppgPhaseAnomaly,
                    )
                    val provisionalNotice = "Provisional video verdict: heuristic multimodal temporal blend without dedicated video calibration"
                    val note = (listOf(provisionalNotice) + issues).distinct().joinToString("; ")
                    return Outcome(overall, overall >= threshold, 0f, avg(srmScores), avg(fftScores), avg(klScores),
                        avg(flatScores), temporal, audioScore, video.frames.size, video.width, video.height,
                        video.durationMs, AnalysisStatus.SUCCESS, note, true, video.timestampsMs, audioDuration,
                        threshold, report.releaseId, report.frontendId, "provisional-uncalibrated",
                        avg(semanticScores), landmarkJitter = if (bioLip.hasFace && measuredJitter.isFinite()) measuredJitter else Float.NaN)
                }
                return Outcome(Float.NaN, false, 0f, avg(srmScores), avg(fftScores), avg(klScores), avg(flatScores), temporal,
                    audioScore, video.frames.size, video.width, video.height, video.durationMs,
                    AnalysisStatus.INCONCLUSIVE, (if (!report.validated) report.validationReason else "Insufficient usable video frames") +
                        if (issues.isEmpty()) "" else "; " + issues.distinct().joinToString(), false, video.timestampsMs, audioDuration,
                    semanticScore = avg(semanticScores),
                    landmarkJitter = if (bioLip.hasFace && measuredJitter.isFinite()) measuredJitter else Float.NaN)
            } finally { video.frames.forEach { if (!it.isRecycled) it.recycle() } }
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { return failed(AnalysisFailurePolicy.status(e), e.message ?: "Video analysis failed") }
    }
}

