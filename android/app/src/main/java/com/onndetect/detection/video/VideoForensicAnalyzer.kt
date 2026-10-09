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
import com.onndetect.inference.ModelIntegrityChecker
import com.onndetect.inference.ModelIntegrityGate
import com.onndetect.inference.AnalysisFailurePolicy
import java.util.concurrent.CancellationException
import javax.inject.Inject
import javax.inject.Singleton

/** Sampled-frame and sampled-audio diagnostics, without an untrained video verdict. */
@Singleton
class VideoForensicAnalyzer @Inject constructor(private val context: Context,
    private val videoDecoder: VideoDecoder, private val srmAnalyzer: SRMForensicAnalyzer,
    private val fftAnalyzer: FFTSpectralAnalyzer, private val klAnalyzer: KLCooccurrenceAnalyzer,
    private val metaClassifier: XGBoostMetaClassifier, private val audioClassifier: AudioClassifier) {
    data class Outcome(val overallScore: Float, val isAiGenerated: Boolean, val confidence: Float,
        val srmScore: Float, val fftScore: Float, val klScore: Float, val flatnessScore: Float,
        val temporalScore: Float, val audioScore: Float, val framesAnalyzed: Int,
        val width: Int, val height: Int, val durationMs: Long,
        val status: AnalysisStatus = AnalysisStatus.INCONCLUSIVE, val reason: String? = null,
        val modelValidated: Boolean = false, val timestampsMs: List<Long> = emptyList(),
        val audioAnalyzedDurationMs: Long = 0)
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
                val issues = ArrayList<String>()
                video.frames.forEachIndexed { index, bitmap ->
                    checkProgress()
                    srmScores += srmAnalyzer.analyze(bitmap, ::checkProgress)
                    try {
                        val spectral = fftAnalyzer.analyze(bitmap, ::checkProgress)
                        fftScores += spectral.score; flatScores += spectral.spectralFlatness
                    } catch (e: CancellationException) { throw e }
                    catch (_: IllegalArgumentException) { issues += "Insufficient spectrum in frame ${index + 1}" }
                    klScores += klAnalyzer.analyze(bitmap)
                    onStage("Frame ${index + 1}/${video.frames.size} analyzed...")
                }
                checkProgress()
                val temporal = if (video.frames.size > 1) FrameCosineDelta.anomalyScore(FrameCosineDelta.similarities(video.frames)) else -1f
                var audioScore = -1f
                var audioDuration = 0L
                if (video.hasAudioTrack) {
                    val audio = audioClassifier.analyze(uri, ::checkProgress) { onStage(it) }
                    audioScore = audio.score; audioDuration = audio.analyzedDurationMs
                    if (audio.status != AnalysisStatus.SUCCESS) issues += audio.reason ?: "Audio inconclusive"
                }
                fun avg(values: List<Float>) = if (values.isEmpty()) -1f else values.average().toFloat()
                // There is no validated file-level multimodal aggregation.
                // OR and max scores are not container probabilities.
                return Outcome(-1f, false, 0f, avg(srmScores), avg(fftScores), avg(klScores), avg(flatScores), temporal,
                    audioScore, video.frames.size, video.width, video.height, video.durationMs,
                    AnalysisStatus.INCONCLUSIVE, "Video aggregation and sampled-frame coverage are unvalidated" +
                        if (issues.isEmpty()) "" else "; " + issues.distinct().joinToString(), false, video.timestampsMs, audioDuration)
            } finally { video.frames.forEach { if (!it.isRecycled) it.recycle() } }
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) { return failed(AnalysisFailurePolicy.status(e), e.message ?: "Video analysis failed") }
    }
}
