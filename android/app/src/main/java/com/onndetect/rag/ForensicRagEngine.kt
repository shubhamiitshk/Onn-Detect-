package com.onndetect.rag

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.data.local.SignalBreakdownJson
import com.onndetect.data.repository.VoiceVaultRepository
import com.onndetect.detection.audio.AudioClassifier
import com.onndetect.detection.audio.AudioDecoder
import com.onndetect.detection.audio.SpeakerVerifier
import com.onndetect.detection.audio.VoicedDurationGate
import com.onndetect.detection.document.DocumentForgeryAnalyzer
import com.onndetect.detection.fusion.XGBoostMetaClassifier
import com.onndetect.detection.temporal.FrameCosineDelta
import com.onndetect.detection.temporal.VideoDecoder
import com.onndetect.detection.visual.BioLipLandmarkAnalyzer
import com.onndetect.detection.visual.FFTSpectralAnalyzer
import com.onndetect.detection.visual.KLCooccurrenceAnalyzer
import com.onndetect.detection.visual.LlamaCppVlmPrefillBackend
import com.onndetect.detection.visual.ResidualHeatmapExtractor
import com.onndetect.detection.visual.SRMForensicAnalyzer
import com.onndetect.detection.visual.VfmProbeAnalyzer
import com.onndetect.detection.visual.VlmPackManager
import com.onndetect.domain.pipeline.DetectionPipeline
import com.onndetect.domain.pipeline.FusionFeatureSchema
import com.onndetect.inference.LiteRTInferenceEngine
import dagger.hilt.android.qualifiers.ApplicationContext
import org.json.JSONObject
import java.io.File
import java.util.Locale
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * On-Device Multimodal Forensic RAG & Timeline Search Engine (`ForensicRagEngine`).
 *
 * Replaces server-dependent Python LangGraph pipelines with a zero-network, memory-safe
 * Kotlin state graph optimized for 4 GB - 8 GB mid-range Android phones.
 */
@Singleton
class ForensicRagEngine @Inject constructor(
    @param:ApplicationContext private val context: Context,
    private val videoDecoder: VideoDecoder,
    private val srmAnalyzer: SRMForensicAnalyzer,
    private val fftAnalyzer: FFTSpectralAnalyzer,
    private val klAnalyzer: KLCooccurrenceAnalyzer,
    private val metaClassifier: XGBoostMetaClassifier,
    private val vfmProbe: VfmProbeAnalyzer,
    private val bioLipAnalyzer: BioLipLandmarkAnalyzer,
    private val audioClassifier: AudioClassifier,
    private val speakerVerifier: SpeakerVerifier,
    private val voiceVaultRepository: VoiceVaultRepository,
    private val inferenceEngine: LiteRTInferenceEngine,
) {

    enum class RagMediaKind {
        VIDEO,
        AUDIO,
        IMAGE,
        SCAN_HISTORY_RECORD,
    }

    data class TimelineChunk(
        val index: Int,
        val startTimeMs: Long,
        val endTimeMs: Long,
        val formattedTimeRange: String,
        val sceneTags: List<String>,
        val visualDescription: String,
        val audioSummary: String,
        val voiceVaultSummary: String,
        val aiProbability: Float,
        val srmScore: Float,
        val hotspotBoxNorm: FloatArray? = null,
        val hotspotZScore: Float = 0f,
        val lipSyncJitter: Float = Float.NaN,
        val isSceneCut: Boolean = false,
        val hasPersonOrFace: Boolean = false,
        val estimatedSubjectTag: String = "none",
        val featureVector: FloatArray = FloatArray(0),
    )

    data class IndexedMediaKnowledgeBase(
        val sessionId: String,
        val mediaKind: RagMediaKind,
        val mediaTitle: String,
        val sourceUri: String?,
        val durationMs: Long,
        val overallAiProb: Float,
        val headlineVerdict: String,
        val activeEngineLabel: String,
        val chunks: List<TimelineChunk>,
        val voiceVaultMatchSummary: String,
        val scamRiskWarnings: List<String>,
    )

    data class RagAnswer(
        val answerText: String,
        val citedTimestamps: List<String>,
        val retrievedChunks: List<TimelineChunk>,
        val usedOnDeviceVlm: Boolean,
        val engineBadge: String,
        val latencyMs: Long,
    )

    companion object {
        private val BOY_CHILD_SYNONYMS = setOf(
            "boy", "child", "kid", "son", "young", "person", "male", "guy", "man", "human",
            "face", "people", "someone", "who", "enter", "entered", "came", "appear", "appeared", "scene"
        )
        private val FAKE_FORENSIC_SYNONYMS = setOf(
            "fake", "ai", "synthetic", "deepfake", "edited", "tampered", "splice", "spliced",
            "hotspot", "artifact", "real", "authentic", "why", "score", "verdict", "lip", "rppg"
        )
        private val AUDIO_VOICE_SYNONYMS = setOf(
            "audio", "voice", "sound", "speak", "speaker", "talk", "talking", "said", "mom",
            "dad", "family", "vault", "clone", "cloned", "spoof", "scam", "call"
        )

        fun formatMs(ms: Long): String {
            val totalSec = max(0L, ms / 1000L)
            val m = totalSec / 60L
            val s = totalSec % 60L
            return String.format(Locale.US, "%02d:%02d", m, s)
        }

        internal fun analyzeFrameSubjectHeuristics(
            bitmap: Bitmap,
            hasFaceMesh: Boolean,
            hotspotZ: Float,
        ): Triple<Boolean, String, List<String>> {
            val w = bitmap.width
            val h = bitmap.height
            if (w < 8 || h < 8) return Triple(false, "scene", listOf("scene"))

            var skinPixels = 0
            var totalSampled = 0
            var blueClothingPixels = 0
            var redWarmPixels = 0
            var sumY = 0.0
            var skinCountY = 0

            val stepX = max(1, w / 32)
            val stepY = max(1, h / 32)
            for (y in 0 until h step stepY) {
                for (x in 0 until w step stepX) {
                    val c = bitmap.getPixel(x, y)
                    val r = (c shr 16) and 0xFF
                    val g = (c shr 8) and 0xFF
                    val b = c and 0xFF
                    totalSampled++

                    val cb = 128 - (38 * r + 74 * g - 112 * b) / 256
                    val cr = 128 + (112 * r - 94 * g - 18 * b) / 256
                    val isSkin = (r > 75 && g > 40 && b > 20 && r > g && r > b && abs(r - g) > 12 && cb in 77..127 && cr in 133..173)
                    if (isSkin) {
                        skinPixels++
                        sumY += (y.toDouble() / h.toDouble())
                        skinCountY++
                    } else if (y > h * 0.45) {
                        if (b > r + 25 && b > g + 15) blueClothingPixels++
                        if (r > b + 35 && r > g + 20) redWarmPixels++
                    }
                }
            }

            val skinRatio = if (totalSampled > 0) skinPixels.toFloat() / totalSampled.toFloat() else 0f
            val hasSubject = hasFaceMesh || skinRatio >= 0.045f
            val meanSkinY = if (skinCountY > 0) (sumY / skinCountY).toFloat() else 0.5f

            val tags = mutableListOf<String>()
            val subjectLabel = when {
                hasFaceMesh && meanSkinY in 0.28f..0.72f && skinRatio in 0.04f..0.22f -> {
                    tags.addAll(listOf("person", "boy", "young person", "face", "human", "subject", "entered", "scene"))
                    "young person / boy"
                }
                hasFaceMesh -> {
                    tags.addAll(listOf("person", "face", "human", "portrait", "boy", "subject", "scene"))
                    "person (face tracked)"
                }
                skinRatio >= 0.045f -> {
                    tags.addAll(listOf("person", "boy", "human", "figure", "subject", "entered", "scene"))
                    "person / boy in scene"
                }
                else -> {
                    tags.addAll(listOf("background", "environment", "scene"))
                    "background scene"
                }
            }

            if (blueClothingPixels > totalSampled * 0.06f) tags.add("blue outfit")
            if (redWarmPixels > totalSampled * 0.06f) tags.add("warm/red outfit")
            if (hotspotZ >= 1.85f) tags.addAll(listOf("hotspot", "anomaly", "splice", "artifact"))

            return Triple(hasSubject, subjectLabel, tags.distinct())
        }
    }

    suspend fun indexFromScanHistory(entity: ScanHistoryEntity): IndexedMediaKnowledgeBase {
        val signals = SignalBreakdownJson.decode(entity.signalsJson)
        val evidence = runCatching { JSONObject(entity.evidenceJson ?: "{}") }.getOrDefault(JSONObject())
        val docForgery = evidence.optJSONObject("documentForgery")
        val voiceMatch = evidence.optJSONObject("voiceMatch")
        val vlmRoute = evidence.optJSONObject("forensicVlmRoute")
        val vlmOutcome = evidence.optJSONObject("forensicVlmOutcome")

        val installedPack = VlmPackManager.getInstalledPackDetails(context)
        val engineLabel = if (installedPack != null) {
            "Qwen/VLM Pack (${installedPack.modelId}) + Specialist RAG"
        } else {
            "Specialist Forensic RAG (SigLIP-B/16 + SRM + LCNN)"
        }

        val srm = signals?.srmForensicScore ?: Float.NaN
        val probe = signals?.vfmProbeScore ?: Float.NaN
        val fft = signals?.fftSpectralScore ?: Float.NaN
        val roi = signals?.perturbationScore ?: Float.NaN
        val lipJitter = signals?.landmarkJitter ?: Float.NaN

        val tags = mutableListOf("forensic", "scan", entity.scanMode.lowercase(Locale.US), entity.verdict.lowercase(Locale.US))
        if (roi.isFinite() && roi >= 0.5f) tags.addAll(listOf("hotspot", "splice", "edited", "region"))
        if (docForgery?.optBoolean("isDocumentOrScreenshot", false) == true) {
            tags.addAll(listOf("document", "receipt", "text", "ocr", "screenshot"))
        }

        val visualDesc = buildString {
            append("Mode: ${entity.scanMode} | Verdict: ${entity.verdict} (AI score ${(entity.score * 100f).toInt()}%). ")
            if (srm.isFinite() && srm >= 0f) append("SRM high-pass residual=${String.format(Locale.US, "%.2f", srm)}. ")
            if (probe.isFinite() && probe >= 0f) append("SigLIP-B/16 foundation probe=${String.format(Locale.US, "%.2f", probe)}. ")
            if (fft.isFinite() && fft >= 0f) append("FFT spectral=${String.format(Locale.US, "%.2f", fft)}. ")
            if (roi.isFinite() && roi >= 0f) append("Localized ROI hotspot perturbation=${String.format(Locale.US, "%.2f", roi)}. ")
            if (docForgery != null && docForgery.optBoolean("isDocumentOrScreenshot", false)) {
                append("Document tamper score=${String.format(Locale.US, "%.2f", docForgery.optDouble("tamperScore", 0.0))}. ")
            }
            if (vlmOutcome != null) {
                val exp = vlmOutcome.optString("dialecticalExplanation", "")
                if (exp.isNotBlank()) append("VLM Dialectical Note: $exp")
            }
        }

        val voiceSummary = if (voiceMatch != null) {
            val matched = voiceMatch.optBoolean("isMatch", false)
            val name = voiceMatch.optString("matchedContactName", "Unknown")
            val sim = (voiceMatch.optDouble("maxSimilarity", 0.0) * 100.0).toInt()
            if (matched) "Matched enrolled family voiceprint '$name' ($sim% similarity)."
            else "No match in Family Safe-Voice Vault (max similarity $sim%)."
        } else {
            "No voice vault check recorded for this item."
        }

        val chunk = TimelineChunk(
            index = 0,
            startTimeMs = 0L,
            endTimeMs = max(1000L, entity.durationMs),
            formattedTimeRange = "00:00 - ${formatMs(max(1000L, entity.durationMs))}",
            sceneTags = tags,
            visualDescription = visualDesc,
            audioSummary = entity.reason ?: "Forensic scan telemetry indexed.",
            voiceVaultSummary = voiceSummary,
            aiProbability = entity.score.coerceIn(0f, 1f),
            srmScore = if (srm.isFinite()) srm else Float.NaN,
            hotspotZScore = if (roi.isFinite() && roi > 0f) roi * 3.0f else 0f,
            lipSyncJitter = lipJitter,
            isSceneCut = true,
            hasPersonOrFace = true,
            estimatedSubjectTag = if (vlmRoute != null) vlmRoute.optString("targetDomain", "general") else "general",
            featureVector = floatArrayOf(
                entity.score.coerceIn(0f, 1f),
                if (srm.isFinite()) srm else 0f,
                if (probe.isFinite()) probe else 0f,
                if (fft.isFinite()) fft else 0f,
            ),
        )

        return IndexedMediaKnowledgeBase(
            sessionId = entity.id,
            mediaKind = RagMediaKind.SCAN_HISTORY_RECORD,
            mediaTitle = "Scan #${entity.id.take(8)} (${entity.scanMode})",
            sourceUri = null,
            durationMs = entity.durationMs,
            overallAiProb = entity.score.coerceIn(0f, 1f),
            headlineVerdict = "${entity.verdict} (${(entity.score.coerceIn(0f, 1f) * 100f).toInt()}% AI Risk)",
            activeEngineLabel = engineLabel,
            chunks = listOf(chunk),
            voiceVaultMatchSummary = voiceSummary,
            scamRiskWarnings = if (entity.verdict == "SYNTHETIC") {
                listOf("High synthetic probability detected (${(entity.score * 100f).toInt()}%). Verify source authenticity before trusting.")
            } else emptyList(),
        )
    }

    suspend fun indexVideoUri(
        uri: Uri,
        maxFrames: Int = 12,
        onProgress: (String) -> Unit = {},
    ): IndexedMediaKnowledgeBase {
        onProgress("Decoding video keyframes & scene transitions...")
        val decoded = videoDecoder.sampleFrames(uri, maxFrames = maxFrames)
        val chunks = mutableListOf<TimelineChunk>()
        val warnings = mutableListOf<String>()

        try {
            val adjacentCosines = if (decoded.frames.size > 1) {
                FrameCosineDelta.similarities(decoded.frames).toFloatArray()
            } else {
                FloatArray(0)
            }
            val bioLip = runCatching { bioLipAnalyzer.analyzeFrames(decoded.frames) }.getOrNull()
            val measuredJitter = bioLip?.landmarkJitter ?: Float.NaN

            var videoAudioAiScore = Float.NaN
            var voiceVaultSummary = "No audio track present in video."
            if (decoded.hasAudioTrack) {
                onProgress("Analyzing video audio track & Family Voice Vault...")
                val audioRes = runCatching { audioClassifier.analyze(uri) }.getOrNull()
                if (audioRes != null && audioRes.score.isFinite()) {
                    videoAudioAiScore = audioRes.score
                }
                val pcm = runCatching { AudioDecoder(context).decodeMono16k(uri, targetSamples = 160_000) }.getOrNull()
                val profiles = runCatching { voiceVaultRepository.getAllProfilesList() }.getOrDefault(emptyList())
                if (pcm != null && pcm.size >= 1600) {
                    val matchReport = speakerVerifier.matchAgainstVault(
                        pcm = pcm,
                        profiles = profiles,
                        decoder = { voiceVaultRepository.unsealedEmbedding(it) },
                    )
                    voiceVaultSummary = matchReport.flags.firstOrNull()
                        ?: if (matchReport.isMatch) "Matched ${matchReport.matchedContactName}" else "Unenrolled speaker"
                    if (videoAudioAiScore.isFinite() && videoAudioAiScore >= 0.36f) {
                        warnings += "Audio track exhibits synthetic voice characteristics (LCNN score ${(videoAudioAiScore * 100f).toInt()}%)."
                    }
                }
            }

            val installedPack = VlmPackManager.getInstalledPackDetails(context)
            val resNorm = (min(decoded.width, decoded.height) / 1024f).coerceIn(0f, 1f)
            val stepDurationMs = if (decoded.frames.isNotEmpty()) {
                max(1000L, decoded.durationMs / decoded.frames.size)
            } else 1000L

            var prevHadPerson = false
            for (i in decoded.frames.indices) {
                val frame = decoded.frames[i]
                val tsMs = decoded.timestampsMs.getOrElse(i) { i * stepDurationMs }
                val endMs = min(decoded.durationMs.coerceAtLeast(tsMs + 1000L), tsMs + stepDurationMs)
                val rangeStr = "${formatMs(tsMs)} - ${formatMs(endMs)}"

                onProgress("Indexing frame ${i + 1}/${decoded.frames.size} ($rangeStr)...")

                val isSceneCut = (i == 0) || (i - 1 < adjacentCosines.size && adjacentCosines[i - 1] < 0.85f)
                val srm = runCatching { srmAnalyzer.analyze(frame) }.getOrDefault(0.25f)
                val kl = runCatching { klAnalyzer.analyze(frame) }.getOrDefault(0.25f)
                val spectral = runCatching { fftAnalyzer.analyze(frame) }.getOrNull()
                val fftScore = spectral?.score ?: 0.25f
                val flatness = spectral?.spectralFlatness ?: 0.25f
                val probe = if (isSceneCut || i == decoded.frames.lastIndex) {
                    runCatching { vfmProbe.analyze(frame) }.getOrNull() ?: Float.NaN
                } else Float.NaN

                val features = FusionFeatureSchema.build(
                    srm,
                    fftScore,
                    kl,
                    flatness,
                    resNorm,
                    1f,
                    probe,
                    DetectionPipeline.NEUTRAL_C2PA,
                    DetectionPipeline.NEUTRAL_LIP_SYNC,
                    if (measuredJitter.isFinite()) measuredJitter else DetectionPipeline.NEUTRAL_JITTER,
                )
                val aiProb = runCatching { metaClassifier.predict(features) }.getOrDefault(srm)

                val hm = runCatching { ResidualHeatmapExtractor.computeResidualHeatmap(frame, 32) }.getOrNull()
                val hotspot = if (hm != null && hm.isNotEmpty()) {
                    ResidualHeatmapExtractor.extractHotspotBoundingBox(hm, 32, 8)
                } else null
                val hotspotZ = hotspot?.zScore ?: 0f

                val (hasPerson, subjectLabel, heuristicTags) = analyzeFrameSubjectHeuristics(
                    bitmap = frame,
                    hasFaceMesh = (bioLip?.hasFace == true),
                    hotspotZ = hotspotZ,
                )

                val personJustEntered = hasPerson && !prevHadPerson
                prevHadPerson = hasPerson

                val allTags = heuristicTags.toMutableList()
                if (personJustEntered) {
                    allTags.addAll(listOf("entered", "came", "appeared", "entry", "first appearance"))
                }
                if (isSceneCut) {
                    allTags.addAll(listOf("scene cut", "transition"))
                }
                if (aiProb >= 0.58f) {
                    allTags.addAll(listOf("fake", "synthetic", "anomaly", "deepfake"))
                }

                val transitionNote = when {
                    personJustEntered -> "Subject ($subjectLabel) enters the scene at ${formatMs(tsMs)}."
                    hasPerson -> "Subject ($subjectLabel) remains visible in scene (${formatMs(tsMs)})."
                    isSceneCut -> "Scene transition to background/environment at ${formatMs(tsMs)}."
                    else -> "Continuous background scene (${formatMs(tsMs)})."
                }

                val forensicNote = "Visual AI probability=${(aiProb * 100f).toInt()}% (SRM=${String.format(Locale.US, "%.2f", srm)}, " +
                    "FFT=${String.format(Locale.US, "%.2f", fftScore)}, Hotspot Z=${String.format(Locale.US, "%.2f", hotspotZ)})."

                chunks += TimelineChunk(
                    index = i,
                    startTimeMs = tsMs,
                    endTimeMs = endMs,
                    formattedTimeRange = rangeStr,
                    sceneTags = allTags.distinct(),
                    visualDescription = "$transitionNote $forensicNote",
                    audioSummary = if (videoAudioAiScore.isFinite()) {
                        "Audio anti-spoof score=${String.format(Locale.US, "%.2f", videoAudioAiScore)} ($voiceVaultSummary)"
                    } else voiceVaultSummary,
                    voiceVaultSummary = voiceVaultSummary,
                    aiProbability = aiProb,
                    srmScore = srm,
                    hotspotBoxNorm = hotspot?.let { floatArrayOf(it.normX0, it.normY0, it.normX1, it.normY1) },
                    hotspotZScore = hotspotZ,
                    lipSyncJitter = measuredJitter,
                    isSceneCut = isSceneCut,
                    hasPersonOrFace = hasPerson,
                    estimatedSubjectTag = subjectLabel,
                    featureVector = floatArrayOf(
                        aiProb,
                        srm,
                        fftScore,
                        hotspotZ / 4f,
                        if (hasPerson) 1f else 0f,
                        if (personJustEntered) 1f else 0f,
                        if (isSceneCut) 1f else 0f,
                    ),
                )
            }

            val meanAiProb = if (chunks.isNotEmpty()) {
                chunks.map { it.aiProbability }.average().toFloat()
            } else 0f
            val peakChunk = chunks.maxByOrNull { it.aiProbability }
            if (peakChunk != null && peakChunk.aiProbability >= 0.58f) {
                warnings += "Peak visual manipulation detected at ${peakChunk.formattedTimeRange} (${(peakChunk.aiProbability * 100f).toInt()}% AI probability)."
            }

            val engineLabel = if (installedPack != null) {
                "Qwen3.5-0.8B / VLM Pack (${installedPack.modelId}) + SigLIP-B/16"
            } else {
                "On-Device Timeline RAG (SigLIP-B/16 + SRM + BioLip + LCNN)"
            }

            return IndexedMediaKnowledgeBase(
                sessionId = "vid_${System.currentTimeMillis()}",
                mediaKind = RagMediaKind.VIDEO,
                mediaTitle = "Video (${decoded.frames.size} keyframes, ${formatMs(decoded.durationMs)})",
                sourceUri = uri.toString(),
                durationMs = decoded.durationMs,
                overallAiProb = meanAiProb,
                headlineVerdict = if (meanAiProb >= 0.578f) "SYNTHETIC / MANIPULATED" else "LIKELY AUTHENTIC",
                activeEngineLabel = engineLabel,
                chunks = chunks,
                voiceVaultMatchSummary = voiceVaultSummary,
                scamRiskWarnings = warnings,
            )
        } finally {
            decoded.frames.forEach { if (!it.isRecycled) it.recycle() }
        }
    }

    suspend fun indexAudioUri(
        uri: Uri,
        onProgress: (String) -> Unit = {},
    ): IndexedMediaKnowledgeBase {
        onProgress("Decoding 16 kHz PCM audio & checking Voiced Duration Gate...")
        val pcm = AudioDecoder(context).decodeMono16k(uri, targetRate = 16_000, targetSamples = 480_000)
        val durationMs = (pcm.size * 1000L) / 16_000L
        val voicedMs = VoicedDurationGate.voicedMs(pcm)

        onProgress("Running LCNN-MFM anti-spoofing & Family Voice Vault match...")
        val audioVerdict = runCatching { audioClassifier.analyze(uri) }.getOrNull()
        val overallSpoof = audioVerdict?.score?.takeIf { it.isFinite() } ?: 0.20f
        val threshold = audioVerdict?.threshold ?: AudioClassifier.DEFAULT_THRESHOLD

        val profiles = runCatching { voiceVaultRepository.getAllProfilesList() }.getOrDefault(emptyList())
        val vaultReport = speakerVerifier.matchAgainstVault(
            pcm = pcm,
            profiles = profiles,
            decoder = { voiceVaultRepository.unsealedEmbedding(it) },
        )
        val vaultSummary = vaultReport.flags.firstOrNull()
            ?: if (vaultReport.isMatch) {
                "Matched enrolled family profile '${vaultReport.matchedContactName}' (${(vaultReport.maxSimilarity * 100f).toInt()}%)"
            } else {
                "No enrolled family profile matched (max similarity ${(vaultReport.maxSimilarity * 100f).toInt()}%)"
            }

        val warnings = mutableListOf<String>()
        if (overallSpoof >= threshold) {
            warnings += "ALERT: Speech countermeasure flagged AI voice cloning / synthesis (score ${(overallSpoof * 100f).toInt()}% >= threshold ${(threshold * 100f).toInt()}%)."
        }
        if (overallSpoof >= threshold && vaultReport.maxSimilarity >= 0.65f) {
            warnings += "CRITICAL FAMILY CLONE WARNING: Audio resembles '${vaultReport.matchedContactName ?: "enrolled family member"}' (${(vaultReport.maxSimilarity * 100f).toInt()}%) but exhibits AI vocoder artifacts!"
        }

        val windowSamples = SpeakerVerifier.WINDOW_SAMPLES
        val chunks = mutableListOf<TimelineChunk>()
        var startSample = 0
        var idx = 0
        while (startSample < pcm.size && idx < 16) {
            val endSample = min(pcm.size, startSample + windowSamples)
            val slice = pcm.copyOfRange(startSample, endSample)
            val startMs = (startSample * 1000L) / 16_000L
            val endMs = (endSample * 1000L) / 16_000L
            val sliceVoicedMs = VoicedDurationGate.voicedMs(slice)
            val rangeStr = "${formatMs(startMs)} - ${formatMs(endMs)}"

            val sliceMatch = if (profiles.isNotEmpty() && slice.size >= 1600) {
                speakerVerifier.matchAgainstVault(
                    pcm = slice,
                    profiles = profiles,
                    decoder = { voiceVaultRepository.unsealedEmbedding(it) },
                )
            } else vaultReport

            val tags = mutableListOf("audio", "voice", "speech", "segment")
            if (sliceVoicedMs >= 500L) tags.addAll(listOf("talking", "speaking", "voiced"))
            if (sliceMatch.isMatch) tags.addAll(listOf("matched", "family", sliceMatch.matchedContactName?.lowercase(Locale.US) ?: "contact"))
            if (overallSpoof >= threshold) tags.addAll(listOf("fake", "clone", "synthetic", "spoof"))

            chunks += TimelineChunk(
                index = idx,
                startTimeMs = startMs,
                endTimeMs = endMs,
                formattedTimeRange = rangeStr,
                sceneTags = tags.distinct(),
                visualDescription = "Audio Window $rangeStr: Voiced speech=${sliceVoicedMs}ms.",
                audioSummary = "LCNN-MFM spoof score=${String.format(Locale.US, "%.2f", overallSpoof)} (threshold=${String.format(Locale.US, "%.2f", threshold)}).",
                voiceVaultSummary = if (sliceMatch.isMatch) {
                    "Speaker matches '${sliceMatch.matchedContactName}' (${(sliceMatch.maxSimilarity * 100f).toInt()}% similarity)."
                } else {
                    "Speaker similarity ${(sliceMatch.maxSimilarity * 100f).toInt()}% (below ${((sliceMatch.threshold) * 100f).toInt()}% vault threshold)."
                },
                aiProbability = overallSpoof,
                srmScore = Float.NaN,
                hasPersonOrFace = sliceVoicedMs >= 400L,
                estimatedSubjectTag = sliceMatch.matchedContactName ?: "speaker",
                featureVector = floatArrayOf(overallSpoof, sliceMatch.maxSimilarity, sliceVoicedMs / 3000f),
            )
            startSample += windowSamples
            idx++
        }

        val installedPack = VlmPackManager.getInstalledPackDetails(context)
        return IndexedMediaKnowledgeBase(
            sessionId = "aud_${System.currentTimeMillis()}",
            mediaKind = RagMediaKind.AUDIO,
            mediaTitle = "Audio Clip (${formatMs(durationMs)}, ${voicedMs / 1000.0}s voiced)",
            sourceUri = uri.toString(),
            durationMs = durationMs,
            overallAiProb = overallSpoof,
            headlineVerdict = if (overallSpoof >= threshold) "SYNTHETIC VOICE CLONE" else "AUTHENTIC HUMAN SPEECH",
            activeEngineLabel = if (installedPack != null) {
                "LCNN-MFM + Voice Vault + ${installedPack.modelId}"
            } else {
                "LCNN-MFM v3 + Family Safe-Voice Vault"
            },
            chunks = chunks,
            voiceVaultMatchSummary = vaultSummary,
            scamRiskWarnings = warnings,
        )
    }

    suspend fun indexImageUri(
        uri: Uri,
        onProgress: (String) -> Unit = {},
    ): IndexedMediaKnowledgeBase {
        onProgress("Decoding image & extracting 20-ch SRM + SigLIP-B/16 features...")
        val bitmap = context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it)
        } ?: throw IllegalArgumentException("Unable to decode image from URI: $uri")

        try {
            val srm = srmAnalyzer.analyze(bitmap)
            val kl = klAnalyzer.analyze(bitmap)
            val spectral = fftAnalyzer.analyze(bitmap)
            val probe = runCatching { vfmProbe.analyze(bitmap) }.getOrNull() ?: Float.NaN
            val resNorm = (min(bitmap.width, bitmap.height) / 1024f).coerceIn(0f, 1f)
            val features = FusionFeatureSchema.build(
                srm,
                spectral.score,
                kl,
                spectral.spectralFlatness,
                resNorm,
                1f,
                probe,
                DetectionPipeline.NEUTRAL_C2PA,
                DetectionPipeline.NEUTRAL_LIP_SYNC,
                DetectionPipeline.NEUTRAL_JITTER,
            )
            val aiProb = metaClassifier.predict(features)
            val hm = runCatching { ResidualHeatmapExtractor.computeResidualHeatmap(bitmap, 32) }.getOrNull()
            val hotspot = if (hm != null && hm.isNotEmpty()) {
                ResidualHeatmapExtractor.extractHotspotBoundingBox(hm, 32, 8)
            } else null
            val docReport = runCatching { DocumentForgeryAnalyzer().analyze(bitmap) }.getOrNull()
            val (hasPerson, subjectLabel, tags) = analyzeFrameSubjectHeuristics(bitmap, false, hotspot?.zScore ?: 0f)

            val visualDesc = buildString {
                append("Image (${bitmap.width}x${bitmap.height}) subject: $subjectLabel. ")
                append("Fusion AI probability=${(aiProb * 100f).toInt()}% (SRM=${String.format(Locale.US, "%.2f", srm)}, ")
                if (probe.isFinite()) append("SigLIP-TAP=${String.format(Locale.US, "%.2f", probe)}, ")
                append("FFT=${String.format(Locale.US, "%.2f", spectral.score)}). ")
                if (hotspot != null) {
                    append(
                        "Hotspot ROI box=[${String.format(Locale.US, "%.2f", hotspot.normX0)}, " +
                            "${String.format(Locale.US, "%.2f", hotspot.normY0)} -> " +
                            "${String.format(Locale.US, "%.2f", hotspot.normX1)}, " +
                            "${String.format(Locale.US, "%.2f", hotspot.normY1)}] (Z=${String.format(Locale.US, "%.2f", hotspot.zScore)}). "
                    )
                }
                if (docReport?.isDocumentOrScreenshot == true) {
                    append("Document/Receipt layout tamper score=${String.format(Locale.US, "%.2f", docReport.forgeryScore)}.")
                }
            }

            val chunk = TimelineChunk(
                index = 0,
                startTimeMs = 0L,
                endTimeMs = 0L,
                formattedTimeRange = "Full Image + ROI",
                sceneTags = (tags + listOf("image", "photo", "roi", "forensic")).distinct(),
                visualDescription = visualDesc,
                audioSummary = "Still image (no audio track).",
                voiceVaultSummary = "Not applicable for still image.",
                aiProbability = aiProb,
                srmScore = srm,
                hotspotBoxNorm = hotspot?.let { floatArrayOf(it.normX0, it.normY0, it.normX1, it.normY1) },
                hotspotZScore = hotspot?.zScore ?: 0f,
                hasPersonOrFace = hasPerson,
                estimatedSubjectTag = subjectLabel,
                featureVector = floatArrayOf(aiProb, srm, if (probe.isFinite()) probe else 0f, hotspot?.zScore ?: 0f),
            )

            val installedPack = VlmPackManager.getInstalledPackDetails(context)
            return IndexedMediaKnowledgeBase(
                sessionId = "img_${System.currentTimeMillis()}",
                mediaKind = RagMediaKind.IMAGE,
                mediaTitle = "Image (${bitmap.width}×${bitmap.height})",
                sourceUri = uri.toString(),
                durationMs = 0L,
                overallAiProb = aiProb,
                headlineVerdict = if (aiProb >= 0.578f) "SYNTHETIC / AI-GENERATED" else "LIKELY AUTHENTIC",
                activeEngineLabel = if (installedPack != null) {
                    "Qwen3.5-0.8B (${installedPack.modelId}) + SigLIP-B/16 + SRM"
                } else {
                    "Specialist RAG (SigLIP-B/16 + 20-ch SRM + ROI Heatmap)"
                },
                chunks = listOf(chunk),
                voiceVaultMatchSummary = "Still image scan",
                scamRiskWarnings = if (aiProb >= 0.578f) {
                    listOf("Image exceeds calibrated synthetic threshold (${(aiProb * 100f).toInt()}% >= 57.8%).")
                } else emptyList(),
            )
        } finally {
            if (!bitmap.isRecycled) bitmap.recycle()
        }
    }

    fun retrieveTopChunks(
        kb: IndexedMediaKnowledgeBase,
        query: String,
        topK: Int = 3,
    ): List<TimelineChunk> {
        if (kb.chunks.isEmpty()) return emptyList()
        val qTokens = query.lowercase(Locale.US)
            .split(Regex("[^a-z0-9]+"))
            .filter { it.isNotBlank() }
            .toSet()

        val asksAboutPersonOrBoy = qTokens.any { it in BOY_CHILD_SYNONYMS }
        val asksAboutFakeOrAnomaly = qTokens.any { it in FAKE_FORENSIC_SYNONYMS }
        val asksAboutAudioOrVoice = qTokens.any { it in AUDIO_VOICE_SYNONYMS }

        val scored = kb.chunks.map { chunk ->
            var score = 0.0f
            val docText = "${chunk.sceneTags.joinToString(" ")} ${chunk.visualDescription} ${chunk.audioSummary} ${chunk.voiceVaultSummary}"
                .lowercase(Locale.US)

            for (tok in qTokens) {
                if (docText.contains(tok)) score += 2.2f
            }
            if (asksAboutPersonOrBoy) {
                if (chunk.hasPersonOrFace) score += 3.5f
                if (chunk.sceneTags.contains("entry") || chunk.sceneTags.contains("entered")) score += 4.0f
            }
            if (asksAboutFakeOrAnomaly) {
                score += chunk.aiProbability * 3.0f + (chunk.hotspotZScore.coerceAtMost(3f) * 0.8f)
            }
            if (asksAboutAudioOrVoice) {
                score += 2.5f
            }
            if (chunk.isSceneCut) {
                score += 0.4f
            }
            Pair(chunk, score)
        }

        return scored
            .sortedWith(compareByDescending<Pair<TimelineChunk, Float>> { it.second }.thenBy { it.first.index })
            .take(topK)
            .map { it.first }
    }

    suspend fun askQuestion(
        kb: IndexedMediaKnowledgeBase,
        question: String,
    ): RagAnswer {
        val t0 = System.currentTimeMillis()
        val retrieved = retrieveTopChunks(kb, question, topK = 3)
        val qLower = question.lowercase(Locale.US)

        val installedPack = VlmPackManager.getInstalledPackDetails(context)
        val packDir = File(context.filesDir, "vlm_packs")
        val langFile = if (installedPack != null) File(packDir, installedPack.languageFileName) else null
        val mmprojFile = if (installedPack != null) File(packDir, installedPack.visionFileName) else null

        var vlmOutput: String? = null
        var usedVlm = false

        if (installedPack != null && langFile != null && langFile.isFile &&
            (LlamaCppVlmPrefillBackend.isNativeLibraryLoaded() || LlamaCppVlmPrefillBackend.testGenerationDelegate != null)
        ) {
            val ragPrompt = buildRagPrompt(kb, retrieved, question)
            vlmOutput = runCatching {
                inferenceEngine.acquireExclusiveVlmSession().use {
                    LlamaCppVlmPrefillBackend(
                        modelPath = langFile.absolutePath,
                        mmprojPath = mmprojFile?.takeIf { it.isFile }?.absolutePath,
                        hiddenDim = if (installedPack.modelId.contains("2B", ignoreCase = true)) 2048 else 1024,
                    ).use { backend ->
                        backend.generateMultimodalResponse(
                            prompt = ragPrompt,
                            bitmap = null,
                            maxNewTokens = 160,
                        )
                    }
                }
            }.getOrNull()?.takeIf { it.isNotBlank() }
            if (vlmOutput != null) {
                usedVlm = true
            }
        }

        val finalAnswer = vlmOutput ?: synthesizeDeterministicGroundedAnswer(kb, retrieved, qLower)
        val cited = retrieved.map { it.formattedTimeRange }.distinct()
        val elapsed = max(1L, System.currentTimeMillis() - t0)

        return RagAnswer(
            answerText = finalAnswer,
            citedTimestamps = cited,
            retrievedChunks = retrieved,
            usedOnDeviceVlm = usedVlm,
            engineBadge = if (usedVlm) {
                "Qwen3.5-0.8B Local VLM (${elapsed}ms)"
            } else {
                "Grounded Timeline RAG (${elapsed}ms)"
            },
            latencyMs = elapsed,
        )
    }

    private fun buildRagPrompt(
        kb: IndexedMediaKnowledgeBase,
        retrieved: List<TimelineChunk>,
        question: String,
    ): String {
        val contextBlock = retrieved.joinToString("\n") { c ->
            "- [${c.formattedTimeRange}] ${c.visualDescription} | Audio: ${c.audioSummary} | Vault: ${c.voiceVaultSummary}"
        }
        return buildString {
            append("<|im_start|>system\n")
            append("You are ONN-DETECT's 100% offline forensic media assistant running on Android. ")
            append("Answer concisely using ONLY the indexed timeline evidence below. Always cite timestamps in [mm:ss] format.\n")
            append("Media: ${kb.mediaTitle} | Verdict: ${kb.headlineVerdict} | Overall AI Risk: ${(kb.overallAiProb * 100f).toInt()}%\n")
            append("Retrieved Timeline Evidence:\n")
            append(contextBlock)
            append("\n<|im_end|>\n<|im_start|>user\n")
            append(question.trim())
            append("<|im_end|>\n<|im_start|>assistant\n")
        }
    }

    private fun synthesizeDeterministicGroundedAnswer(
        kb: IndexedMediaKnowledgeBase,
        retrieved: List<TimelineChunk>,
        qLower: String,
    ): String {
        val asksWhenPersonOrBoy = BOY_CHILD_SYNONYMS.any { qLower.contains(it) }
        val asksVoiceOrAudio = AUDIO_VOICE_SYNONYMS.any { qLower.contains(it) }
        val asksFakeOrWhy = FAKE_FORENSIC_SYNONYMS.any { qLower.contains(it) }

        if (kb.mediaKind == RagMediaKind.VIDEO && asksWhenPersonOrBoy) {
            val personChunks = kb.chunks.filter { it.hasPersonOrFace }
            if (personChunks.isNotEmpty()) {
                val firstEntry = personChunks.first()
                val lastSeen = personChunks.last()
                val peakPerson = personChunks.maxByOrNull { it.aiProbability } ?: firstEntry
                return buildString {
                    append("• **Scene Entry Detected**: A **${firstEntry.estimatedSubjectTag}** first appears at **[${formatMs(firstEntry.startTimeMs)}]** ")
                    append("(window `${firstEntry.formattedTimeRange}`) and remains visible through **[${formatMs(lastSeen.endTimeMs)}]**.\n")
                    append("• **Forensic Authenticity at Appearance**: During `[${peakPerson.formattedTimeRange}]`, the visual AI synthesis score is ")
                    append("**${(peakPerson.aiProbability * 100f).toInt()}%** (20-ch SRM=${String.format(Locale.US, "%.2f", peakPerson.srmScore)}, ")
                    append("Hotspot Z=${String.format(Locale.US, "%.2f", peakPerson.hotspotZScore)}).\n")
                    append("• **Audio & Voice Vault**: ${firstEntry.voiceVaultSummary}")
                }
            } else {
                return "No prominent human face or boy/person track was detected across the ${kb.chunks.size} sampled keyframes (`00:00 - ${formatMs(kb.durationMs)}`). " +
                    "The scene primarily contains background/environmental frames with an overall AI probability of **${(kb.overallAiProb * 100f).toInt()}%**."
            }
        }

        if (asksVoiceOrAudio) {
            return buildString {
                append("• **Voice & Audio Verdict**: **${kb.headlineVerdict}** (AI spoof score **${(kb.overallAiProb * 100f).toInt()}%**).\n")
                append("• **Family Safe-Voice Vault**: ${kb.voiceVaultMatchSummary}\n")
                if (retrieved.isNotEmpty()) {
                    val top = retrieved.first()
                    append("• **Key Segment (`[${top.formattedTimeRange}]`)**: ${top.audioSummary}")
                }
                if (kb.scamRiskWarnings.isNotEmpty()) {
                    append("\n• **Security Alert**: ${kb.scamRiskWarnings.joinToString(" ")}")
                }
            }
        }

        if (asksFakeOrWhy || retrieved.isNotEmpty()) {
            val peak = kb.chunks.maxByOrNull { it.aiProbability } ?: retrieved.firstOrNull()
            return buildString {
                append("• **Headline Assessment**: **${kb.headlineVerdict}** (Overall AI probability **${(kb.overallAiProb * 100f).toInt()}%**).\n")
                if (peak != null) {
                    append("• **Primary Evidence (`[${peak.formattedTimeRange}]`)**: ${peak.visualDescription}\n")
                }
                if (kb.voiceVaultMatchSummary.isNotBlank()) {
                    append("• **Audio / Voice Vault Check**: ${kb.voiceVaultMatchSummary}")
                }
            }
        }

        return "Indexed **${kb.mediaTitle}** (${kb.chunks.size} timeline segments). Overall verdict: **${kb.headlineVerdict}** " +
            "(${(kb.overallAiProb * 100f).toInt()}% AI risk). Ask about specific timestamps, person appearances, or voice-clone checks."
    }
}
