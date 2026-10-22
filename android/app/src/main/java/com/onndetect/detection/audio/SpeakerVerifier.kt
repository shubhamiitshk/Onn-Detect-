package com.onndetect.detection.audio

import com.onndetect.data.local.VoiceProfileEntity
import com.onndetect.detection.visual.FFTUtil
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

enum class VoiceMatchConfidence {
    HIGH_MATCH,
    MEDIUM_MATCH,
    NO_MATCH,
    NO_PROFILES_ENROLLED,
}

data class VoiceMatchReport(
    val hasEnrolledProfiles: Boolean,
    val isMatch: Boolean,
    val matchedContactName: String? = null,
    val matchedRelationship: String? = null,
    val matchedProfileId: String? = null,
    val maxSimilarity: Float = 0f,
    val threshold: Float = SpeakerVerifier.DEFAULT_MATCH_THRESHOLD,
    val confidence: VoiceMatchConfidence = VoiceMatchConfidence.NO_PROFILES_ENROLLED,
    val candidateScores: Map<String, Float> = emptyMap(),
    val flags: List<String> = emptyList(),
)

/**
 * On-device speaker identity verification engine for the Family Safe-Voice Vault.
 * Extracts normalized 160-dimensional acoustic cepstral voiceprint embeddings and
 * performs cosine-similarity matching against locally enrolled trusted family contacts.
 *
 * Runs 100% offline with zero network calls and zero external server dependencies.
 */
@Singleton
class SpeakerVerifier @Inject constructor() {

    companion object {
        const val DEFAULT_MATCH_THRESHOLD = 0.80f
        const val HIGH_CONFIDENCE_THRESHOLD = 0.88f
        const val EMBEDDING_DIM = 160
        private const val MIN_SAMPLES_FOR_VOICEPRINT = 1600 // At least 100ms at 16kHz
    }

    /**
     * Extracts a normalized 160-dimensional acoustic speaker voiceprint embedding vector
     * from 16 kHz mono PCM samples:
     * - Dims 0..79: Harmonic pitch autocorrelation profile (lag 30 to 188)
     * - Dims 80..159: Standardized vocal-tract filterbank resonance envelope (80 bands)
     */
    fun extractVoiceprint(pcm: FloatArray): FloatArray {
        if (pcm.size < MIN_SAMPLES_FOR_VOICEPRINT) {
            return FloatArray(EMBEDDING_DIM)
        }

        return try {
            val autoCorr = FloatArray(80)
            val analysisLen = min(pcm.size, 8000)
            var sumPcm = 0.0
            for (i in 0 until analysisLen) sumPcm += pcm[i]
            val meanPcm = (sumPcm / analysisLen).toFloat()

            var varPcm = 0.0
            for (i in 0 until analysisLen) {
                val d = pcm[i] - meanPcm
                varPcm += d * d
            }
            val varDenom = max(varPcm, 1e-9)

            // Lags 30 to 188 (step 2) -> 80 lags covering 85 Hz to 533 Hz pitch range
            for (idx in 0 until 80) {
                val lag = 30 + idx * 2
                if (lag < analysisLen) {
                    var c = 0.0
                    for (i in 0 until analysisLen - lag) {
                        c += (pcm[i] - meanPcm) * (pcm[i + lag] - meanPcm)
                    }
                    autoCorr[idx] = (c / varDenom).toFloat()
                }
            }

            // L2 normalize autoCorr
            var acNormSq = 0.0
            for (v in autoCorr) acNormSq += v * v
            val acNorm = sqrt(acNormSq).toFloat()
            if (acNorm > 1e-6f) {
                for (i in autoCorr.indices) autoCorr[i] /= acNorm
            }

            // Spectral Filterbank Envelope (80 bands)
            val nFft = 512
            val hop = 160
            val nFrames = (pcm.size - nFft) / hop + 1
            val specAccum = FloatArray(nFft / 2 + 1)
            val re = DoubleArray(nFft)
            val im = DoubleArray(nFft)

            if (nFrames > 0) {
                for (t in 0 until nFrames) {
                    val offset = t * hop
                    for (i in 0 until nFft) {
                        val w = 0.5 * (1.0 - cos(2.0 * Math.PI * i / nFft))
                        re[i] = pcm[offset + i] * w
                        im[i] = 0.0
                    }
                    FFTUtil.transform(re, im)
                    for (f in specAccum.indices) {
                        val p = (re[f] * re[f] + im[f] * im[f]).toFloat()
                        specAccum[f] += p
                    }
                }
                for (f in specAccum.indices) specAccum[f] /= nFrames
            }

            // Pool 257 bins into 80 log-energy bands
            val logBands = FloatArray(80)
            val bandSize = specAccum.size / 80
            for (b in 0 until 80) {
                var bandSum = 0.0
                val start = b * bandSize
                val end = if (b == 79) specAccum.size else (b + 1) * bandSize
                for (k in start until end) bandSum += specAccum[k]
                logBands[b] = kotlin.math.log10(max(bandSum.toFloat(), 1e-9f))
            }

            // Standardize logBands: (x - mean) / std
            var lbSum = 0.0
            for (v in logBands) lbSum += v
            val lbMean = (lbSum / 80).toFloat()
            var lbVar = 0.0
            for (v in logBands) {
                val d = v - lbMean
                lbVar += d * d
            }
            val lbStd = sqrt(lbVar / 80).toFloat()
            if (lbStd > 1e-6f) {
                for (i in logBands.indices) logBands[i] = (logBands[i] - lbMean) / lbStd
            }

            // L2 normalize logBands
            var lbNormSq = 0.0
            for (v in logBands) lbNormSq += v * v
            val lbNorm = sqrt(lbNormSq).toFloat()
            if (lbNorm > 1e-6f) {
                for (i in logBands.indices) logBands[i] /= lbNorm
            }

            // Concatenate: 80 pitch lags + 80 spectral shape bands = 160 dims
            val embedding = FloatArray(EMBEDDING_DIM)
            val scale = (1.0 / sqrt(2.0)).toFloat()
            for (i in 0 until 80) {
                embedding[i] = autoCorr[i] * scale
                embedding[80 + i] = logBands[i] * scale
            }
            embedding
        } catch (_: Exception) {
            FloatArray(EMBEDDING_DIM)
        }
    }

    /**
     * Computes the cosine similarity between two voiceprint embedding vectors.
     * Returns a float in [-1.0, 1.0].
     */
    fun computeCosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.isEmpty() || b.isEmpty() || a.size != b.size) return 0f
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = sqrt(normA) * sqrt(normB)
        if (denom < 1e-9) return 0f
        return (dot / denom).toFloat().coerceIn(-1f, 1f)
    }

    /**
     * Matches an incoming audio sample against all enrolled profiles in the vault.
     */
    fun matchAgainstVault(
        pcm: FloatArray,
        profiles: List<VoiceProfileEntity>,
        threshold: Float = DEFAULT_MATCH_THRESHOLD,
    ): VoiceMatchReport {
        if (profiles.isEmpty()) {
            return VoiceMatchReport(
                hasEnrolledProfiles = false,
                isMatch = false,
                threshold = threshold,
                confidence = VoiceMatchConfidence.NO_PROFILES_ENROLLED,
                flags = listOf("No family voiceprints currently enrolled in Safe-Voice Vault"),
            )
        }

        val incomingEmbedding = extractVoiceprint(pcm)
        if (incomingEmbedding.all { it == 0f }) {
            return VoiceMatchReport(
                hasEnrolledProfiles = true,
                isMatch = false,
                threshold = threshold,
                confidence = VoiceMatchConfidence.NO_MATCH,
                flags = listOf("Insufficient audio duration or speech energy to extract voiceprint"),
            )
        }

        val scores = mutableMapOf<String, Float>()
        var bestProfile: VoiceProfileEntity? = null
        var maxSim = -1.0f

        for (profile in profiles) {
            val enrolledVec = profile.toEmbeddingArray()
            if (enrolledVec.isEmpty()) continue
            val sim = computeCosineSimilarity(incomingEmbedding, enrolledVec)
            scores[profile.id] = sim
            if (sim > maxSim) {
                maxSim = sim
                bestProfile = profile
            }
        }

        val flags = mutableListOf<String>()
        val isMatch = maxSim >= threshold && bestProfile != null

        val confidence = when {
            !isMatch -> {
                flags.add("Voice does not match any enrolled family member (Max similarity %.1f%% < threshold %.0f%%)".format(
                    java.util.Locale.US,
                    max(0f, maxSim) * 100f,
                    threshold * 100f
                ))
                VoiceMatchConfidence.NO_MATCH
            }
            maxSim >= HIGH_CONFIDENCE_THRESHOLD -> {
                flags.add("High-confidence acoustic match with enrolled profile '${bestProfile?.contactName}' (${bestProfile?.relationship})")
                VoiceMatchConfidence.HIGH_MATCH
            }
            else -> {
                flags.add("Moderate acoustic match with enrolled profile '${bestProfile?.contactName}' (${bestProfile?.relationship})")
                VoiceMatchConfidence.MEDIUM_MATCH
            }
        }

        return VoiceMatchReport(
            hasEnrolledProfiles = true,
            isMatch = isMatch,
            matchedContactName = if (isMatch) bestProfile?.contactName else null,
            matchedRelationship = if (isMatch) bestProfile?.relationship else null,
            matchedProfileId = if (isMatch) bestProfile?.id else null,
            maxSimilarity = max(0f, maxSim),
            threshold = threshold,
            confidence = confidence,
            candidateScores = scores,
            flags = flags,
        )
    }
}
