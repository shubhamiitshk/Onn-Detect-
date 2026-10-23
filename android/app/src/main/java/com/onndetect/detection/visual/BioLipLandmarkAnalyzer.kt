package com.onndetect.detection.visual

import android.graphics.Bitmap
import com.onndetect.domain.pipeline.DetectionPipeline
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.exp
import kotlin.math.sqrt

data class BioLipResult(
    val hasFace: Boolean,
    /** Slot-9 value in [0.0, 1.0]; neutral is 0.10f. */
    val landmarkJitter: Float,
    val speechApertureDelta: Float,
    val framesTracked: Int,
    val notes: String? = null,
)

/**
 * On-device facial perioral landmark jitter analyzer based on BioLip
 * (arXiv:2604.16808). Evaluates the biomechanical constraints of human
 * orofacial articulation across video frames.
 *
 * Natural human speech exhibits smooth muscle damping in lip contours.
 * Generative talking-head models (Wav2Lip, SadTalker, LivePortrait) exhibit
 * high-frequency second-order temporal micro-jitter due to unconstrained
 * frame-by-frame synthesis.
 *
 * F5 rewrite: the frontend is REAL landmarks now.
 *  - Preferred: MediaPipe FaceLandmarker, 478-point mesh, 64-point perioral
 *    set, pose-normalized by the inter-ocular axis (33/263) around the nose
 *    tip (1) — head translation, scale and roll are removed before any
 *    derivative is taken.
 *  - The score is a fixed 0.6/0.4 blend of second-order acceleration RMS
 *    and MAR (mouth aspect ratio) flicker, per the BioLip formulation.
 *  - The legacy red-chrominance 16-point proxy survives ONLY as a flagged
 *    degradation path when the mesh asset is unavailable on-device.
 *
 * Injects real temporal jitter into slot 9 of FusionFeatureSchema.
 */
@Singleton
class BioLipLandmarkAnalyzer @Inject constructor() {

    private var provider: (() -> LandmarkFrontend)? = null

    /** Secondary constructor used by DI (PipelineModule) for the mesh path. */
    constructor(frontendProvider: (() -> LandmarkFrontend)?) : this() {
        provider = frontendProvider
    }

    companion object {
        /** The canonical perioral set size (MediaPipe path). */
        const val NUM_PERIORAL_LANDMARKS = 64

        // Sigmoid maps for the two components.
        const val JITTER_LOGIT_SCALE = 35.0f
        const val JITTER_RMS_OFFSET = 0.05f
        const val MAR_LOGIT_SCALE = 12.0f
        const val MAR_FLICKER_OFFSET = 0.35f

        /** Fixed BioLip blend: acceleration jitter 60%, MAR flicker 40%. */
        const val ACCELERATION_WEIGHT = 0.6f
        const val MAR_FLICKER_WEIGHT = 0.4f
    }

    private val proxyFrontend = ChrominanceProxyFrontend()
    private var cachedFrontend: LandmarkFrontend? = null
    private var frontendResolved = false

    /**
     * First usable frontend wins; a failed MediaPipe build (missing asset,
     * no EGL context on this thread) degrades to the flagged proxy for the
     * whole session, and says so in the notes.
     */
    private fun activeFrontend(): LandmarkFrontend {
        if (frontendResolved) return cachedFrontend ?: proxyFrontend
        val p = provider
        if (p == null) {
            frontendResolved = true
            return proxyFrontend
        }
        val built = runCatching { p() }.getOrNull()
        if (built != null) {
            cachedFrontend = built
            frontendResolved = true
            return built
        }
        frontendResolved = true
        return proxyFrontend
    }

    /**
     * Analyzes sampled video frames to detect orofacial contours and measure
     * temporal jitter. Every frame goes through the ACTIVE frontend; frames
     * where no face was found are dropped (not zero-filled: a missing face is
     * not a still face).
     */
    fun analyzeFrames(frames: List<Bitmap>): BioLipResult {
        if (frames.size < 3) {
            return BioLipResult(
                hasFace = false,
                landmarkJitter = DetectionPipeline.NEUTRAL_JITTER,
                speechApertureDelta = 0f,
                framesTracked = frames.size,
                notes = "Insufficient frames for temporal second-order acceleration analysis",
            )
        }
        val frontend = activeFrontend()
        val trajectories = mutableListOf<Array<FloatArray>>() // [T][N][2]
        val marSeries = mutableListOf<Float>()

        for (frame in frames) {
            val perioral = frontend.detectPerioral(frame) ?: continue
            if (perioral.size < 16) continue
            trajectories += perioral
            marSeries += PerioralTopology.takeIf { perioral.size == 64 }
                ?.mouthAspectRatio(perioral)
                ?: proxyAperture(perioral) // proxy path: no inner-lip ring
        }

        if (trajectories.size < 3) {
            return BioLipResult(
                hasFace = false,
                landmarkJitter = DetectionPipeline.NEUTRAL_JITTER,
                speechApertureDelta = 0f,
                framesTracked = trajectories.size,
                notes = "No sustained orofacial lip landmarks detected across frames",
            )
        }

        val trajectoryArray = trajectories.toTypedArray()
        val accelScore = computeJitterFromTrajectories(trajectoryArray)
        val marScore = if (marSeries.size == trajectories.size && trajectories[0].size == 64) {
            computeMarFlicker(marSeries.toFloatArray())
        } else 0.10f // proxy path: no MAR ring; acceleration carries the score
        val jitter = computeBioLipScore(accelScore, marScore)

        val meanMar = if (marSeries.isNotEmpty()) marSeries.average().toFloat() else 0f
        val proxyNote = if (frontend.id.startsWith("chrominance")) {
            "DEGRADED: landmark frontend is the chrominance proxy (mesh asset unavailable); results are proxy-grade"
        } else {
            "Landmark frontend: ${frontend.id}"
        }

        return BioLipResult(
            hasFace = true,
            landmarkJitter = jitter,
            speechApertureDelta = meanMar,
            framesTracked = trajectories.size,
            notes = if (jitter > 0.40f) "$proxyNote; elevated orofacial micro-jitter indicates generative lip synthesis"
            else "$proxyNote; natural biological lip dynamics",
        )
    }

    /**
     * Pure unit-testable calculation of 2nd-order temporal acceleration
     * variance across landmark trajectory arrays shaped [T][N][2]
     * (T frames, N points, (x,y)) — unchanged contract, real or proxy
     * trajectories alike.
     */
    fun computeJitterFromTrajectories(trajectories: Array<Array<FloatArray>>): Float {
        val t = trajectories.size
        if (t < 3) return DetectionPipeline.NEUTRAL_JITTER

        val n = trajectories[0].size
        var sumSquaredAcc = 0.0
        var count = 0

        for (frameIdx in 1 until t - 1) {
            val prev = trajectories[frameIdx - 1]
            val curr = trajectories[frameIdx]
            val next = trajectories[frameIdx + 1]
            for (p in 0 until n) {
                // Acceleration vector: A_t = L_{t+1} - 2*L_t + L_{t-1}
                val ax = next[p][0] - 2f * curr[p][0] + prev[p][0]
                val ay = next[p][1] - 2f * curr[p][1] + prev[p][1]
                sumSquaredAcc += (ax * ax + ay * ay).toDouble()
                count++
            }
        }
        if (count == 0) return DetectionPipeline.NEUTRAL_JITTER

        val rmsAcc = sqrt(sumSquaredAcc / count).toFloat()
        val logit = JITTER_LOGIT_SCALE * (rmsAcc - JITTER_RMS_OFFSET)
        return (1.0f / (1.0f + exp(-logit))).coerceIn(0.01f, 0.99f)
    }

    /**
     * The fixed BioLip blend (pure, testable): 0.6 * acceleration jitter
     * + 0.4 * MAR flicker, clamped into the slot-9 value range.
     */
    fun computeBioLipScore(accelerationScore: Float, marFlickerScore: Float): Float =
        (ACCELERATION_WEIGHT * accelerationScore + MAR_FLICKER_WEIGHT * marFlickerScore)
            .coerceIn(0.01f, 0.99f)

    /**
     * MAR flicker (F5): the RMS of the first difference of the mouth aspect
     * ratio series, normalized by the mean aperture, mapped through a
     * sigmoid. Natural articulation drifts smoothly (low flicker); Wav2Lip
     * class synths snap the aperture frame-to-frame (high flicker).
     */
    fun computeMarFlicker(marSeries: FloatArray): Float {
        if (marSeries.size < 3) return DetectionPipeline.NEUTRAL_JITTER
        val mean = marSeries.average().toFloat()
        if (mean < 1e-4f) return DetectionPipeline.NEUTRAL_JITTER
        var sumSq = 0.0
        val count = marSeries.size - 1
        for (i in 0 until count) {
            val d = marSeries[i + 1] - marSeries[i]
            sumSq += (d * d).toDouble()
        }
        val flickerRms = (sqrt(sumSq / count) / mean).toFloat()
        val logit = MAR_LOGIT_SCALE * (flickerRms - MAR_FLICKER_OFFSET)
        return (1.0f / (1.0f + exp(-logit))).coerceIn(0.01f, 0.99f)
    }

    /** Proxy-path aperture estimate (legacy semantic, kept for continuity). */
    private fun proxyAperture(perioral: Array<FloatArray>): Float {
        if (perioral.size < 16) return 0f
        return kotlin.math.abs(perioral[4][1] - perioral[12][1])
    }
}
