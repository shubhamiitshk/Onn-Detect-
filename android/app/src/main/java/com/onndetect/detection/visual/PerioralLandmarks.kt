package com.onndetect.detection.visual

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import kotlin.math.hypot

/**
 * Perioral landmark frontends — the F5 fix.
 *
 * The pre-audit frontend located the mouth with a red-chrominance heuristic
 * and then drew a SYNTHETIC 16-point ellipse around the brightest red pixel:
 * the "landmarks" were fiction, and the jitter measured the fiction. This
 * file replaces it with two real frontends:
 *
 *  - [MediaPipeLandmarkFrontend]: the 478-point face mesh (face_landmarker.task,
 *    float16, ~3.8 MB), pose-normalized to the inter-ocular axis (outer eye
 *    corners 33/263) around the nose tip (index 1) and scaled to unit
 *    inter-ocular distance — so head translation, scale and roll are removed
 *    before any temporal derivative is taken. GPU delegate first, with an
 *    explicit CPU fallback for background coroutine threads that carry no
 *    EGL context (~10 ms/frame on CPU).
 *
 *  - [ChrominanceProxyFrontend]: the legacy heuristic, kept ONLY as the
 *    degradation path when the mesh asset is unavailable (the analyzer
 *    marks its results accordingly). It is a proxy, and it says so.
 */
interface LandmarkFrontend {
    /** Human-readable frontend id, surfaced in the analysis notes. */
    val id: String

    /**
     * Perioral landmarks for one frame: [N][2] in pose-normalized units
     * (roughly [-1, 1]), or null when no face was found.
     */
    fun detectPerioral(bitmap: Bitmap): Array<FloatArray>?

    fun close() {}
}

/** The canonical 64-point perioral set on the MediaPipe topology (F5). */
object PerioralTopology {

    /** 20 outer-lip ring points. */
    val OUTER_LIP = intArrayOf(
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    )

    /** 20 inner-lip ring points. */
    val INNER_LIP = intArrayOf(
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    )

    /** 12 lower-jaw arc anchors (menton to gonion span). */
    val JAW_ANCHORS = intArrayOf(152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127)

    /** 12 nasolabial / cheek anchors that ride articulation. */
    val CHEEK_ANCHORS = intArrayOf(205, 425, 98, 327, 206, 426, 214, 434, 216, 436, 212, 432)

    /** The full 64-point set, in stable order. */
    val PERIORAL_64: IntArray = OUTER_LIP + INNER_LIP + JAW_ANCHORS + CHEEK_ANCHORS

    const val POINTS = 64

    // MediaPipe canonical indices used by the normalizer / MAR.
    const val RIGHT_EYE_OUTER = 33
    const val LEFT_EYE_OUTER = 263
    const val NOSE_TIP = 1
    const val UPPER_INNER_LIP = 13
    const val LOWER_INNER_LIP = 14
    const val MOUTH_CORNER_RIGHT = 78
    const val MOUTH_CORNER_LEFT = 308

    /**
     * Pose normalization: rotates every point by the negative inter-ocular
     * roll around the nose tip and divides by inter-ocular distance, so the
     * output is invariant to head translation, scale and in-plane roll.
     */
    fun normalizePose(mesh: Array<FloatArray>): Array<FloatArray>? {
        if (mesh.size <= CHEEK_ANCHORS.max()) return null
        val right = mesh[RIGHT_EYE_OUTER]
        val left = mesh[LEFT_EYE_OUTER]
        val nose = mesh[NOSE_TIP]
        val dx = left[0] - right[0]
        val dy = left[1] - right[1]
        val iod = hypot(dx, dy)
        if (iod < 1e-4f) return null // degenerate: eye corners collapsed
        val cos = dx / iod
        val sin = dy / iod
        return Array(POINTS) { i ->
            val p = mesh[PERIORAL_64[i]]
            val vx = p[0] - nose[0]
            val vy = p[1] - nose[1]
            floatArrayOf(
                (vx * cos + vy * sin) / iod,
                (-vx * sin + vy * cos) / iod,
            )
        }
    }

    /** Mouth aspect ratio in pose-normalized units: |y13-y14| / |x78-x308|. */
    fun mouthAspectRatio(perioral: Array<FloatArray>): Float {
        val upper = pointOf(perioral, UPPER_INNER_LIP) ?: return 0f
        val lower = pointOf(perioral, LOWER_INNER_LIP) ?: return 0f
        val rightCorner = pointOf(perioral, MOUTH_CORNER_RIGHT) ?: return 0f
        val leftCorner = pointOf(perioral, MOUTH_CORNER_LEFT) ?: return 0f
        val width = hypot(leftCorner[0] - rightCorner[0], leftCorner[1] - rightCorner[1])
        if (width < 1e-4f) return 0f
        return kotlin.math.abs(lower[1] - upper[1]) / width
    }

    private fun pointOf(perioral: Array<FloatArray>, meshIndex: Int): FloatArray? {
        val slot = PERIORAL_64.indexOf(meshIndex)
        if (slot < 0) return null
        return perioral.getOrNull(slot)
    }
}

/**
 * Real-landmark frontend over the MediaPipe face mesh.
 * GPU first; any failure (typically the missing EGL context on a background
 * coroutine thread) falls back to CPU, which the guardrail mandates.
 */
class MediaPipeLandmarkFrontend(private val context: Context) : LandmarkFrontend {

    override val id: String = "mediapipe-facelandmarker-478"

    private var faceLandmarker: FaceLandmarker? = null
    private var failed = false
    private var usedCpuFallback = false

    private fun assetAvailable(): Boolean = runCatching {
        context.assets.open(MODEL_ASSET).use { it.read() }
        true
    }.getOrDefault(false)

    private fun build(delegate: Delegate): FaceLandmarker? {
        val base = BaseOptions.builder()
            .setModelAssetPath(MODEL_ASSET)
            .setDelegate(delegate)
            .build()
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.IMAGE)
            .setNumFaces(1)
            .build()
        return FaceLandmarker.createFromOptions(context, options)
    }

    private fun landmarker(): FaceLandmarker? {
        if (failed) return null
        faceLandmarker?.let { return it }
        if (!assetAvailable()) { failed = true; return null }
        faceLandmarker = runCatching { build(Delegate.GPU) }.getOrNull()
            ?: runCatching { build(Delegate.CPU).also { usedCpuFallback = true } }.getOrNull()
        if (faceLandmarker == null) failed = true
        return faceLandmarker
    }

    override fun detectPerioral(bitmap: Bitmap): Array<FloatArray>? {
        if (bitmap.isRecycled || bitmap.width < 32 || bitmap.height < 32) return null
        var marker = landmarker() ?: return null
        return runCatching {
            val image = BitmapImageBuilder(bitmap).build()
            val result = marker.detect(image)
            val landmarks = result.faceLandmarks()?.firstOrNull() ?: return null
            if (landmarks.size < 478) return null
            val mesh = Array(landmarks.size) { i ->
                floatArrayOf(landmarks[i].x(), landmarks[i].y())
            }
            PerioralTopology.normalizePose(mesh)
        }.recoverCatching {
            // GPU delegates usually build fine and then FAIL at the first
            // inference (no EGL context on this background thread). That
            // failure mode rebuilds the whole graph on CPU exactly once.
            if (!usedCpuFallback) {
                usedCpuFallback = true
                runCatching { faceLandmarker?.close() }
                faceLandmarker = build(Delegate.CPU)
                marker = faceLandmarker ?: return null
                val image = BitmapImageBuilder(bitmap).build()
                val result = marker.detect(image)
                val landmarks = result.faceLandmarks()?.firstOrNull() ?: return null
                val mesh = Array(landmarks.size) { i -> floatArrayOf(landmarks[i].x(), landmarks[i].y()) }
                PerioralTopology.normalizePose(mesh)
            } else {
                null
            }
        }.getOrNull()
    }

    override fun close() {
        runCatching { faceLandmarker?.close() }
        faceLandmarker = null
    }

    /** True when the GPU path was unavailable and CPU (~10 ms/frame) is in use. */
    fun usedCpuFallback(): Boolean = usedCpuFallback

    companion object {
        const val MODEL_ASSET = "models/face_landmarker.task"
    }
}

/**
 * Legacy heuristic frontend (pre-F5 code, unchanged in spirit): a red
 * chrominance mouth-centre search followed by a synthetic 16-point ellipse.
 * Used ONLY when the mesh frontend is unavailable; results are flagged as
 * proxy-grade in the analyzer notes.
 */
class ChrominanceProxyFrontend : LandmarkFrontend {

    override val id: String = "chrominance-proxy-16"

    override fun detectPerioral(bitmap: Bitmap): Array<FloatArray>? {
        if (bitmap.isRecycled || bitmap.width < 32 || bitmap.height < 32) return null
        val w = bitmap.width
        val h = bitmap.height
        val startY = h / 3
        val endY = (h * 0.90f).toInt()
        val startX = w / 4
        val endX = (w * 0.75f).toInt()
        var bestRatio = 0.0
        var cx = -1
        var cy = -1
        val step = maxOf(2, (endX - startX) / 40)
        var y = startY
        while (y < endY) {
            var x = startX
            while (x < endX) {
                val pixel = bitmap.getPixel(x, y)
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                val ratio = r.toFloat() / (g.toFloat() + b.toFloat() + 1e-4f)
                if (ratio > 1.25f && ratio > bestRatio) { bestRatio = ratio.toDouble(); cx = x; cy = y }
                x += step
            }
            y += step
        }
        if (cx < 0 || bestRatio < 1.15) return null
        val radiusX = maxOf(10f, w * 0.06f)
        val radiusY = maxOf(5f, h * 0.03f)
        return Array(PROXY_POINTS) { i ->
            val angle = 2.0 * Math.PI * i / PROXY_POINTS
            floatArrayOf(
                (cx + kotlin.math.cos(angle).toFloat() * radiusX) / w,
                (cy + kotlin.math.sin(angle).toFloat() * radiusY) / h,
            )
        }
    }

    companion object {
        const val PROXY_POINTS = 16
    }
}
