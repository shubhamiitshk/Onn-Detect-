package com.onndetect.detection.visual

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.lang.reflect.Constructor
import java.lang.reflect.Field
import java.lang.reflect.Method
import java.util.concurrent.Executors

/** Lightweight 2D point avoiding Android framework stub dependencies during JVM tests. */
internal data class Point2D(val x: Float, val y: Float) {
    fun distanceTo(other: Point2D): Float = kotlin.math.hypot(x - other.x, y - other.y)
}

/**
 * Computes pose-normalized biomechanical lip acceleration jitter (BioLip, arXiv:2604.16808)
 * across a sequence of per-frame 468/478 MediaPipe FaceMesh landmarks.
 *
 * Returns Float.NaN when fewer than 3 consecutive frames contain a valid face.
 *
 * The algorithm:
 * 1. For each valid frame, normalize landmarks by inter-ocular distance and head roll
 *    using eye corners (33/263) and nose tip (1) as reference points
 * 2. Extract 40 canonical lip indices (inner + outer contour + corners)
 * 3. Compute MAR (Mouth Aspect Ratio) and second-order temporal acceleration
 * 4. Aggregate across frame triplets: Jitter_accel (0.6 weight) + MAR jerk (0.4 weight)
 * 5. Normalize to [0, 1] with saturation at 0.08 threshold
 *
 * Returns Float.NaN when fewer than 3 consecutive frames contain a valid face.
 */
object PerioralLandmarkAnalyzer {

    // 40 canonical MediaPipe inner + outer lip contour indices
    val LIP_INDICES = intArrayOf(
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        308, 191, 80, 81, 82, 13, 312, 311, 310, 415
    )

    // Canonical left & right malar (cheek) vascular ROI indices for bilateral rPPG phase coherence
    val LEFT_MALAR_INDICES = intArrayOf(116, 117, 118, 123, 187, 205)
    val RIGHT_MALAR_INDICES = intArrayOf(345, 346, 347, 352, 411, 425)

    const val LEFT_EYE_OUTER = 33
    const val RIGHT_EYE_OUTER = 263
    const val NOSE_TIP = 1
    const val UPPER_INNER_LIP = 13
    const val LOWER_INNER_LIP = 14
    const val LEFT_MOUTH_CORNER = 78
    const val RIGHT_MOUTH_CORNER = 308

    const val MIN_EYE_DIST_PX = 12f
    const val MIN_CONSECUTIVE_FRAMES = 3
    const val ACCEL_WEIGHT = 0.6f
    const val MAR_WEIGHT = 0.4f
    const val NORMALIZATION_SCALE = 0.08f

    /**
     * Computes pose-normalized biomechanical lip acceleration jitter (BioLip, arXiv:2604.16808)
     * across a sequence of per-frame 468/478 MediaPipe FaceMesh landmarks.
     *
     * Returns Float.NaN when fewer than 3 consecutive frames contain a valid face.
     *
     * @param frameLandmarks List of per-frame MediaPipe FaceMesh landmarks (468 or 478 pts)
     * @param minEyeDistPx Minimum inter-ocular distance in pixels to accept a frame
     * @return Normalized jitter score in [0, 1], or Float.NaN if insufficient valid frames
     */
    fun computeJitterScore(
        frameLandmarks: List<List<Any>?>,
        minEyeDistPx: Float = 12f
    ): Float {
        // This function uses duck-typing via reflection to work with any landmark type
        // that has x, y properties (NormalizedLandmark, Landmark, etc.)
        val normalized = frameLandmarks.map { pts ->
            if (pts == null || pts.size <= 415) {
                null
            } else {
                try {
                    val le = pts[LEFT_EYE_OUTER]
                    val re = pts[RIGHT_EYE_OUTER]
                    val nose = pts[NOSE_TIP]
                    val dx = (getLandmarkX(re) - getLandmarkX(le)).toDouble()
                    val dy = (getLandmarkY(re) - getLandmarkY(le)).toDouble()
                    val eyeDist = kotlin.math.hypot(dx, dy)
                    if (!eyeDist.isFinite() || eyeDist < minEyeDistPx) {
                        null
                    } else {
                        val cosA = dx / eyeDist
                        val sinA = -dy / eyeDist
                        fun norm(idx: Int): Point2D {
                            val p = pts[idx]
                            val rx = ((getLandmarkX(p) - getLandmarkX(nose)) * cosA - (getLandmarkY(p) - getLandmarkY(nose)) * sinA).toFloat()
                            val ry = ((getLandmarkX(p) - getLandmarkX(nose)) * sinA + (getLandmarkY(p) - getLandmarkY(nose)) * cosA).toFloat()
                            val eyeDistF = eyeDist.toFloat()
                            return Point2D(rx / eyeDistF, ry / eyeDistF)
                        }
                        val lipPts = Array(LIP_INDICES.size) { i -> norm(LIP_INDICES[i]) }
                        val u = norm(UPPER_INNER_LIP); val l = norm(LOWER_INNER_LIP)
                        val lm = norm(LEFT_MOUTH_CORNER); val rm = norm(RIGHT_MOUTH_CORNER)
                        val mar = kotlin.math.hypot(u.x - l.x, u.y - l.y) / (kotlin.math.hypot(lm.x - rm.x, lm.y - rm.y) + 1e-6f)
                        Pair(lipPts, mar)
                    }
                } catch (e: Exception) {
                    Log.w("PerioralLandmarkAnalyzer", "Failed to normalize landmarks", e)
                    null
                }
            }
        }

        if (normalized.size < MIN_CONSECUTIVE_FRAMES) return Float.NaN

        var accelSum = 0.0
        var marJerkSum = 0.0
        var triplets = 0
        for (t in 1 until normalized.size - 1) {
            val prev = normalized[t - 1] ?: continue
            val curr = normalized[t] ?: continue
            val next = normalized[t + 1] ?: continue
            var frameAccel = 0.0
            for (i in LIP_INDICES.indices) {
                val ax = next.first[i].x - 2f * curr.first[i].x + prev.first[i].x
                val ay = next.first[i].y - 2f * curr.first[i].y + prev.first[i].y
                frameAccel += kotlin.math.hypot(ax.toDouble(), ay.toDouble())
            }
            accelSum += frameAccel / LIP_INDICES.size
            marJerkSum += kotlin.math.abs(next.second - 2f * curr.second + prev.second).toDouble()
            triplets++
        }
        if (triplets == 0) return Float.NaN
        val meanAccel = (accelSum / triplets).toFloat()
        val meanMarJerk = (marJerkSum / triplets).toFloat()
        val raw = ACCEL_WEIGHT * meanAccel + MAR_WEIGHT * meanMarJerk
        return (raw / NORMALIZATION_SCALE).coerceIn(0f, 1f)
    }

    /**
     * Extracts normalized per-frame Mouth Aspect Ratio (MAR) values from frame landmarks.
     * Returns an empty FloatArray if fewer than [MIN_CONSECUTIVE_FRAMES] frames have valid faces.
     */
    fun extractMarSeries(
        frameLandmarks: List<List<Any>?>,
        minEyeDistPx: Float = 12f
    ): FloatArray {
        val series = mutableListOf<Float>()
        for (pts in frameLandmarks) {
            if (pts == null || pts.size <= 415) continue
            try {
                val le = pts[LEFT_EYE_OUTER]
                val re = pts[RIGHT_EYE_OUTER]
                val nose = pts[NOSE_TIP]
                val dx = (getLandmarkX(re) - getLandmarkX(le))
                val dy = (getLandmarkY(re) - getLandmarkY(le))
                val eyeDist = kotlin.math.hypot(dx, dy)
                if (!eyeDist.isFinite() || eyeDist < minEyeDistPx) continue
                val cosA = dx / eyeDist
                val sinA = -dy / eyeDist
                fun norm(idx: Int): Point2D {
                    val p = pts[idx]
                    val rx = ((getLandmarkX(p) - getLandmarkX(nose)) * cosA - (getLandmarkY(p) - getLandmarkY(nose)) * sinA).toFloat()
                    val ry = ((getLandmarkX(p) - getLandmarkX(nose)) * sinA + (getLandmarkY(p) - getLandmarkY(nose)) * cosA).toFloat()
                    val eyeDistF = eyeDist.toFloat()
                    return Point2D(rx / eyeDistF, ry / eyeDistF)
                }
                val u = norm(UPPER_INNER_LIP); val l = norm(LOWER_INNER_LIP)
                val lm = norm(LEFT_MOUTH_CORNER); val rm = norm(RIGHT_MOUTH_CORNER)
                val mar = kotlin.math.hypot((u.x - l.x).toDouble(), (u.y - l.y).toDouble()) /
                    (kotlin.math.hypot((lm.x - rm.x).toDouble(), (lm.y - rm.y).toDouble()) + 1e-6)
                series.add(mar.toFloat())
            } catch (_: Exception) {
                // skip frame with invalid landmarks
            }
        }
        return series.toFloatArray()
    }

    /**
     * Computes the maximum cross-correlation in `[0, 1]` between per-frame MAR and frame-synchronous
     * audio RMS envelope across lag window `[-2, +2]` frames.
     *
     * Semantic Contract for Fusion Slot 8 (`lip_sync_corr`):
     * - Authentic synchronized speech: correlation is positive (typically ~0.7 to 1.0; neutral default is 0.9f).
     * - Desynchronized deepfake / reenactment: correlation is near 0.0 or negative.
     * - Returns Float.NaN when fewer than 6 frames are available or signals have negligible variance.
     */
    fun computeAvffLipSyncCorrelation(
        marSeries: FloatArray,
        audioRmsEnvelope: FloatArray
    ): Float {
        val n = minOf(marSeries.size, audioRmsEnvelope.size)
        if (n < 6) return Float.NaN

        val mSub = DoubleArray(n) { marSeries[it].toDouble() }
        val aSub = DoubleArray(n) { audioRmsEnvelope[it].toDouble() }

        var mSum = 0.0
        var aSum = 0.0
        for (i in 0 until n) {
            mSum += mSub[i]
            aSum += aSub[i]
        }
        val mMean = mSum / n
        val aMean = aSum / n

        var mVar = 0.0
        var aVar = 0.0
        for (i in 0 until n) {
            mSub[i] -= mMean
            aSub[i] -= aMean
            mVar += mSub[i] * mSub[i]
            aVar += aSub[i] * aSub[i]
        }
        val mStd = kotlin.math.sqrt(mVar / n)
        val aStd = kotlin.math.sqrt(aVar / n)
        if (mStd < 1e-6 || aStd < 1e-6) return Float.NaN

        for (i in 0 until n) {
            mSub[i] /= mStd
            aSub[i] /= aStd
        }

        var bestCorr = -1.0
        for (lag in intArrayOf(-2, -1, 0, 1, 2)) {
            var sum = 0.0
            val count: Int
            when {
                lag < 0 -> {
                    count = n + lag
                    for (i in 0 until count) {
                        sum += mSub[i] * aSub[i - lag]
                    }
                }
                lag > 0 -> {
                    count = n - lag
                    for (i in 0 until count) {
                        sum += mSub[i + lag] * aSub[i]
                    }
                }
                else -> {
                    count = n
                    for (i in 0 until count) {
                        sum += mSub[i] * aSub[i]
                    }
                }
            }
            if (count > 0) {
                val c = sum / count
                if (c > bestCorr) bestCorr = c
            }
        }
        return kotlin.math.max(0.0, kotlin.math.min(1.0, bestCorr)).toFloat()
    }

    /**
     * Returns the AVFF lip-sync anomaly score in `[0, 1]`.
     * Anomaly is `1.0f - correlation`.
     *
     * WARNING: Slot 8 in XGBoostMetaClassifier expects CORRELATION (authentic = high),
     * NOT anomaly score. Never inject anomaly score directly into slot 8.
     */
    fun computeAvffLipSyncAnomaly(
        marSeries: FloatArray,
        audioRmsEnvelope: FloatArray
    ): Float {
        val corr = computeAvffLipSyncCorrelation(marSeries, audioRmsEnvelope)
        return if (corr.isNaN()) Float.NaN else (1.0f - corr).coerceIn(0f, 1f)
    }

    /**
     * Computes Bilateral Malar rPPG (Remote Photoplethysmography) Chromaticity Phase Anomaly in [0, 1].
     *
     * In a real living face, green-normalized skin chromaticity `G / (R + G + B)` across the left cheek
     * and right cheek oscillates in phase (cardiac blood-volume pulse + shared illumination, rho >= 0.55).
     * In face-swap, neural reenactment, or diffusion-generated video (Sora, HeyGen, DeepFaceLab),
     * micro-vascular chromaticity across bilateral malar patches is uncorrelated or suppressed.
     */
    fun computeRppgBilateralPhaseAnomaly(
        frames: List<Bitmap>,
        frameLandmarks: List<List<Any>?>
    ): Float {
        val nFrames = minOf(frames.size, frameLandmarks.size)
        if (nFrames < 4) return Float.NaN

        val perFrameLeft = arrayOfNulls<Double>(nFrames)
        val perFrameRight = arrayOfNulls<Double>(nFrames)

        for (t in 0 until nFrames) {
            val bmp = frames[t]
            val pts = frameLandmarks[t]
            if (bmp.isRecycled || bmp.width < 16 || bmp.height < 16 || pts == null || pts.size <= 425) continue
            try {
                val leftCg = sampleMalarChromaticity(bmp, pts, LEFT_MALAR_INDICES)
                val rightCg = sampleMalarChromaticity(bmp, pts, RIGHT_MALAR_INDICES)
                if (leftCg.isFinite() && rightCg.isFinite()) {
                    perFrameLeft[t] = leftCg
                    perFrameRight[t] = rightCg
                }
            } catch (_: Exception) {
                // Skip invalid frame
            }
        }

        return computeRppgAnomalyFromTimeline(perFrameLeft.toList(), perFrameRight.toList())
    }

    /**
     * Computes bilateral malar rPPG phase anomaly strictly across contiguous, uninterrupted
     * runs of tracked frames from timeline-indexed series (null represents missing/untracked frames).
     *
     * Returns Float.NaN when fewer than 4 contiguous frames exist.
     */
    internal fun computeRppgAnomalyFromTimeline(
        leftTimeline: List<Double?>,
        rightTimeline: List<Double?>
    ): Float {
        val n = minOf(leftTimeline.size, rightTimeline.size)
        if (n < 4) return Float.NaN

        val contiguousRuns = mutableListOf<MutableList<Pair<Double, Double>>>()
        var currentRun = mutableListOf<Pair<Double, Double>>()

        for (t in 0 until n) {
            val l = leftTimeline[t]
            val r = rightTimeline[t]
            if (l != null && r != null && l.isFinite() && r.isFinite()) {
                currentRun.add(Pair(l, r))
            } else {
                if (currentRun.isNotEmpty()) {
                    contiguousRuns.add(currentRun)
                    currentRun = mutableListOf()
                }
            }
        }
        if (currentRun.isNotEmpty()) {
            contiguousRuns.add(currentRun)
        }

        val validRuns = contiguousRuns.filter { it.size >= 4 }
        if (validRuns.isEmpty()) return Float.NaN

        val longestRun = validRuns.maxByOrNull { it.size } ?: return Float.NaN
        val leftSeries = DoubleArray(longestRun.size) { longestRun[it].first }
        val rightSeries = DoubleArray(longestRun.size) { longestRun[it].second }
        return computeRppgAnomalyFromSeries(leftSeries, rightSeries)
    }

    /**
     * Pure helper for unit-testing bilateral rPPG chromaticity correlation directly from
     * per-frame left/right malar green-ratio series.
     */
    internal fun computeRppgAnomalyFromSeries(leftSeries: DoubleArray, rightSeries: DoubleArray): Float {
        val n = minOf(leftSeries.size, rightSeries.size)
        if (n < 4) return Float.NaN
        val m = n - 1
        val dLeft = DoubleArray(m) { i -> leftSeries[i + 1] - leftSeries[i] }
        val dRight = DoubleArray(m) { i -> rightSeries[i + 1] - rightSeries[i] }
        val corr = pearsonCorrelation(dLeft, dRight)
        if (!corr.isFinite()) return Float.NaN
        return ((0.55 - corr) / 0.85).coerceIn(0.0, 1.0).toFloat()
    }

    private fun pearsonCorrelation(x: DoubleArray, y: DoubleArray): Double {
        val n = minOf(x.size, y.size)
        if (n < 2) return Double.NaN
        var meanX = 0.0
        var meanY = 0.0
        for (i in 0 until n) {
            meanX += x[i]
            meanY += y[i]
        }
        meanX /= n
        meanY /= n
        var cov = 0.0
        var varX = 0.0
        var varY = 0.0
        for (i in 0 until n) {
            val dx = x[i] - meanX
            val dy = y[i] - meanY
            cov += dx * dy
            varX += dx * dx
            varY += dy * dy
        }
        val denom = kotlin.math.sqrt(varX * varY)
        if (denom < 1e-12) return Double.NaN
        return (cov / denom).coerceIn(-1.0, 1.0)
    }

    private fun sampleMalarChromaticity(bmp: Bitmap, pts: List<Any>, indices: IntArray): Double {
        var gRatioSum = 0.0
        var count = 0
        val w = bmp.width
        val h = bmp.height
        for (idx in indices) {
            if (idx >= pts.size) continue
            val lm = pts[idx]
            val rawX = getLandmarkX(lm)
            val rawY = getLandmarkY(lm)
            val cx = (if (rawX in 0.0..1.0 && rawY in 0.0..1.0) rawX * w else rawX).toInt().coerceIn(1, w - 2)
            val cy = (if (rawX in 0.0..1.0 && rawY in 0.0..1.0) rawY * h else rawY).toInt().coerceIn(1, h - 2)
            for (dy in -1..1) {
                for (dx in -1..1) {
                    val c = bmp.getPixel(cx + dx, cy + dy)
                    val r = (c shr 16 and 0xFF).toDouble()
                    val g = (c shr 8 and 0xFF).toDouble()
                    val b = (c and 0xFF).toDouble()
                    val total = r + g + b
                    if (total > 15.0) {
                        gRatioSum += g / total
                        count++
                    }
                }
            }
        }
        return if (count > 0) gRatioSum / count else Double.NaN
    }

    private fun getLandmarkX(landmark: Any): Double {
        return try {
            // Try property access first (Kotlin property / field)
            try {
                val field = landmark.javaClass.getDeclaredField("x")
                field.isAccessible = true
                return field.getDouble(landmark)
            } catch (_: NoSuchFieldException) { }

            // Try getter methods: getX() or x()
            val method = try {
                landmark.javaClass.getMethod("getX")
            } catch (_: NoSuchMethodException) {
                landmark.javaClass.getMethod("x")
            }
            val res = method.invoke(landmark)
            (res as? Number)?.toDouble() ?: throw IllegalStateException("Unexpected return type: $res")
        } catch (e: Exception) {
            throw RuntimeException("Failed to get landmark X", e)
        }
    }

    private fun getLandmarkY(landmark: Any): Double {
        return try {
            try {
                val field = landmark.javaClass.getDeclaredField("y")
                field.isAccessible = true
                return field.getDouble(landmark)
            } catch (_: NoSuchFieldException) { }

            val method = try {
                landmark.javaClass.getMethod("getY")
            } catch (_: NoSuchMethodException) {
                landmark.javaClass.getMethod("y")
            }
            val res = method.invoke(landmark)
            (res as? Number)?.toDouble() ?: throw IllegalStateException("Unexpected return type: $res")
        } catch (e: Exception) {
            throw RuntimeException("Failed to get landmark Y", e)
        }
    }
}

/**
 * Detects lip-sync deepfakes using BioLip biomechanical landmarks.
 *
 * Uses reflection to load MediaPipe FaceLandmarker at runtime, allowing the code
 * to compile even if the MediaPipe FaceLandmarker class is not available in the
 * current dependency version.
 *
 * Usage:
 *   val detector = LipSyncDetector(context)
 *   detector.initialize()
 *   val result = detector.analyzeVideo(uri)
 *   detector.close()
 *
 * Returns Float.NaN when video has no face or insufficient frames.
 */
class LipSyncDetector(private val context: Context) : AutoCloseable {

    companion object {
        const val MODEL_ASSET_PATH = "models/face_landmarker.task"
    }

    private var faceLandmarker: Any? = null
    private var faceLandmarkerClass: Class<*>? = null
    private var detectForVideoMethod: Method? = null
    private var closeMethod: Method? = null
    private var faceLandmarksMethod: Method? = null
    private var faceLandmarksField: Field? = null
    private var landmarkXMethod: Method? = null
    private var landmarkYMethod: Method? = null
    private var imageClass: Class<*>? = null
    private var bitmapImageBuilderConstructor: Constructor<*>? = null
    private var buildImageMethod: Method? = null
    private val executor = Executors.newSingleThreadExecutor()

    fun initialize() {
        try {
            // Load MediaPipe FaceLandmarker classes via reflection using 0.10+ package hierarchy
            faceLandmarkerClass = Class.forName("com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker")
            val optionsClass = Class.forName("com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker\$FaceLandmarkerOptions")
            val baseOptionsClass = Class.forName("com.google.mediapipe.tasks.core.BaseOptions")
            val runningModeClass = Class.forName("com.google.mediapipe.tasks.vision.core.RunningMode")
            imageClass = Class.forName("com.google.mediapipe.tasks.vision.core.Image")
            val bitmapImageBuilderClass = Class.forName("com.google.mediapipe.tasks.vision.core.Image\$BitmapImageBuilder")

            // BaseOptions.builder().setModelAssetPath(MODEL_ASSET_PATH).build()
            val baseBuilder = baseOptionsClass.getMethod("builder").invoke(null)
            baseBuilder.javaClass.getMethod("setModelAssetPath", String::class.java).invoke(baseBuilder, MODEL_ASSET_PATH)
            val baseOptions = baseBuilder.javaClass.getMethod("build").invoke(baseBuilder)

            // FaceLandmarkerOptions.builder().setBaseOptions(baseOptions).setRunningMode(RunningMode.VIDEO).build()
            val builder = optionsClass.getMethod("builder").invoke(null)
            builder.javaClass.getMethod("setBaseOptions", baseOptionsClass).invoke(builder, baseOptions)
            builder.javaClass.getMethod("setOutputFaceBlendshapes", Boolean::class.javaPrimitiveType).invoke(builder, false)
            builder.javaClass.getMethod("setOutputFacialTransformationMatrix", Boolean::class.javaPrimitiveType).invoke(builder, false)
            builder.javaClass.getMethod("setRunningMode", runningModeClass).invoke(builder, runningModeClass.getField("VIDEO").get(null))
            val options = builder.javaClass.getMethod("build").invoke(builder)

            faceLandmarker = faceLandmarkerClass!!.getMethod("createFromOptions", Context::class.java, optionsClass).invoke(null, context, options)

            // Cache methods for later use
            detectForVideoMethod = faceLandmarkerClass!!.getMethod("detectForVideo", imageClass!!, Long::class.javaPrimitiveType)
            closeMethod = faceLandmarkerClass!!.getMethod("close")

            val resultClass = Class.forName("com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult")
            faceLandmarksMethod = runCatching { resultClass.getMethod("faceLandmarks") }.getOrNull()
            faceLandmarksField = runCatching { resultClass.getField("faceLandmarks") }.getOrNull()

            bitmapImageBuilderConstructor = bitmapImageBuilderClass.getConstructor(Bitmap::class.java)
            buildImageMethod = bitmapImageBuilderClass.getMethod("build")

            val normalizedLandmarkClass = Class.forName("com.google.mediapipe.tasks.components.containers.NormalizedLandmark")
            landmarkXMethod = runCatching { normalizedLandmarkClass.getMethod("x") }
                .getOrElse { normalizedLandmarkClass.getMethod("getX") }
            landmarkYMethod = runCatching { normalizedLandmarkClass.getMethod("y") }
                .getOrElse { normalizedLandmarkClass.getMethod("getY") }

            Log.i("LipSyncDetector", "MediaPipe FaceLandmarker initialized successfully with $MODEL_ASSET_PATH")
        } catch (e: Exception) {
            Log.w("LipSyncDetector", "MediaPipe FaceLandmarker not available: ${e.message}. Lip-sync detection will be disabled.")
            faceLandmarker = null
        }
    }

    /**
     * Analyzes a video file for lip-sync deepfake artifacts using BioLip biomechanical landmarks.
     *
     * @param videoUri Content URI of video file
     * @return BioLip jitter score in [0, 1] where higher = more likely deepfake,
     *         or Float.NaN if insufficient valid frames or MediaPipe not available
     */
    suspend fun analyzeVideo(videoUri: Uri): Float = withContext(Dispatchers.IO) {
        if (faceLandmarker == null || detectForVideoMethod == null || imageClass == null ||
            bitmapImageBuilderConstructor == null || buildImageMethod == null) {
            return@withContext Float.NaN
        }

        val fileDescriptor = try {
            context.contentResolver.openFileDescriptor(videoUri, "r")
        } catch (e: Exception) {
            Log.w("LipSyncDetector", "Failed to open video URI: $videoUri", e)
            null
        } ?: return@withContext Float.NaN

        fileDescriptor.use { pfd ->
            val mediaMetadata = MediaMetadataRetriever()
            try {
                mediaMetadata.setDataSource(pfd.fileDescriptor)
                val rawFrameCount = mediaMetadata.extractMetadata(
                    MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT
                )?.toIntOrNull()
                val durationMs = mediaMetadata.extractMetadata(
                    MediaMetadataRetriever.METADATA_KEY_DURATION
                )?.toLongOrNull() ?: 0L

                val frameCount = when {
                    rawFrameCount != null && rawFrameCount > 0 -> rawFrameCount
                    durationMs > 0L -> (durationMs * 30L / 1000L).toInt()
                    else -> 0
                }

                if (frameCount < 3) return@withContext Float.NaN

                val landmarks = mutableListOf<List<Any>?>()
                val frameIndices = (0 until frameCount step 2).take(30) // Sample up to 30 frames

                for (frameIdx in frameIndices) {
                    val timeUs = (frameIdx * 1_000_000L / 30L)
                    val bitmap = mediaMetadata.getFrameAtTime(
                        timeUs,
                        MediaMetadataRetriever.OPTION_CLOSEST_SYNC
                    ) ?: continue

                    try {
                        val builder = bitmapImageBuilderConstructor!!.newInstance(bitmap)
                        val mpImage = buildImageMethod!!.invoke(builder)
                        val timestampMs = (frameIdx * 1000L / 30L)

                        val result = detectForVideoMethod!!.invoke(faceLandmarker, mpImage, timestampMs)
                        val faceLandmarksList = (faceLandmarksMethod?.invoke(result) ?: faceLandmarksField?.get(result)) as? List<*>
                        val face = faceLandmarksList?.firstOrNull() as? List<*>
                        if (face == null || face.isEmpty()) {
                            landmarks.add(null)
                            continue
                        }

                        // Pass the full face landmarks list (all 478 points) so pose normalization
                        // and eye-distance scaling can locate outer eye corners (33, 263) and nose tip (1).
                        val fullFaceLandmarks = ArrayList<Any>(face.size)
                        for (item in face) {
                            if (item != null) fullFaceLandmarks.add(item)
                        }
                        landmarks.add(fullFaceLandmarks)
                    } finally {
                        if (!bitmap.isRecycled) {
                            bitmap.recycle()
                        }
                    }
                }

                PerioralLandmarkAnalyzer.computeJitterScore(landmarks)
            } catch (e: Exception) {
                Log.w("LipSyncDetector", "Error during video analysis", e)
                Float.NaN
            } finally {
                try {
                    mediaMetadata.release()
                } catch (e: Exception) {
                    Log.w("LipSyncDetector", "Error releasing MediaMetadataRetriever", e)
                }
            }
        }
    }

    override fun close() {
        if (faceLandmarker != null && closeMethod != null) {
            try {
                closeMethod!!.invoke(faceLandmarker)
            } catch (e: Exception) {
                Log.w("LipSyncDetector", "Error closing FaceLandmarker", e)
            }
        }
        faceLandmarker = null
        executor.shutdown()
    }
}