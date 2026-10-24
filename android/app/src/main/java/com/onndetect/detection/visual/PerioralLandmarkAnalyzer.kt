package com.onndetect.detection.visual

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
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
    @Suppress("UNUSED_PARAMETER")
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
                    if (!eyeDist.isFinite() || eyeDist < MIN_EYE_DIST_PX) {
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
        }.filterNotNull()

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

    private fun getLandmarkX(landmark: Any): Double {
        return try {
            // Try property access first (Kotlin property)
            landmark.javaClass.getDeclaredField("x").let { field ->
                field.isAccessible = true
                field.getDouble(landmark)
            }
        } catch (_: NoSuchFieldException) {
            // Try getter method
            try {
                landmark.javaClass.getMethod("getX").invoke(landmark) as Double
            } catch (_: Exception) {
                (landmark.javaClass.getMethod("getX").invoke(landmark) as Number).toDouble()
            }
        } catch (e: Exception) {
            throw RuntimeException("Failed to get landmark X", e)
        }
    }

    private fun getLandmarkY(landmark: Any): Double {
        return try {
            landmark.javaClass.getDeclaredField("y").let { field ->
                field.isAccessible = true
                field.getDouble(landmark)
            }
        } catch (_: NoSuchFieldException) {
            try {
                landmark.javaClass.getMethod("getY").invoke(landmark) as Double
            } catch (_: Exception) {
                (landmark.javaClass.getMethod("getY").invoke(landmark) as Number).toDouble()
            }
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

    private var faceLandmarker: Any? = null
    private var faceLandmarkerClass: Class<*>? = null
    private var detectForVideoMethod: Method? = null
    private var closeMethod: Method? = null
    private var faceLandmarksField: Field? = null
    private var landmarkXMethod: Method? = null
    private var landmarkYMethod: Method? = null
    private var imageClass: Class<*>? = null
    private var bitmapImageBuilderClass: Class<*>? = null
    private val executor = Executors.newSingleThreadExecutor()

    fun initialize() {
        try {
            // Try to load MediaPipe FaceLandmarker classes via reflection
            faceLandmarkerClass = Class.forName("com.google.mediapipe.tasks.vision.face.FaceLandmarker")
            val optionsClass = Class.forName("com.google.mediapipe.tasks.vision.face.FaceLandmarkerOptions")
            val optionsBuilderClass = Class.forName("com.google.mediapipe.tasks.vision.face.FaceLandmarkerOptions\$Builder")
            val runningModeClass = Class.forName("com.google.mediapipe.tasks.vision.core.RunningMode")
            imageClass = Class.forName("com.google.mediapipe.tasks.vision.core.Image")
            bitmapImageBuilderClass = Class.forName("com.google.mediapipe.tasks.vision.core.Image\$BitmapImageBuilder")

            val builder = optionsBuilderClass.getDeclaredConstructor().newInstance()
            builder.javaClass.getMethod("setOutputFaceBlendshapes", Boolean::class.javaPrimitiveType).invoke(builder, false)
            builder.javaClass.getMethod("setOutputFacialTransformationMatrix", Boolean::class.javaPrimitiveType).invoke(builder, false)
            builder.javaClass.getMethod("setRunningMode", runningModeClass).invoke(builder, runningModeClass.getField("VIDEO").get(null))
            val options = builder.javaClass.getMethod("build").invoke(builder)

            faceLandmarker = faceLandmarkerClass!!.getMethod("createFromOptions", Context::class.java, optionsClass).invoke(null, context, options)

            // Cache methods for later use
            detectForVideoMethod = faceLandmarkerClass!!.getMethod("detectForVideo", imageClass!!, Long::class.javaPrimitiveType)
            closeMethod = faceLandmarkerClass!!.getMethod("close")
            faceLandmarksField = Class.forName("com.google.mediapipe.tasks.vision.face.FaceLandmarkerResult").getField("faceLandmarks")
            val normalizedLandmarkClass = Class.forName("com.google.mediapipe.tasks.components.containers.NormalizedLandmark")
            landmarkXMethod = normalizedLandmarkClass.getMethod("getX")
            landmarkYMethod = normalizedLandmarkClass.getMethod("getY")

            Log.i("LipSyncDetector", "MediaPipe FaceLandmarker initialized successfully")
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
        if (faceLandmarker == null || detectForVideoMethod == null || imageClass == null || bitmapImageBuilderClass == null) {
            return@withContext Float.NaN
        }

        val fileDescriptor = context.contentResolver.openFileDescriptor(videoUri, "r")
        if (fileDescriptor == null) return@withContext Float.NaN

        val mediaMetadata = MediaMetadataRetriever()
        mediaMetadata.setDataSource(fileDescriptor.fileDescriptor)
        val frameCount = mediaMetadata.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull() ?: 0
        mediaMetadata.release()

        if (frameCount < 3) return@withContext Float.NaN

        val landmarks = mutableListOf<List<Any>?>()
        val frameIndices = (0 until frameCount step 2).take(30) // Sample up to 30 frames

        try {
            for (frameIdx in frameIndices) {
                val bitmap = mediaMetadata.getFrameAtTime(
                    (frameIdx * 1000000L * 1000 / 30).toLong(),
                    MediaMetadataRetriever.OPTION_CLOSEST_SYNC
                ) ?: continue

                val builder = bitmapImageBuilderClass!!.getConstructor(Bitmap::class.java).newInstance(bitmap)
                val mpImage = builder.javaClass.getMethod("build").invoke(builder)
                val timestampMs = (frameIdx * 1000L / 30).toLong()

                val result = detectForVideoMethod!!.invoke(faceLandmarker, mpImage, timestampMs)
                val faceLandmarksList = faceLandmarksField!!.get(result) as List<*>
                val face = faceLandmarksList.firstOrNull() ?: continue

                val lipLandmarks = mutableListOf<Any>()
                val faceSize = (face as? List<*>)?.size ?: (face.javaClass.getMethod("size").invoke(face) as Int)
                for (idx in PerioralLandmarkAnalyzer.LIP_INDICES) {
                    if (idx < faceSize) {
                        val lm = (face as? List<*>)?.get(idx) ?: face.javaClass.getMethod("get", Int::class.javaPrimitiveType).invoke(face, idx)
                        lipLandmarks.add(lm)
                    }
                }
                landmarks.add(lipLandmarks)
            }
        } catch (e: Exception) {
            Log.w("LipSyncDetector", "Error during video analysis", e)
            return@withContext Float.NaN
        }

        PerioralLandmarkAnalyzer.computeJitterScore(landmarks)
    }

    override fun close() {
        if (faceLandmarker != null && closeMethod != null) {
            try {
                closeMethod!!.invoke(faceLandmarker)
            } catch (e: Exception) {
                Log.w("LipSyncDetector", "Error closing FaceLandmarker", e)
            }
        }
        executor.shutdown()
    }
}