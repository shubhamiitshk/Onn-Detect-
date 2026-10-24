package com.onndetect.detection.visual

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.lang.reflect.Field
import java.lang.reflect.Method

/**
 * BioLip Landmark Analyzer - computes lip-sync jitter score from video frames.
 *
 * Uses reflection to load MediaPipe FaceLandmarker at runtime, allowing the code
 * to compile even if the MediaPipe FaceLandmarker class is not available in the
 * current dependency version.
 */
class BioLipLandmarkAnalyzer(private val context: Context) {

    companion object {
        const val MIN_CONSECUTIVE_FRAMES = 3
    }

    private var faceLandmarker: Any? = null
    private var faceLandmarkerClass: Class<*>? = null
    private var detectForVideoMethod: Method? = null
    private var faceLandmarksField: Field? = null
    private var landmarkXMethod: Method? = null
    private var landmarkYMethod: Method? = null
    private var imageClass: Class<*>? = null
    private var bitmapImageBuilderClass: Class<*>? = null

    init {
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

            detectForVideoMethod = faceLandmarkerClass!!.getMethod("detectForVideo", imageClass!!, Long::class.javaPrimitiveType)
            faceLandmarksField = Class.forName("com.google.mediapipe.tasks.vision.face.FaceLandmarkerResult").getField("faceLandmarks")
            val normalizedLandmarkClass = Class.forName("com.google.mediapipe.tasks.components.containers.NormalizedLandmark")
            landmarkXMethod = normalizedLandmarkClass.getMethod("getX")
            landmarkYMethod = normalizedLandmarkClass.getMethod("getY")

            Log.i("BioLipLandmarkAnalyzer", "MediaPipe FaceLandmarker initialized successfully")
        } catch (e: Exception) {
            Log.w("BioLipLandmarkAnalyzer", "MediaPipe FaceLandmarker not available: ${e.message}. BioLip analysis will be disabled.")
            faceLandmarker = null
        }
    }

    /**
     * Analyzes a sequence of video frames for lip-sync deepfake artifacts
     * using biomechanical lip movement analysis.
     *
     * @param frames List of video frames (Bitmap objects)
     * @return BioLipResult containing the landmark jitter score and metadata
     */
    fun analyzeFrames(frames: List<Bitmap>): BioLipResult {
        if (frames.size < 3) {
            return BioLipResult(
                landmarkJitter = Float.NaN,
                hasFace = false,
                notes = "Insufficient frames for BioLip analysis (need >= 3 frames)"
            )
        }

        if (faceLandmarker == null || detectForVideoMethod == null || imageClass == null || bitmapImageBuilderClass == null) {
            return BioLipResult(
                landmarkJitter = Float.NaN,
                hasFace = false,
                notes = "MediaPipe FaceLandmarker not available. Add com.google.mediapipe:tasks-vision with FaceLandmarker support."
            )
        }

        val landmarks = mutableListOf<List<Any>?>()

        try {
            for ((index, frame) in frames.withIndex()) {
                val builder = bitmapImageBuilderClass!!.getConstructor(Bitmap::class.java).newInstance(frame)
                val mpImage = builder.javaClass.getMethod("build").invoke(builder)
                val timestampMs = (index * 33L).toLong() // ~30fps

                val result = detectForVideoMethod!!.invoke(faceLandmarker, mpImage, timestampMs)
                val faceLandmarksList = faceLandmarksField!!.get(result) as List<*>
                val face = faceLandmarksList.firstOrNull() ?: run {
                    landmarks.add(null)
                    continue
                }

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
            Log.w("BioLipLandmarkAnalyzer", "Error during frame analysis", e)
            return BioLipResult(
                landmarkJitter = Float.NaN,
                hasFace = false,
                notes = "Error during BioLip analysis: ${e.message}"
            )
        }

        val jitterScore = PerioralLandmarkAnalyzer.computeJitterScore(landmarks)
        val marSeries = PerioralLandmarkAnalyzer.extractMarSeries(landmarks)
        val hasFace = landmarks.any { it != null && it.isNotEmpty() }

        return BioLipResult(
            landmarkJitter = jitterScore,
            hasFace = hasFace,
            notes = if (hasFace) "BioLip analysis complete" else "No face detected in frames",
            marSeries = marSeries
        )
    }
}

/**
 * Result of BioLip lip-sync analysis.
 *
 * @param landmarkJitter Normalized jitter score in [0, 1], or NaN if unavailable
 * @param hasFace Whether a face was detected in the frames
 * @param notes Diagnostic notes about the analysis
 * @param marSeries Per-frame Mouth Aspect Ratio sequence for AVFF synchrony correlation
 */
data class BioLipResult(
    val landmarkJitter: Float,
    val hasFace: Boolean,
    val notes: String?,
    val marSeries: FloatArray = FloatArray(0)
)