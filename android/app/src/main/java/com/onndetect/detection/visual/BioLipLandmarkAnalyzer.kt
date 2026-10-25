package com.onndetect.detection.visual

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.lang.reflect.Constructor
import java.lang.reflect.Method

/**
 * BioLip Landmark Analyzer - computes lip-sync jitter score from video frames.
 *
 * Uses reflection to load MediaPipe FaceLandmarker at runtime, allowing the code
 * to execute safely on Android devices while gracefully degrading on JVM unit tests.
 */
class BioLipLandmarkAnalyzer(private val context: Context) {

    companion object {
        const val MIN_CONSECUTIVE_FRAMES = 3
        const val MODEL_ASSET_PATH = "models/face_landmarker.task"
    }

    private var faceLandmarker: Any? = null
    private var detectForVideoMethod: Method? = null
    private var faceLandmarksMethod: Method? = null
    private var imageClass: Class<*>? = null
    private var bitmapImageBuilderConstructor: Constructor<*>? = null
    private var buildImageMethod: Method? = null

    init {
        try {
            val faceLandmarkerClass = Class.forName("com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker")
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

            // FaceLandmarker.createFromOptions(context, options)
            faceLandmarker = faceLandmarkerClass.getMethod("createFromOptions", Context::class.java, optionsClass).invoke(null, context, options)

            detectForVideoMethod = faceLandmarkerClass.getMethod("detectForVideo", imageClass, Long::class.javaPrimitiveType)
            val resultClass = Class.forName("com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult")
            faceLandmarksMethod = resultClass.getMethod("faceLandmarks")

            bitmapImageBuilderConstructor = bitmapImageBuilderClass.getConstructor(Bitmap::class.java)
            buildImageMethod = bitmapImageBuilderClass.getMethod("build")

            Log.i("BioLipLandmarkAnalyzer", "MediaPipe FaceLandmarker initialized successfully with $MODEL_ASSET_PATH")
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
        if (frames.size < MIN_CONSECUTIVE_FRAMES) {
            return BioLipResult(
                landmarkJitter = Float.NaN,
                hasFace = false,
                notes = "Insufficient frames for BioLip analysis (need >= 3 frames)"
            )
        }

        if (faceLandmarker == null || detectForVideoMethod == null || faceLandmarksMethod == null ||
            imageClass == null || bitmapImageBuilderConstructor == null || buildImageMethod == null) {
            return BioLipResult(
                landmarkJitter = Float.NaN,
                hasFace = false,
                notes = "MediaPipe FaceLandmarker not available"
            )
        }

        val landmarks = mutableListOf<List<Any>?>()

        try {
            for ((index, frame) in frames.withIndex()) {
                val builder = bitmapImageBuilderConstructor!!.newInstance(frame)
                val mpImage = buildImageMethod!!.invoke(builder)
                val timestampMs = index * 33L // ~30fps

                val result = detectForVideoMethod!!.invoke(faceLandmarker, mpImage, timestampMs)
                val faceLandmarksList = faceLandmarksMethod!!.invoke(result) as? List<*>
                val face = faceLandmarksList?.firstOrNull() as? List<*>
                if (face == null || face.isEmpty()) {
                    landmarks.add(null)
                    continue
                }

                // Pass the full face landmarks list (all 478 landmarks) so pose normalization
                // and eye-distance scaling can locate outer eye corners (33, 263) and nose tip (1).
                val fullFaceLandmarks = ArrayList<Any>(face.size)
                for (item in face) {
                    if (item != null) fullFaceLandmarks.add(item)
                }
                landmarks.add(fullFaceLandmarks)
            }
        } catch (e: Exception) {
            Log.w("BioLipLandmarkAnalyzer", "Error during frame analysis", e)
            return BioLipResult(
                landmarkJitter = Float.NaN,
                hasFace = false,
                notes = "Error during BioLip analysis: ${e.message}"
            )
        }

        // Keep `landmarkJitter` as the pure geometric BioLip acceleration measurement
        // so it is never diluted by sparse-frame rPPG noise; expose `rppgAnomaly`
        // separately as an experimental diagnostic field.
        val jitterScore = PerioralLandmarkAnalyzer.computeJitterScore(landmarks)
        val rppgAnomaly = PerioralLandmarkAnalyzer.computeRppgBilateralPhaseAnomaly(frames, landmarks)
        val marSeries = PerioralLandmarkAnalyzer.extractMarSeries(landmarks)
        val hasFace = landmarks.any { it != null && it.size > 415 }

        return BioLipResult(
            landmarkJitter = jitterScore,
            hasFace = hasFace,
            notes = if (hasFace) "BioLip analysis complete" else "No face detected in frames",
            marSeries = marSeries,
            rppgPhaseAnomaly = rppgAnomaly,
        )
    }
}

/**
 * Result of BioLip lip-sync and bilateral malar rPPG analysis.
 *
 * @param landmarkJitter Normalized jitter + rPPG phase anomaly score in [0, 1], or NaN if unavailable
 * @param hasFace Whether a face was detected in the frames
 * @param notes Diagnostic notes about the analysis
 * @param marSeries Per-frame Mouth Aspect Ratio sequence for AVFF synchrony correlation
 * @param rppgPhaseAnomaly Bilateral malar rPPG chromaticity phase desynchronization score in [0, 1]
 */
data class BioLipResult(
    val landmarkJitter: Float,
    val hasFace: Boolean,
    val notes: String?,
    val marSeries: FloatArray = FloatArray(0),
    val rppgPhaseAnomaly: Float = Float.NaN,
)
