package com.onndetect

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.onndetect.detection.visual.LlamaCppVlmPrefillBackend
import com.onndetect.detection.visual.OnDemandForensicVlmAnalyzer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * On-device instrumentation test for [LlamaCppVlmPrefillBackend] verifying real
 * ART JNI execution through `libllama_forensic_jni.so` packaged in the APK.
 *
 * Verifies that the native loader strictly rejects dummy/invalid GGUFs with explicit
 * exceptions and never falls back to synthetic or fabricated vectors.
 */
@RunWith(AndroidJUnit4::class)
class LlamaCppNativeInstrumentationTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun nativeSharedLibraryIsLoadedInApk() {
        assertTrue(
            "libllama_forensic_jni.so must be packaged and loaded in Android runtime",
            LlamaCppVlmPrefillBackend.isNativeLibraryLoaded()
        )
    }

    @Test
    fun dummyGgufModelStrictlyFailsAndNeverEmitsSyntheticVectors() {
        // 1. First assert native library is loaded in process memory
        assertTrue(
            "Native shared library must be loaded before running model loader test",
            LlamaCppVlmPrefillBackend.isNativeLibraryLoaded()
        )

        // Ensure testMockDelegate is null: execution must route strictly to native C++ JNI
        LlamaCppVlmPrefillBackend.testMockDelegate = null

        val dummyModel = File(context.cacheDir, "corrupted_dummy_test.gguf").apply {
            writeBytes(ByteArray(1024) { 0x42 })
            deleteOnExit()
        }

        val backend = LlamaCppVlmPrefillBackend(
            modelPath = dummyModel.absolutePath,
            hiddenDim = 1024,
            imageResolutionCapPx = 256,
        )

        try {
            // A 1 KB dummy model must fail parsing in native llama_model_load_from_file
            backend.loadNativeContext()
            fail("Expected native llama loader to throw IllegalStateException for 1KB dummy model")
        } catch (e: IllegalStateException) {
            // Verify failure originates from genuine GGUF parser rejecting invalid binary
            assertTrue(
                "Error message must indicate genuine GGUF weight parsing failure, not missing library: ${e.message}",
                e.message!!.contains("Failed to parse GGUF model weights")
            )
        } finally {
            backend.close()
            dummyModel.delete()
        }
    }

    @Test
    fun pixelPreprocessingValidatesDimensionsOnDeviceBitmap() {
        val backend = LlamaCppVlmPrefillBackend(
            modelPath = File(context.cacheDir, "unused.gguf").apply {
                writeBytes(ByteArray(64))
                deleteOnExit()
            }.absolutePath,
            hiddenDim = 1024,
        )

        val bitmap = Bitmap.createBitmap(128, 64, Bitmap.Config.ARGB_8888)
        try {
            for (y in 0 until 64) {
                for (x in 0 until 128) {
                    bitmap.setPixel(x, y, Color.rgb(x * 2, y * 4, 128))
                }
            }

            val (rgb, targetW, targetH) = backend.bitmapToRgbBytes(bitmap, capPx = 64)
            assertEquals("Width must be capped proportionally", 64, targetW)
            assertEquals("Height must be capped proportionally", 32, targetH)
            assertEquals("Packed RGB size must match 64 * 32 * 3", 64 * 32 * 3, rgb.size)
        } finally {
            bitmap.recycle()
            backend.close()
        }
    }

    /**
     * Positive inference verification using real weights when deployed on a physical device.
     * Searches common device paths or system properties for genuine language model and mmproj weights.
     * When present, executes end-to-end multimodal prefill and asserts representation properties.
     */
    @Test
    fun positiveModelInferenceWhenRealWeightsPresent() {
        assertTrue(
            "Native shared library must be loaded for positive inference",
            LlamaCppVlmPrefillBackend.isNativeLibraryLoaded()
        )
        LlamaCppVlmPrefillBackend.testMockDelegate = null

        val candidatePaths = listOf(
            System.getProperty("vlm.test.model.path"),
            "/data/local/tmp/vlm_model.gguf",
            File(context.filesDir, "vlm_model.gguf").absolutePath,
            File(context.getExternalFilesDir(null), "vlm_model.gguf")?.absolutePath
        ).filterNotNull()

        val candidateMmprojPaths = listOf(
            System.getProperty("vlm.test.mmproj.path"),
            "/data/local/tmp/vlm_mmproj.gguf",
            File(context.filesDir, "vlm_mmproj.gguf").absolutePath,
            File(context.getExternalFilesDir(null), "vlm_mmproj.gguf")?.absolutePath
        ).filterNotNull()

        val modelPath = candidatePaths.firstOrNull { File(it).exists() && File(it).length() > 10 * 1024 * 1024 }
        val mmprojPath = candidateMmprojPaths.firstOrNull { File(it).exists() && File(it).length() > 10 * 1024 * 1024 }

        if (modelPath != null) {
            val hiddenDim = 1024
            val startTime = System.currentTimeMillis()
            val backend = LlamaCppVlmPrefillBackend(
                modelPath = modelPath,
                mmprojPath = mmprojPath,
                hiddenDim = hiddenDim,
                imageResolutionCapPx = 256
            )

            try {
                backend.loadNativeContext()
                val bitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)
                try {
                    for (y in 0 until 256) {
                        for (x in 0 until 256) {
                            bitmap.setPixel(x, y, Color.rgb(x % 256, y % 256, (x + y) % 256))
                        }
                    }

                    val representation = backend.extractFinalHiddenState(
                        fullBitmap = bitmap,
                        roiCropBitmap = null,
                        cropSpec = null,
                        activeDomain = "general"
                    )

                    val durationMs = System.currentTimeMillis() - startTime
                    assertNotNull("Representation must not be null", representation)
                    assertEquals("Representation dimension must match hiddenDim", hiddenDim, representation.size)

                    var nonZeroCount = 0
                    for (v in representation) {
                        assertTrue("Embedding value must be finite", !v.isNaN() && !v.isInfinite())
                        if (kotlin.math.abs(v) > 1e-6f) nonZeroCount++
                    }
                    assertTrue("Genuine embedding must contain non-zero activations", nonZeroCount > 0)
                    android.util.Log.i(
                        "LlamaCppNativeInstrumentationTest",
                        "Positive inference passed with genuine weights in ${durationMs}ms (nonZeroCount=$nonZeroCount/$hiddenDim)"
                    )
                } finally {
                    bitmap.recycle()
                }
            } finally {
                backend.close()
            }
        } else {
            android.util.Log.w(
                "LlamaCppNativeInstrumentationTest",
                "Skipping positive inference execution: no genuine model weights (>10MB) detected on device. " +
                "Push weights to /data/local/tmp/vlm_model.gguf to execute positive on-device test."
            )
            // Assert that non-existent paths are strictly rejected by the backend constructor
            try {
                LlamaCppVlmPrefillBackend(
                    modelPath = "/data/local/tmp/non_existent_weights.gguf",
                    hiddenDim = 1024
                )
                fail("Expected IllegalArgumentException for missing model file")
            } catch (e: IllegalArgumentException) {
                assertTrue(e.message!!.contains("Model file does not exist"))
            }
        }
    }

    @Test(timeout = 5000)
    fun dimensionMismatchFailsWithBoundedCompletionTime() {
        assertTrue(LlamaCppVlmPrefillBackend.isNativeLibraryLoaded())
        val dummyModel = File(context.cacheDir, "invalid_header.gguf").apply {
            writeBytes("GGUF_INVALID_HEADER_DATA".toByteArray())
            deleteOnExit()
        }

        val backend = LlamaCppVlmPrefillBackend(
            modelPath = dummyModel.absolutePath,
            hiddenDim = 2048
        )
        try {
            backend.loadNativeContext()
            fail("Expected load failure on invalid GGUF header")
        } catch (e: IllegalStateException) {
            assertTrue(e.message!!.contains("Failed to parse GGUF model weights"))
        } finally {
            backend.close()
            dummyModel.delete()
        }
    }
}
