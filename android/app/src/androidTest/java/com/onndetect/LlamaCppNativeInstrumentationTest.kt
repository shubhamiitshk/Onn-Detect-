package com.onndetect

import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.onndetect.detection.visual.LlamaCppVlmPrefillBackend
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import kotlin.math.sqrt

/**
 * On-device instrumentation test for [LlamaCppVlmPrefillBackend] verifying real
 * ART JNI execution through `libllama_forensic_jni.so` (upstream llama.cpp b8291 + libmtmd).
 *
 * Verifies:
 * 1. Native shared library packaging and loading.
 * 2. Corrupted binary detection in genuine GGUF parser.
 * 3. Exact aspect-ratio preserving RGB preprocessing.
 * 4. Dedicated real-model test requiring BOTH language weights and vision projector.
 *    - Unconfigured environments skip via [Assume.assumeTrue].
 *    - Valid model with wrong dimension strictly throws dimension contract exception.
 *    - Multimodal prefill on distinct images produces image-dependent representations.
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
    fun dummyGgufModelStrictlyFailsParsing() {
        assertTrue(
            "Native shared library must be loaded before running model loader test",
            LlamaCppVlmPrefillBackend.isNativeLibraryLoaded()
        )
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
            backend.loadNativeContext()
            fail("Expected native llama loader to throw IllegalStateException for corrupted dummy model")
        } catch (e: IllegalStateException) {
            assertTrue(
                "Error message must indicate genuine GGUF weight parsing failure: ${e.message}",
                e.message!!.contains("Failed to parse GGUF model weights")
            )
        } finally {
            backend.close()
            dummyModel.delete()
        }
    }

    @Test
    fun pixelPreprocessingValidatesDimensionsOnDeviceBitmap() {
        val dummyFile = File(context.cacheDir, "unused_pixel_test.gguf").apply {
            writeBytes(ByteArray(64))
            deleteOnExit()
        }
        val backend = LlamaCppVlmPrefillBackend(
            modelPath = dummyFile.absolutePath,
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
            dummyFile.delete()
        }
    }

    /**
     * Dedicated genuine model test requiring BOTH language weights AND multimodal projector.
     * When hardware/weights are not present, test is marked as SKIPPED via [Assume.assumeTrue].
     */
    @Test(timeout = 60000)
    fun dedicatedRealModelInferenceRequiresBothWeightsAndProjector() {
        assertTrue(
            "Native shared library must be loaded for real model inference",
            LlamaCppVlmPrefillBackend.isNativeLibraryLoaded()
        )
        LlamaCppVlmPrefillBackend.testMockDelegate = null

        val candidateLangPaths = listOf(
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

        val langFile = candidateLangPaths.map { File(it) }.firstOrNull { it.exists() && it.length() > 10 * 1024 * 1024 }
        val mmprojFile = candidateMmprojPaths.map { File(it) }.firstOrNull { it.exists() && it.length() > 5 * 1024 * 1024 }

        // Ordinary CI without deployed weights will explicitly mark test as SKIPPED
        assumeTrue("Genuine language weights (>10MB) must be present on device", langFile != null)
        assumeTrue("Genuine vision mmproj projector (>5MB) must be present on device", mmprojFile != null)

        val modelPath = langFile!!.absolutePath
        val mmprojPath = mmprojFile!!.absolutePath

        // ---------------------------------------------------------------------
        // 1. Test Valid Model with Mismatched Hidden Dimension
        // ---------------------------------------------------------------------
        val wrongDimensionBackend = LlamaCppVlmPrefillBackend(
            modelPath = modelPath,
            mmprojPath = mmprojPath,
            hiddenDim = 99999, // Intentional mismatch against model architecture
        )
        try {
            wrongDimensionBackend.loadNativeContext()
            fail("Expected loadNativeContext() to reject mismatched hidden dimension on valid model")
        } catch (e: IllegalArgumentException) {
            assertTrue(
                "Error message must indicate hidden dimension mismatch: ${e.message}",
                e.message!!.contains("does not match policy head contract hiddenDim")
            )
        } finally {
            wrongDimensionBackend.close()
        }

        // ---------------------------------------------------------------------
        // 2. Multimodal Image Inference & Representation Parity Check
        // ---------------------------------------------------------------------
        val targetHiddenDim = 1024 // Architecture contract (e.g. Qwen3.5-0.8B)
        val backend = LlamaCppVlmPrefillBackend(
            modelPath = modelPath,
            mmprojPath = mmprojPath,
            hiddenDim = targetHiddenDim,
            imageResolutionCapPx = 256,
        )

        val bitmapA = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        val bitmapB = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)

        try {
            for (y in 0 until 128) {
                for (x in 0 until 128) {
                    bitmapA.setPixel(x, y, Color.rgb(x * 2, 0, 0)) // Red horizontal gradient
                    bitmapB.setPixel(x, y, Color.rgb(0, 0, y * 2)) // Blue vertical gradient
                }
            }

            backend.loadNativeContext()

            val hA = backend.extractFinalHiddenState(
                fullBitmap = bitmapA,
                roiCropBitmap = null,
                cropSpec = null,
                activeDomain = "general",
            )
            assertNotNull("Representation for Image A must be non-null", hA)
            assertEquals("Representation dimension must match contract", targetHiddenDim, hA.size)
            assertTrue("Embeddings must contain finite activations", hA.all { it.isFinite() })
            assertTrue("Embeddings must contain non-zero activations", hA.any { kotlin.math.abs(it) > 1e-6f })

            val hB = backend.extractFinalHiddenState(
                fullBitmap = bitmapB,
                roiCropBitmap = null,
                cropSpec = null,
                activeDomain = "general",
            )
            assertNotNull("Representation for Image B must be non-null", hB)
            assertEquals("Representation dimension must match contract", targetHiddenDim, hB.size)
            assertTrue("Embeddings must contain finite activations", hB.all { it.isFinite() })

            // Compute cosine similarity between image representations
            var dot = 0.0
            var normA = 0.0
            var normB = 0.0
            for (i in 0 until targetHiddenDim) {
                dot += hA[i] * hB[i]
                normA += hA[i] * hA[i]
                normB += hB[i] * hB[i]
            }
            val cosSim = (dot / (sqrt(normA) * sqrt(normB))).toFloat()

            // Different visual inputs must produce distinct representations
            assertTrue(
                "Distinct images must produce distinct visual embeddings (cosine similarity $cosSim must be < 0.999)",
                cosSim < 0.999f
            )
        } finally {
            bitmapA.recycle()
            bitmapB.recycle()
            backend.close()
        }
    }
}
