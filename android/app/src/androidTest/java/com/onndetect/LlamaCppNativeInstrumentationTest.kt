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
            // A 1 KB dummy model must fail parsing in native llama_load_model_from_file
            backend.loadNativeContext()
            fail("Expected native llama loader to throw IllegalStateException for 1KB dummy model")
        } catch (e: IllegalStateException) {
            assertTrue(
                "Error message must indicate native load or runtime failure: ${e.message}",
                e.message!!.contains("GGUF") || e.message!!.contains("libllama") || e.message!!.contains("context")
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
}
