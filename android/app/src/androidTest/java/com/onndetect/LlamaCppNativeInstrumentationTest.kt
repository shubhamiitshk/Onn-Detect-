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
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * On-device instrumentation test for [LlamaCppVlmPrefillBackend] verifying real
 * ART JNI execution through `libllama_forensic_jni.so` packaged in the APK.
 */
@RunWith(AndroidJUnit4::class)
class LlamaCppNativeInstrumentationTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun nativeSharedLibraryIsLoadedAndExecutesJniHiddenStateExtraction() {
        // Ensure testMockDelegate is null so execution routes strictly through native C++ JNI
        LlamaCppVlmPrefillBackend.testMockDelegate = null
        assertTrue(
            "libllama_forensic_jni.so must be loaded in Android runtime",
            LlamaCppVlmPrefillBackend.isNativeLibraryLoaded()
        )

        val modelFile = File(context.cacheDir, "test_instrumentation_model.gguf").apply {
            writeBytes(ByteArray(1024))
            deleteOnExit()
        }

        val backend = LlamaCppVlmPrefillBackend(
            modelPath = modelFile.absolutePath,
            hiddenDim = 1536,
            imageResolutionCapPx = 256,
        )

        val bitmap = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        try {
            for (y in 0 until 128) {
                for (x in 0 until 128) {
                    bitmap.setPixel(x, y, Color.rgb(x * 2, y * 2, (x + y) % 256))
                }
            }

            // Execute through JNI within leased context
            val hLast = backend.withLeasedContext {
                backend.extractFinalHiddenState(
                    fullBitmap = bitmap,
                    roiCropBitmap = null,
                    cropSpec = null,
                    activeDomain = "human_face",
                )
            }

            assertNotNull("hLast vector must not be null", hLast)
            assertEquals("hLast dimensionality must match hiddenDim", 1536, hLast.size)

            var nonZeroCount = 0
            for (v in hLast) {
                assertTrue("hLast vector element must be finite: $v", v.isFinite() && !v.isNaN())
                if (kotlin.math.abs(v) > 1e-6f) {
                    nonZeroCount++
                }
            }
            assertTrue("hLast vector must have non-zero signal content", nonZeroCount > 100)
        } finally {
            bitmap.recycle()
            backend.close()
            modelFile.delete()
        }
    }

    @Test
    fun nativeJniProcessesRoiCropBitmapWhenSupplied() {
        LlamaCppVlmPrefillBackend.testMockDelegate = null
        assertTrue(LlamaCppVlmPrefillBackend.isNativeLibraryLoaded())

        val modelFile = File(context.cacheDir, "test_roi_model.gguf").apply {
            writeBytes(ByteArray(1024))
            deleteOnExit()
        }

        val backend = LlamaCppVlmPrefillBackend(
            modelPath = modelFile.absolutePath,
            hiddenDim = 1536,
            imageResolutionCapPx = 128,
        )

        val fullBmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
        val roiBmp = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888)
        try {
            val cropSpec = OnDemandForensicVlmAnalyzer.OriginalSourceCropSpec(
                left = 16,
                top = 16,
                width = 32,
                height = 32,
                targetEncodedEdgePx = 32,
                downscaleRatio = 1.0f,
            )

            val hLast = backend.withLeasedContext {
                backend.extractFinalHiddenState(
                    fullBitmap = fullBmp,
                    roiCropBitmap = roiBmp,
                    cropSpec = cropSpec,
                    activeDomain = "document_text",
                )
            }

            assertEquals(1536, hLast.size)
            assertTrue(hLast.all { it.isFinite() && !it.isNaN() })
        } finally {
            fullBmp.recycle()
            roiBmp.recycle()
            backend.close()
            modelFile.delete()
        }
    }
}
