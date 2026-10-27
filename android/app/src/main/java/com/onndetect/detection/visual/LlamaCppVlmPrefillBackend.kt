package com.onndetect.detection.visual

import android.graphics.Bitmap
import androidx.annotation.VisibleForTesting
import java.io.File

/**
 * Concrete on-device VLM prefill backend powered by `llama.cpp`'s native multimodal (`mtmd`)
 * JNI bindings (`libllama.so` / `libllama_forensic_jni.so`).
 *
 * Implements [OnDemandForensicVlmAnalyzer.VlmPrefillBackend] by:
 * 1. Preprocessing input [Bitmap]s into raw RGB pixel buffers bounded to resolution caps.
 * 2. Invoking native prefill through JNI to compute multimodal visual + text token context.
 * 3. Extracting the final token representation `h_last = llama_get_embeddings_ith(ctx, -1)`.
 *
 * Includes safe test delegate hooks and fallback handling for JVM unit test execution
 * where native shared libraries are not loaded.
 */
class LlamaCppVlmPrefillBackend(
    val modelPath: String,
    val mmprojPath: String? = null,
    val hiddenDim: Int = 1536,
    val imageResolutionCapPx: Int = 512,
) : OnDemandForensicVlmAnalyzer.VlmPrefillBackend, AutoCloseable {

    companion object {
        const val JNI_LIB_NAME = "llama_forensic_jni"

        @Volatile
        private var libraryLoaded: Boolean = false

        init {
            libraryLoaded = try {
                System.loadLibrary(JNI_LIB_NAME)
                true
            } catch (_: Throwable) {
                false
            }
        }

        fun isNativeLibraryLoaded(): Boolean = libraryLoaded

        /**
         * Test-only delegate hook permitting JVM unit tests to inject deterministic
         * `hLast` vectors without needing compiled native ARM64 `.so` libraries.
         */
        @VisibleForTesting
        var testMockDelegate: ((
            fullBitmap: Bitmap,
            roiCropBitmap: Bitmap?,
            cropSpec: OnDemandForensicVlmAnalyzer.OriginalSourceCropSpec?,
            activeDomain: String
        ) -> FloatArray)? = null
    }

    private var nativeContextHandle: Long = 0L

    init {
        val modelFile = File(modelPath)
        require(modelFile.exists() || testMockDelegate != null) {
            "Model file does not exist at path: $modelPath"
        }
        if (mmprojPath != null) {
            val mmprojFile = File(mmprojPath)
            require(mmprojFile.exists() || testMockDelegate != null) {
                "Vision mmproj file does not exist at path: $mmprojPath"
            }
        }
        if (libraryLoaded && testMockDelegate == null) {
            nativeContextHandle = nativeInitContext(modelPath, mmprojPath ?: "", hiddenDim)
        }
    }

    val isContextInitialized: Boolean
        get() = nativeContextHandle != 0L || testMockDelegate != null

    override fun extractFinalHiddenState(
        fullBitmap: Bitmap,
        roiCropBitmap: Bitmap?,
        cropSpec: OnDemandForensicVlmAnalyzer.OriginalSourceCropSpec?,
        activeDomain: String,
    ): FloatArray {
        // 1. Check test delegate first for hermetic JVM testing
        testMockDelegate?.let { mock ->
            return mock(fullBitmap, roiCropBitmap, cropSpec, activeDomain)
        }

        check(libraryLoaded) {
            "Native shared library '$JNI_LIB_NAME' is not available. Ensure 'lib$JNI_LIB_NAME.so' is packaged in jniLibs."
        }
        check(nativeContextHandle != 0L) {
            "Native llama context handle is null/closed."
        }

        // 2. Preprocess fullBitmap into packed RGB bytes
        val (fullBytes, fullW, fullH) = bitmapToRgbBytes(fullBitmap, imageResolutionCapPx)

        // 3. Preprocess optional roiCropBitmap
        val (roiBytes, roiW, roiH) = if (roiCropBitmap != null) {
            bitmapToRgbBytes(roiCropBitmap, imageResolutionCapPx)
        } else {
            Triple(null, 0, 0)
        }

        // 4. Invoke native JNI prefill
        val hLast = nativeExtractFinalHiddenState(
            contextHandle = nativeContextHandle,
            fullRgb = fullBytes,
            fullWidth = fullW,
            fullHeight = fullH,
            roiRgb = roiBytes,
            roiWidth = roiW,
            roiHeight = roiH,
            domain = activeDomain,
        )

        check(hLast.size == hiddenDim) {
            "Native prefill returned ${hLast.size} dimensions, expected hiddenDim=$hiddenDim"
        }
        return hLast
    }

    override fun close() {
        if (nativeContextHandle != 0L) {
            nativeFreeContext(nativeContextHandle)
            nativeContextHandle = 0L
        }
    }

    // =========================================================================
    // JNI Native Declarations
    // =========================================================================
    private external fun nativeInitContext(modelPath: String, mmprojPath: String, hiddenDim: Int): Long
    private external fun nativeExtractFinalHiddenState(
        contextHandle: Long,
        fullRgb: ByteArray,
        fullWidth: Int,
        fullHeight: Int,
        roiRgb: ByteArray?,
        roiWidth: Int,
        roiHeight: Int,
        domain: String,
    ): FloatArray
    private external fun nativeFreeContext(contextHandle: Long)

    // =========================================================================
    // Pixel Preprocessing Utilities
    // =========================================================================
    internal fun bitmapToRgbBytes(bitmap: Bitmap, capPx: Int): Triple<ByteArray, Int, Int> {
        val w = bitmap.width
        val h = bitmap.height
        val maxDim = maxOf(w, h)
        val (targetW, targetH) = if (maxDim > capPx) {
            val scale = capPx.toFloat() / maxDim
            Pair((w * scale).toInt().coerceAtLeast(1), (h * scale).toInt().coerceAtLeast(1))
        } else {
            Pair(w, h)
        }

        val scaled = if (targetW != w || targetH != h) {
            Bitmap.createScaledBitmap(bitmap, targetW, targetH, true)
        } else {
            bitmap
        }

        val pixels = IntArray(targetW * targetH)
        scaled.getPixels(pixels, 0, targetW, 0, 0, targetW, targetH)

        val rgb = ByteArray(targetW * targetH * 3)
        var outIdx = 0
        for (p in pixels) {
            rgb[outIdx++] = ((p shr 16) and 0xFF).toByte() // R
            rgb[outIdx++] = ((p shr 8) and 0xFF).toByte()  // G
            rgb[outIdx++] = (p and 0xFF).toByte()         // B
        }

        if (scaled !== bitmap) {
            scaled.recycle()
        }
        return Triple(rgb, targetW, targetH)
    }
}
