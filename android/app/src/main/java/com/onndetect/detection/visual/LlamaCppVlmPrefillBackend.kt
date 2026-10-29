package com.onndetect.detection.visual

import android.graphics.Bitmap
import androidx.annotation.Keep
import androidx.annotation.VisibleForTesting
import java.io.File
import java.nio.ByteBuffer
import java.nio.CharBuffer
import java.nio.charset.CodingErrorAction
import java.util.concurrent.CancellationException

/**
 * Concrete on-device VLM prefill & RAG generation backend powered by `llama.cpp`'s native
 * multimodal (`mtmd`) JNI bindings (`libllama.so` / `libllama_forensic_jni.so`).
 *
 * Implements [OnDemandForensicVlmAnalyzer.VlmPrefillBackend] by:
 * 1. Preprocessing input [Bitmap]s into raw RGB pixel buffers bounded to resolution caps.
 * 2. Invoking native prefill through JNI to compute multimodal visual + text token context.
 * 3. Extracting the final token representation `h_last = llama_get_embeddings_ith(ctx, -1)`.
 * 4. Supporting bounded autoregressive multimodal RAG text generation (`generateMultimodalResponse`)
 *    for `Qwen/Qwen3.5-0.8B` and `SmolVLM2-500M-Video-Instruct`.
 *
 * Includes safe test delegate hooks and fallback handling for JVM unit test execution
 * where native shared libraries are not loaded.
 */
class LlamaCppVlmPrefillBackend(
    val modelPath: String,
    val mmprojPath: String? = null,
    val hiddenDim: Int = 1024,
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

        /**
         * Test-only delegate hook permitting JVM unit tests to verify multimodal RAG
         * generation without compiled native ARM64 `.so` libraries.
         */
        @VisibleForTesting
        var testGenerationDelegate: ((
            prompt: String,
            bitmap: Bitmap?,
            maxNewTokens: Int
        ) -> String)? = null
    }

    @Volatile private var nativeContextHandle: Long = 0L

    init {
        val modelFile = File(modelPath)
        require(modelFile.exists() || testMockDelegate != null || testGenerationDelegate != null) {
            "Model file does not exist at path: $modelPath"
        }
        if (mmprojPath != null) {
            val mmprojFile = File(mmprojPath)
            require(mmprojFile.exists() || testMockDelegate != null || testGenerationDelegate != null) {
                "Vision mmproj file does not exist at path: $mmprojPath"
            }
        }
        // Context is NOT allocated in init: model weights remain unmapped until explicit
        // lease acquisition (loadNativeContext), preventing memory coexistence with SigLIP.
    }

    val isContextInitialized: Boolean
        get() = nativeContextHandle != 0L || testMockDelegate != null || testGenerationDelegate != null

    /**
     * Explicitly loads the native model context inside an active exclusive memory lease.
     */
    @Synchronized
    fun loadNativeContext(): Long {
        if (testMockDelegate != null || testGenerationDelegate != null) return 1L
        check(libraryLoaded) {
            "Native shared library '$JNI_LIB_NAME' is not available. Ensure 'lib$JNI_LIB_NAME.so' is packaged in APK."
        }
        if (nativeContextHandle == 0L) {
            nativeContextHandle = nativeInitContext(modelPath, mmprojPath ?: "", hiddenDim)
            check(nativeContextHandle != 0L) { "Failed to allocate native llama context for model: $modelPath" }
        }
        return nativeContextHandle
    }

    /**
     * Explicitly unloads the native model context, freeing all accelerator memory,
     * KV buffers, and memory-mapped weight handles before the lease releases.
     */
    @Synchronized
    fun unloadNativeContext() {
        if (nativeContextHandle != 0L) {
            nativeFreeContext(nativeContextHandle)
            nativeContextHandle = 0L
        }
    }

    /**
     * Scopes native model residency strictly to [block], ensuring complete deallocation.
     */
    inline fun <R> withLeasedContext(block: () -> R): R {
        loadNativeContext()
        try {
            return block()
        } finally {
            unloadNativeContext()
        }
    }

    @Synchronized
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
        if (nativeContextHandle == 0L) {
            loadNativeContext()
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

    /**
     * Generates a bounded multimodal or text RAG response using the loaded GGUF model
     * (`Qwen/Qwen3.5-0.8B` or `SmolVLM2-500M-Video-Instruct`).
     */
    @Synchronized
    fun generateMultimodalResponse(
        prompt: String,
        bitmap: Bitmap? = null,
        maxNewTokens: Int = 160,
    ): String {
        return generateMultimodalResponseStreaming(prompt, bitmap, maxNewTokens, onToken = {}, isCancelled = { false })
    }

    /** Streams native token bytes as they are sampled, including split UTF-8 characters. */
    @Synchronized
    fun generateMultimodalResponseStreaming(
        prompt: String,
        bitmap: Bitmap? = null,
        maxNewTokens: Int = 160,
        onToken: (String) -> Unit,
        isCancelled: () -> Boolean,
    ): String {
        require(prompt.isNotBlank()) { "RAG prompt cannot be blank" }
        val boundedTokens = maxNewTokens.coerceIn(8, 512)
        val receiver = NativeTokenReceiver(onToken, isCancelled)
        receiver.checkActive()

        testGenerationDelegate?.let { mock ->
            // The delegate is a completed-answer test seam, not a substitute native stream.
            receiver.onTokenBytes(mock(prompt, bitmap, boundedTokens).toByteArray(Charsets.UTF_8))
            return receiver.finish()
        }

        check(libraryLoaded) {
            "Native shared library '$JNI_LIB_NAME' is not available."
        }
        if (nativeContextHandle == 0L) {
            loadNativeContext()
        }
        check(nativeContextHandle != 0L) {
            "Native llama context handle is null/closed."
        }

        val (imgBytes, imgW, imgH) = if (bitmap != null) {
            bitmapToRgbBytes(bitmap, imageResolutionCapPx)
        } else {
            Triple(null, 0, 0)
        }

        receiver.checkActive()
        nativeGenerateMultimodalResponseStreaming(
            contextHandle = nativeContextHandle,
            prompt = prompt,
            imageRgb = imgBytes,
            imageWidth = imgW,
            imageHeight = imgH,
            maxNewTokens = boundedTokens,
            receiver = receiver,
        )
        return receiver.finish()
    }

    @Synchronized
    override fun close() {
        unloadNativeContext()
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
    private external fun nativeGenerateMultimodalResponseStreaming(
        contextHandle: Long,
        prompt: String,
        imageRgb: ByteArray?,
        imageWidth: Int,
        imageHeight: Int,
        maxNewTokens: Int,
        receiver: NativeTokenReceiver,
    )
    private external fun nativeFreeContext(contextHandle: Long)

    @Keep
    private class NativeTokenReceiver(
        private val onToken: (String) -> Unit,
        private val cancelled: () -> Boolean,
    ) {
        private val decoder = Utf8TokenDecoder()
        private val output = StringBuilder()

        // llama.cpp can poll this from an attached CPU worker thread.
        fun shouldCancel(): Boolean = Thread.currentThread().isInterrupted || runCatching(cancelled).getOrDefault(true)

        fun checkActive() {
            if (shouldCancel()) throw CancellationException("Response stopped")
        }

        fun onTokenBytes(bytes: ByteArray): Boolean {
            checkActive()
            emit(decoder.append(bytes))
            return !shouldCancel()
        }

        fun finish(): String {
            checkActive()
            emit(decoder.finish())
            return output.toString()
        }

        private fun emit(text: String) {
            if (text.isNotEmpty()) {
                output.append(text)
                onToken(text)
            }
        }
    }

    /** A tokenizer may split an emoji or non-Latin character across several byte tokens. */
    internal class Utf8TokenDecoder {
        private val decoder = Charsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPLACE)
            .onUnmappableCharacter(CodingErrorAction.REPLACE)
        private var pending = ByteArray(0)

        fun append(bytes: ByteArray): String = decode(bytes, endOfInput = false)
        fun finish(): String = decode(ByteArray(0), endOfInput = true)

        private fun decode(bytes: ByteArray, endOfInput: Boolean): String {
            val input = ByteBuffer.wrap(pending + bytes)
            val output = CharBuffer.allocate(input.remaining() + 2)
            decoder.decode(input, output, endOfInput).throwExceptionIfError()
            pending = ByteArray(input.remaining()).also { input.get(it) }
            if (endOfInput) decoder.flush(output).throwExceptionIfError()
            output.flip()
            return output.toString()
        }

        private fun java.nio.charset.CoderResult.throwExceptionIfError() {
            if (isError) throwException()
        }
    }

    // =========================================================================
    // Pixel Preprocessing Utilities
    // =========================================================================
    internal fun pixelsToRgbBytes(
        pixels: IntArray,
        width: Int,
        height: Int,
        capPx: Int,
    ): Triple<ByteArray, Int, Int> {
        require(width > 0 && height > 0) { "width and height must be positive: width=$width, height=$height" }
        require(pixels.size == width * height) {
            "pixels.size (${pixels.size}) must match width * height ($width * $height = ${width * height})"
        }
        val maxDim = maxOf(width, height)
        val (targetW, targetH) = if (maxDim > capPx) {
            val scale = capPx.toFloat() / maxDim
            Pair((width * scale).toInt().coerceAtLeast(1), (height * scale).toInt().coerceAtLeast(1))
        } else {
            Pair(width, height)
        }

        val rgb = ByteArray(targetW * targetH * 3)
        var outIdx = 0
        if (targetW == width && targetH == height) {
            for (p in pixels) {
                rgb[outIdx++] = ((p shr 16) and 0xFF).toByte() // R
                rgb[outIdx++] = ((p shr 8) and 0xFF).toByte()  // G
                rgb[outIdx++] = (p and 0xFF).toByte()         // B
            }
        } else {
            val xRatio = width.toFloat() / targetW
            val yRatio = height.toFloat() / targetH
            for (y in 0 until targetH) {
                val srcY = (y * yRatio).toInt().coerceIn(0, height - 1)
                val srcRowOffset = srcY * width
                for (x in 0 until targetW) {
                    val srcX = (x * xRatio).toInt().coerceIn(0, width - 1)
                    val p = pixels[srcRowOffset + srcX]
                    rgb[outIdx++] = ((p shr 16) and 0xFF).toByte() // R
                    rgb[outIdx++] = ((p shr 8) and 0xFF).toByte()  // G
                    rgb[outIdx++] = (p and 0xFF).toByte()         // B
                }
            }
        }
        return Triple(rgb, targetW, targetH)
    }

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

        if (scaled !== bitmap) {
            scaled.recycle()
        }
        return pixelsToRgbBytes(pixels, targetW, targetH, capPx)
    }
}
