package com.onndetect

import com.onndetect.detection.visual.LlamaCppVlmPrefillBackend
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicBoolean

/** Optional integration gate: uses actual deployed weights, never fabricated model output. */
class NativeStreamingInstrumentationTest {
    private fun backend(): LlamaCppVlmPrefillBackend {
        val model = File("/data/local/tmp/vlm_model.gguf")
        assumeTrue("A real GGUF model must be deployed to run native streaming", model.isFile && model.length() > 50_000_000)
        assertTrue(LlamaCppVlmPrefillBackend.isNativeLibraryLoaded())
        return LlamaCppVlmPrefillBackend(model.absolutePath, hiddenDim = 1024)
    }

    @Test(timeout = 120_000) fun actualGgufEmitsTokensBeforeReturningFinalResponse() {
        backend().use { backend ->
            val chunks = mutableListOf<String>()
            val answer = backend.generateMultimodalResponseStreaming(
                prompt = "Say hello in one short sentence.", maxNewTokens = 24,
                onToken = { chunks.add(it) }, isCancelled = { false },
            )
            assertTrue("Real model must emit a response", answer.isNotBlank())
            assertTrue("Response must arrive incrementally", chunks.size > 1)
            assertEquals(answer, chunks.joinToString(""))
            assertFalse("Streaming UTF-8 decoding must preserve characters", answer.contains('\uFFFD'))
        }
    }

    @Test(timeout = 120_000) fun nativeGenerationStopsAfterCancellationDuringTokenDelivery() {
        backend().use { backend ->
            val cancelled = AtomicBoolean(false)
            var received = false
            var stopped = false
            try {
                backend.generateMultimodalResponseStreaming(
                    prompt = "Describe how to inspect a photograph in several paragraphs.", maxNewTokens = 128,
                    onToken = { received = true; cancelled.set(true) }, isCancelled = { cancelled.get() },
                )
            } catch (_: CancellationException) { stopped = true }
            assertTrue("Cancellation test must reach genuine inference", received)
            assertTrue("Native generation must honor cancellation", stopped)
        }
    }
}
