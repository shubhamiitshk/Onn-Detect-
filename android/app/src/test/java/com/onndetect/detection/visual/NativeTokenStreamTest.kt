package com.onndetect.detection.visual

import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.CancellationException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

class NativeTokenStreamTest {
    @After
    fun resetDelegates() {
        LlamaCppVlmPrefillBackend.testGenerationDelegate = null
        LlamaCppVlmPrefillBackend.testMockDelegate = null
    }

    @Test
    fun splitEmojiAndNonLatinTokenBytesReassembleWithoutReplacement() {
        val expected = "Image 🔍 — नमस्ते 世界"
        val decoder = LlamaCppVlmPrefillBackend.Utf8TokenDecoder()
        val output = buildString {
            expected.toByteArray(Charsets.UTF_8).forEach { append(decoder.append(byteArrayOf(it))) }
            append(decoder.finish())
        }
        assertEquals(expected, output)
        assertFalse(output.contains('\uFFFD'))
    }

    @Test
    fun incompleteFinalCharacterIsReplacedOnlyWhenStreamEnds() {
        val decoder = LlamaCppVlmPrefillBackend.Utf8TokenDecoder()
        assertEquals("", decoder.append(byteArrayOf(0xF0.toByte(), 0x9F.toByte())))
        assertEquals("\uFFFD", decoder.finish())
    }

    @Test(expected = CancellationException::class)
    fun cancelledRequestDoesNotInvokeModelDelegate() {
        LlamaCppVlmPrefillBackend.testGenerationDelegate = { _, _, _ -> error("Cancelled request ran inference") }
        LlamaCppVlmPrefillBackend("virtual.gguf").generateMultimodalResponseStreaming(
            prompt = "Inspect the evidence", onToken = {}, isCancelled = { true },
        )
    }

    @Test
    fun streamingCallbackAndReturnedAnswerHaveIdenticalUnicodeContent() {
        val expected = "No reliable conclusion.\nTry a clearer image. 🔍"
        LlamaCppVlmPrefillBackend.testGenerationDelegate = { _, _, _ -> expected }
        val pieces = mutableListOf<String>()
        val answer = LlamaCppVlmPrefillBackend("virtual.gguf").generateMultimodalResponseStreaming(
            prompt = "Inspect the evidence", onToken = { pieces.add(it) }, isCancelled = { false },
        )
        assertEquals(expected, answer)
        assertEquals(answer, pieces.joinToString(""))
    }

    @Test
    fun closeWaitsForActiveGenerationBeforeReleasingContext() {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val closed = CountDownLatch(1)
        val failed = AtomicBoolean(false)
        LlamaCppVlmPrefillBackend.testGenerationDelegate = { _, _, _ ->
            entered.countDown()
            check(release.await(3, TimeUnit.SECONDS))
            "Answer"
        }
        val backend = LlamaCppVlmPrefillBackend("virtual.gguf")
        val generation = thread {
            try { backend.generateMultimodalResponseStreaming("Question", onToken = {}, isCancelled = { false }) }
            catch (_: Throwable) { failed.set(true) }
        }
        assertTrue(entered.await(3, TimeUnit.SECONDS))
        val closing = thread { backend.close(); closed.countDown() }
        try {
            assertFalse("close raced an active invocation", closed.await(80, TimeUnit.MILLISECONDS))
        } finally {
            release.countDown()
            generation.join(3_000)
            closing.join(3_000)
        }
        assertFalse(failed.get())
        assertEquals(0L, closed.count)
    }
}
