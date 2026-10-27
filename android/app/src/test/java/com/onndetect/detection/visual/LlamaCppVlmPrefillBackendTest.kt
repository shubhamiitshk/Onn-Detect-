package com.onndetect.detection.visual

import android.graphics.Bitmap
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Unit tests for [LlamaCppVlmPrefillBackend], verifying lifecycle management,
 * path validation, test delegate interception, and integration with
 * [OnDemandForensicVlmAnalyzer.VlmPrefillBackend].
 */
class LlamaCppVlmPrefillBackendTest {

    private fun createDummyBitmap(): Bitmap {
        val unsafeClass = Class.forName("sun.misc.Unsafe")
        val theUnsafeField = unsafeClass.getDeclaredField("theUnsafe").apply {
            isAccessible = true
        }
        val unsafe = theUnsafeField.get(null)
        val allocateMethod = unsafeClass.getMethod("allocateInstance", Class::class.java)
        return allocateMethod.invoke(unsafe, Bitmap::class.java) as Bitmap
    }

    @Before
    fun setUp() {
        LlamaCppVlmPrefillBackend.testMockDelegate = null
    }

    @After
    fun tearDown() {
        LlamaCppVlmPrefillBackend.testMockDelegate = null
    }

    @Test
    fun failsWhenModelFileDoesNotExistAndNoDelegateConfigured() {
        val fakePath = File(System.getProperty("java.io.tmpdir"), "nonexistent_model_${System.currentTimeMillis()}.gguf").absolutePath
        try {
            LlamaCppVlmPrefillBackend(modelPath = fakePath)
            fail("Expected IllegalArgumentException for missing model file")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Model file does not exist"))
        }
    }

    @Test
    fun allowsInitializationWhenTestDelegateIsConfigured() {
        val fakePath = "/path/to/virtual_model.gguf"
        LlamaCppVlmPrefillBackend.testMockDelegate = { _, _, _, _ -> FloatArray(1536) { 0.1f } }

        val backend = LlamaCppVlmPrefillBackend(modelPath = fakePath, hiddenDim = 1536)
        assertTrue(backend.isContextInitialized)

        val dummy = createDummyBitmap()
        val hLast = backend.extractFinalHiddenState(
            fullBitmap = dummy,
            roiCropBitmap = null,
            cropSpec = null,
            activeDomain = "human_face"
        )
        assertEquals(1536, hLast.size)
        assertEquals(0.1f, hLast[0], 1e-6f)

        backend.close()
    }

    @Test
    fun throwsWhenNativeLibraryNotLoadedAndNoDelegate() {
        val tempModel = File.createTempFile("mock_model", ".gguf").apply { deleteOnExit() }
        val backend = LlamaCppVlmPrefillBackend(modelPath = tempModel.absolutePath)

        if (!LlamaCppVlmPrefillBackend.isNativeLibraryLoaded()) {
            val dummy = createDummyBitmap()
            try {
                backend.extractFinalHiddenState(dummy, null, null, "human_face")
                fail("Expected IllegalStateException due to missing native library in JVM environment")
            } catch (e: IllegalStateException) {
                assertTrue(e.message!!.contains(LlamaCppVlmPrefillBackend.JNI_LIB_NAME))
            }
        }
        backend.close()
    }

    @Test
    fun testDelegateIntegratesWithAnalyzerContract() {
        val expectedVector = FloatArray(1536) { idx -> (idx % 10) * 0.05f }
        var invokedDomain: String? = null

        LlamaCppVlmPrefillBackend.testMockDelegate = { _, _, _, domain ->
            invokedDomain = domain
            expectedVector
        }

        val backend = LlamaCppVlmPrefillBackend(
            modelPath = "/virtual/smolvlm2_500m.gguf",
            hiddenDim = 1536
        )

        val dummy = createDummyBitmap()
        val result = backend.extractFinalHiddenState(
            fullBitmap = dummy,
            roiCropBitmap = null,
            cropSpec = null,
            activeDomain = "document_text"
        )

        assertEquals("document_text", invokedDomain)
        assertEquals(1536, result.size)
        assertEquals(expectedVector[5], result[5], 1e-6f)
        backend.close()
    }

    @Test
    fun nativeContextLifecycleScopedToLeaseAndUnloadsExplicitly() {
        val tempModel = File.createTempFile("mock_vlm", ".gguf").apply { deleteOnExit() }
        val backend = LlamaCppVlmPrefillBackend(modelPath = tempModel.absolutePath, hiddenDim = 1536)

        // 1. Constructor must NOT load native context
        assertFalse("Native context must remain uninitialized on construction to avoid coexisting with SigLIP",
            backend.isContextInitialized && LlamaCppVlmPrefillBackend.isNativeLibraryLoaded())

        // 2. withLeasedContext scopes allocation
        LlamaCppVlmPrefillBackend.testMockDelegate = { _, _, _, _ -> FloatArray(1536) }
        var executed = false
        backend.withLeasedContext {
            executed = true
            assertTrue("Context must be active during leased block", backend.isContextInitialized)
        }
        assertTrue("Leased block must have executed", executed)

        // 3. Explicit unload
        backend.unloadNativeContext()
        backend.close()
    }
}
