package com.onndetect

import android.net.Uri
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.onndetect.ui.screens.scanner.ProviderMediaImporter
import com.onndetect.ui.screens.scanner.cancellableProviderWork
import kotlinx.coroutines.*
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.Closeable
import java.util.concurrent.atomic.AtomicBoolean

@RunWith(AndroidJUnit4::class)
class ProviderImportTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext
    private fun uri(path: String) = Uri.parse("content://com.onndetect.test.delayed-media/$path")
    private fun opened(): Int = context.contentResolver.query(uri("status"), null, null, null, null)!!.use {
        assertTrue(it.moveToFirst()); it.getInt(0)
    }
    private fun temporaryFiles(): Set<String> = context.cacheDir.listFiles()?.filter { it.name.startsWith("scan-") && it.extension == "media" }?.map(File::getName)?.toSet().orEmpty()

    @Test fun timeoutCancelsDelayedProviderQueryPromptly() = runBlocking {
        opened() // Resolve/start the test provider before measuring a blocked operation.
        val started = SystemClock.elapsedRealtime()
        try {
            ProviderMediaImporter(context).import(uri("delayed-query"), 1024, timeoutMs = 300)
            fail("Delayed query must time out")
        } catch (_: TimeoutCancellationException) { }
        assertTrue("Provider query prevented timeout completion", SystemClock.elapsedRealtime() - started < 2_000)
    }
    @Test fun cancellingBlockedPipeReturnsWithoutWaitingForProviderAndCleansTempFile() = runBlocking {
        val before = temporaryFiles()
        val count = opened()
        val operation = async { ProviderMediaImporter(context).import(uri("blocked-pipe"), 1024, timeoutMs = 10_000) }
        withTimeout(5_000) { while (opened() <= count) delay(10) }
        delay(100) // Ensure the input read is pending, not just the descriptor request.
        val started = SystemClock.elapsedRealtime()
        operation.cancelAndJoin()
        assertTrue("Cancellation waited for the provider's pipe EOF", SystemClock.elapsedRealtime() - started < 2_000)
        withTimeout(5_000) { while (temporaryFiles() != before) delay(20) }
    }
    @Test fun unknownSizeStillEnforcesActualByteLimitAndRemovesPartialCopy() = runBlocking {
        val before = temporaryFiles()
        try {
            ProviderMediaImporter(context).import(uri("oversized"), 1024)
            fail("Unknown-size provider bypassed the actual byte limit")
        } catch (_: IllegalArgumentException) { }
        assertEquals(before, temporaryFiles())
    }
    @Test fun dispatchedErrorReachesCallerAndClosesOwnedResources() = runBlocking {
        val closed = AtomicBoolean(false)
        try {
            withTimeout(2_000) {
                cancellableProviderWork<Unit> {
                    own(Closeable { closed.set(true) })
                    throw AssertionError("Simulated allocation failure; no memory exhaustion")
                }
            }
            fail("Dispatched Error was not propagated")
        } catch (failure: AssertionError) {
            assertTrue(failure.message.orEmpty().startsWith("Simulated allocation failure"))
        }
        withTimeout(2_000) { while (!closed.get()) delay(10) }
    }
}
