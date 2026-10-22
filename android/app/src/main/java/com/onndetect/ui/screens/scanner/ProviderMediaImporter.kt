package com.onndetect.ui.screens.scanner

import android.content.Context
import android.net.Uri
import android.os.CancellationSignal
import android.provider.OpenableColumns
import com.onndetect.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeout
import java.io.Closeable
import java.io.File
import kotlin.coroutines.EmptyCoroutineContext
import kotlin.coroutines.resumeWithException

/** Copies selected content into bounded private storage before a decoder touches it. */
internal class ProviderMediaImporter(private val context: Context) {
    private fun requireMedia(condition: Boolean, messageResource: Int) {
        if (!condition) throw MediaImportException(messageResource)
    }
    suspend fun import(uri: Uri, maxBytes: Long, timeoutMs: Long = 20_000): File = withTimeout(timeoutMs) {
        require(uri.scheme == "content" && maxBytes > 0)
        cancellableProviderWork(dispose = { file: File -> file.delete() }) {
            own(context.contentResolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null, signal))?.use { cursor ->
                checkActive()
                if (cursor.moveToFirst() && !cursor.isNull(0)) {
                    val reported = cursor.getLong(0)
                    requireMedia(reported != 0L, R.string.invalid_input)
                    requireMedia(reported <= maxBytes, R.string.input_too_large)
                }
            }
            checkActive()
            val descriptor = own(context.contentResolver.openAssetFileDescriptor(uri, "r", signal))
            if (descriptor != null && descriptor.declaredLength > 0L) {
                requireMedia(descriptor.declaredLength <= maxBytes, R.string.input_too_large)
            }
            val input = own(descriptor?.createInputStream()) ?: own(context.contentResolver.openInputStream(uri))
                ?: throw MediaImportException(R.string.invalid_input)
            val file = File.createTempFile("scan-", ".media", context.cacheDir)
            var delivered = false
            try {
                input.use { inStream ->
                    own(file.outputStream())!!.use { output ->
                        val buffer = ByteArray(32 * 1024)
                        var total = 0L
                        while (true) {
                            checkActive()
                            val count = inStream.read(buffer)
                            checkActive()
                            if (count < 0) break
                            if (count == 0) continue
                            total += count
                            requireMedia(total <= maxBytes, R.string.input_too_large)
                            output.write(buffer, 0, count)
                        }
                        requireMedia(total > 0, R.string.invalid_input)
                    }
                }
                checkActive()
                delivered = true
                file
            } finally {
                if (!delivered) file.delete()
            }
        }
    }
}

internal class MediaImportException(val messageResource: Int) : IllegalArgumentException()

/** Cancellation closes resources even while a provider is blocked inside a read or Binder call. */
@OptIn(ExperimentalCoroutinesApi::class)
internal suspend fun <T> cancellableProviderWork(dispose: (T) -> Unit = {}, block: ProviderWork.() -> T): T =
    suspendCancellableCoroutine { continuation ->
        val work = ProviderWork()
        continuation.invokeOnCancellation { work.cancel() }
        Dispatchers.IO.dispatch(EmptyCoroutineContext, Runnable {
            try {
                work.checkActive()
                val value = work.block()
                continuation.resume(value) { dispose(value) }
            } catch (failure: Throwable) {
                if (continuation.isActive) continuation.resumeWithException(failure)
            } finally {
                work.close()
            }
        })
    }

internal class ProviderWork : Closeable {
    val signal = CancellationSignal()
    private val lock = Any()
    private val resources = mutableListOf<Closeable>()
    private var closed = false
    fun checkActive() = signal.throwIfCanceled()
    fun <T : Closeable> own(resource: T?): T? {
        if (resource == null) return null
        synchronized(lock) {
            if (closed) {
                runCatching { resource.close() }
                signal.throwIfCanceled()
                error("Import resources already closed")
            }
            resources += resource
        }
        return resource
    }
    fun cancel() {
        // Close the owned descriptor/streams before asking the remote provider to cancel.
        close()
        runCatching { signal.cancel() }
    }
    override fun close() {
        val owned = synchronized(lock) {
            if (closed) return
            closed = true
            resources.toList().asReversed().also { resources.clear() }
        }
        owned.forEach { runCatching { it.close() } }
    }
}
