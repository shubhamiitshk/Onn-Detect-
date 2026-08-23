package com.onndetect.inference

import android.content.Context
import androidx.annotation.WorkerThread

/** A model flatbuffer plus whatever resource must stay open while it lives. */
data class MappedModel(val buffer: java.nio.ByteBuffer, val closeable: AutoCloseable?)

/**
 * Loads TFLite flatbuffers from assets fully memory-mapped, so the OS pages
 * model weights in and out on demand instead of copying them onto the heap.
 */
interface ModelLoader {
    @WorkerThread
    fun loadMapped(fileName: String): MappedModel
}

class MemoryMappedModelLoader(private val context: Context) : ModelLoader {

    override fun loadMapped(fileName: String): MappedModel {
        val afd = context.assets.openFd("models/$fileName")
        val inputStream = java.io.FileInputStream(afd.fileDescriptor)
        val channel = inputStream.channel
        val buffer = channel.map(
            java.nio.channels.FileChannel.MapMode.READ_ONLY,
            afd.startOffset,
            afd.declaredLength
        )
        return MappedModel(buffer, inputStream)
    }
}
