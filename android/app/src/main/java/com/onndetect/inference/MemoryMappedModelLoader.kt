package com.onndetect.inference

import android.content.Context
import androidx.annotation.WorkerThread
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

data class MappedModel(val buffer: ByteBuffer, val closeable: AutoCloseable?)
interface ModelLoader { @WorkerThread fun loadMapped(fileName: String): MappedModel }

/** mmap survives closing the mapping file descriptors. Models must be uncompressed. */
class MemoryMappedModelLoader(private val context: Context) : ModelLoader {
    override fun loadMapped(fileName: String): MappedModel {
        return context.assets.openFd("models/$fileName").use { descriptor ->
            descriptor.createInputStream().use { input ->
                val buffer = input.channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
                buffer.order(ByteOrder.nativeOrder())
                MappedModel(buffer, null)
            }
        }
    }
}
