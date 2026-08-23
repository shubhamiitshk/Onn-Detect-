package com.onndetect.inference

import android.content.Context

/**
 * Low-memory fallback path: reads the asset into a direct ByteBuffer once.
 * Used when memory mapping fails (e.g., compressed APK entry).
 */
class SequentialModelLoader(private val context: Context) : ModelLoader {

    override fun loadMapped(fileName: String): MappedModel {
        val bytes = context.assets.open("models/$fileName").use { it.readBytes() }
        val buffer = java.nio.ByteBuffer.allocateDirect(bytes.size)
        buffer.order(java.nio.ByteOrder.nativeOrder())
        buffer.put(bytes)
        buffer.rewind()
        return MappedModel(buffer, null)
    }
}
