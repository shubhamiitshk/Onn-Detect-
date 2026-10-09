package com.onndetect.inference

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Optional direct-copy loader; compressed entries require a bounded size. */
class SequentialModelLoader(private val context: Context) : ModelLoader {
    override fun loadMapped(fileName: String): MappedModel {
        val size = context.assets.open("models/$fileName").use { stream ->
            var count = 0L
            val chunk = ByteArray(8192)
            while (true) { val n = stream.read(chunk); if (n < 0) break; count += n; require(count <= 32 * 1024 * 1024) { "Model too large" } }
            count.toInt()
        }
        require(size > 0)
        val buffer = ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder())
        context.assets.open("models/$fileName").use { stream ->
            val chunk = ByteArray(8192)
            while (true) { val n = stream.read(chunk); if (n < 0) break; buffer.put(chunk, 0, n) }
        }
        buffer.flip()
        return MappedModel(buffer, null)
    }
}
