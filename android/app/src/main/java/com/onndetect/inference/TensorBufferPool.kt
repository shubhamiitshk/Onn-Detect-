package com.onndetect.inference

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Reusable direct ByteBuffers to avoid per-scan allocation churn. */
class TensorBufferPool(private val maxBuffers: Int = 8) {

    private val pool = ArrayDeque<ByteBuffer>()

    @Synchronized
    fun obtain(sizeBytes: Int): ByteBuffer {
        val existing = pool.removeFirstOrNull()
        if (existing != null && existing.capacity() >= sizeBytes) {
            existing.clear()
            existing.order(ByteOrder.nativeOrder())
            existing.limit(sizeBytes)
            return existing
        }
        return ByteBuffer.allocateDirect(sizeBytes).order(ByteOrder.nativeOrder())
    }

    @Synchronized
    fun recycle(buffer: ByteBuffer) {
        if (pool.size < maxBuffers) {
            buffer.clear()
            pool.addLast(buffer)
        }
    }
}
