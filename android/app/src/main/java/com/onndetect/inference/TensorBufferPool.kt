package com.onndetect.inference

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Bounded direct-buffer cache; checked-out buffers have exclusive callers. */
class TensorBufferPool(private val maxBuffers: Int = 8, private val maxBytes: Int = 8 * 1024 * 1024) {
    private val pool = ArrayDeque<ByteBuffer>()
    @Synchronized fun obtain(sizeBytes: Int): ByteBuffer {
        require(sizeBytes in 1..maxBytes)
        val existing = pool.firstOrNull { it.capacity() >= sizeBytes }
        if (existing != null) pool.remove(existing)
        return (existing ?: ByteBuffer.allocateDirect(sizeBytes)).apply {
            clear(); order(ByteOrder.nativeOrder()); limit(sizeBytes)
        }
    }
    @Synchronized fun recycle(buffer: ByteBuffer) {
        if (buffer.isDirect && pool.none { it === buffer } && pool.size < maxBuffers &&
            pool.sumOf { it.capacity().toLong() } + buffer.capacity() <= maxBytes) {
            buffer.clear(); pool.addLast(buffer)
        }
    }
}
