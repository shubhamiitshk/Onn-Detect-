package com.onndetect.inference

import org.tensorflow.lite.Interpreter

enum class DelegateType { CPU }

/** Conservative standalone CPU baseline; change only after device benchmarks. */
object DelegateSelector {
    fun preferred(): DelegateType = DelegateType.CPU
    @Suppress("UNUSED_PARAMETER")
    fun apply(options: Interpreter.Options, type: DelegateType, holder: DelegateHolder) {
        options.setNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(1, 4))
        options.setUseXNNPACK(true)
    }
}

class DelegateHolder : AutoCloseable {
    override fun close() = Unit
}
