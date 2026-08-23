package com.onndetect.inference

import android.os.Build
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate

enum class DelegateType { GPU, CPU }

/**
 * Chooses the fastest delegate the current device actually supports.
 * The GPU delegate is skipped on emulators and incompatible GPUs because a
 * failed delegate initialization silently falls back to CPU anyway after
 * paying the setup cost.
 */
object DelegateSelector {

    fun preferred(): DelegateType {
        // Force CPU delegate for now! 
        // 1. Android Emulator GPU drivers crash when running TFLite GPU delegates.
        // 2. GPU delegates do not support int8 quantized models (like our audio model).
        return DelegateType.CPU
    }

    fun apply(options: Interpreter.Options, type: DelegateType, holder: DelegateHolder) {
        if (type == DelegateType.GPU) {
            try {
                // Default-construct: CompatibilityList options types differ
                // across TFLite releases; the plain delegate is stable API.
                val delegate = GpuDelegate()
                holder.gpu = delegate
                options.addDelegate(delegate)
                return
            } catch (_: Throwable) {
                holder.gpu?.close()
                holder.gpu = null
            }
        }
        options.setNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(2, 4))
    }
}

/** Owns native delegate resources so they can be released deterministically. */
class DelegateHolder(var gpu: GpuDelegate? = null) : AutoCloseable {
    override fun close() {
        gpu?.close()
        gpu = null
    }
}
