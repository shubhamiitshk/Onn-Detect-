package com.onndetect.inference

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Single owner of TFLite interpreters. Models are memory-mapped, initialized
 * lazily on first use and kept alive for the process lifetime; delegates are
 * released in [close].
 */
@Singleton
class LiteRTInferenceEngine @Inject constructor(
    private val loader: MemoryMappedModelLoader,
) : AutoCloseable {

    private data class LoadedInterpreter(
        val interpreter: Interpreter,
        val holder: DelegateHolder
    )

    private val loaded = HashMap<String, LoadedInterpreter>()
    private val lock = Any()

    @androidx.annotation.WorkerThread
    fun interpreterFor(modelFile: String): Interpreter {
        synchronized(lock) {
            loaded[modelFile]?.let { return it.interpreter }
            val mapped = runCatching { loader.loadMapped(modelFile) }
                .getOrElse { throw IllegalStateException("Cannot load model $modelFile", it) }
            val options = Interpreter.Options()
            val holder = DelegateHolder()
            DelegateSelector.apply(options, DelegateSelector.preferred(), holder)
            val interpreter = Interpreter(mapped.buffer, options)
            Log.i(TAG, "Loaded $modelFile with ${DelegateSelector.preferred()} delegate")
            loaded[modelFile] = LoadedInterpreter(interpreter, holder)
            return interpreter
        }
    }

    /** Runs a 1-input/1-output float model. Input must be a shaped array or ByteBuffer. */
    fun predict(modelFile: String, input: Any, outputShape: IntArray = intArrayOf(1)): Float {
        val output = when (outputShape.size) {
            1 -> Array(1) { FloatArray(outputShape[0]) }
            else -> arrayOf(arrayOf(FloatArray(outputShape.last())))
        }
        interpreterFor(modelFile).run(input, output)
        return when (val o = output[0]) {
            is FloatArray -> o[0]
            is Array<*> -> ((o as Array<FloatArray>)[0])[0]
            else -> error("Unsupported output type")
        }
    }

    override fun close() {
        synchronized(lock) {
            loaded.values.forEach {
                runCatching { it.interpreter.close() }
                runCatching { it.holder.close() }
            }
            loaded.clear()
        }
    }

    companion object {
        private const val TAG = "LiteRTEngine"
    }
}
