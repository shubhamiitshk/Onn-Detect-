package com.onndetect.inference

import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.inject.Inject
import javax.inject.Singleton

/** All interpreter access, invocation and closing have one exclusive owner. */
@Singleton
class LiteRTInferenceEngine @Inject constructor(private val loader: MemoryMappedModelLoader) : AutoCloseable {
    private data class Loaded(val interpreter: Interpreter, val holder: DelegateHolder, val mapped: MappedModel)
    data class Contract(val inputShape: IntArray, val outputShape: IntArray, val inputBytes: Int)
    private val loaded = HashMap<String, Loaded>()
    private val lock = Any()

    private fun load(modelFile: String): Interpreter {
        requireModelContract(modelFile.matches(Regex("[a-zA-Z0-9_.-]+\\.tflite"))) { "Invalid model filename" }
        loaded[modelFile]?.let { return it.interpreter }
        val mapped = loader.loadMapped(modelFile)
        val holder = DelegateHolder()
        try {
            val options = Interpreter.Options()
            DelegateSelector.apply(options, DelegateSelector.preferred(), holder)
            val interpreter = Interpreter(mapped.buffer, options)
            try {
                interpreter.allocateTensors()
                validate(interpreter)
            } catch (e: Throwable) { interpreter.close(); throw e }
            loaded[modelFile] = Loaded(interpreter, holder, mapped)
            return interpreter
        } catch (e: Throwable) {
            holder.close()
            mapped.closeable?.close()
            if (e is IllegalArgumentException) throw ModelContractException("Unable to load the packaged model interface", e)
            throw e
        }
    }

    private fun validate(interpreter: Interpreter) {
        requireModelContract(interpreter.inputTensorCount == 1 && interpreter.outputTensorCount == 1) { "Expected one input/output" }
        val input = interpreter.getInputTensor(0)
        val output = interpreter.getOutputTensor(0)
        requireModelContract(input.dataType() == DataType.FLOAT32 && output.dataType() == DataType.FLOAT32) { "Expected float32 interface" }
        requireModelContract(input.shape().isNotEmpty() && input.shape().all { it > 0 } && input.shape().first() == 1 && input.numBytes() <= 32 * 1024 * 1024) { "Invalid input shape/budget" }
        requireModelContract(output.numElements() == 1) { "Expected scalar model output" }
    }

    fun contract(modelFile: String): Contract = synchronized(lock) {
        val interpreter = load(modelFile)
        Contract(interpreter.getInputTensor(0).shape().copyOf(), interpreter.getOutputTensor(0).shape().copyOf(), interpreter.getInputTensor(0).numBytes())
    }

    /** Float32 buffers have exact tensor byte size; shaped arrays are validated recursively. */
    fun predict(modelFile: String, input: Any, checkStop: () -> Unit = {}): Float = synchronized(lock) {
        checkStop()
        val interpreter = load(modelFile)
        val tensor = interpreter.getInputTensor(0)
        if (input is ByteBuffer) {
            requireModelContract(input.isDirect && input.order() == ByteOrder.nativeOrder() && input.remaining() == tensor.numBytes()) { "Invalid input buffer" }
            val floats = input.duplicate().order(ByteOrder.nativeOrder()).asFloatBuffer()
            while (floats.hasRemaining()) requireModelContract(floats.get().isFinite()) { "Non-finite model input" }
        } else {
            requireModelContract(shapeOf(input).contentEquals(tensor.shape())) { "Model input shape mismatch" }
        }
        val output = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
        try { interpreter.run(input, output) }
        catch (e: IllegalArgumentException) { throw ModelContractException("Packaged model invocation rejected its tensor interface", e) }
        checkStop()
        output.rewind()
        val score = output.float
        requireModelContract(score.isFinite() && score in 0f..1f) { "Invalid model output" }
        score
    }

    private fun shapeOf(value: Any): IntArray = when (value) {
        is FloatArray -> {
            requireModelContract(value.all { it.isFinite() }) { "Non-finite model input" }
            intArrayOf(value.size)
        }
        is Array<*> -> {
            requireModelContract(value.isNotEmpty() && value.all { it != null }) { "Empty tensor dimension" }
            val child = shapeOf(value[0]!!)
            requireModelContract(value.all { shapeOf(it!!).contentEquals(child) }) { "Ragged model input" }
            intArrayOf(value.size) + child
        }
        else -> throw ModelContractException("Unsupported model input")
    }

    override fun close() = synchronized(lock) {
        loaded.values.forEach {
            runCatching { it.interpreter.close() }
            runCatching { it.holder.close() }
            runCatching { it.mapped.closeable?.close() }
        }
        loaded.clear()
    }
}
