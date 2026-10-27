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

    /**
     * One lock per MODEL, not one lock for the whole engine.
     *
     * The engine is a singleton shared by the visual, audio and provenance
     * detectors. With a single global lock the 89.96 MB SigLIP probe's
     * inference (300-540 ms) blocked every other model for its whole
     * duration, and its FIRST load - mmap, Interpreter construction,
     * allocateTensors and XNNPACK weight packing for a 12-block ViT -
     * blocked them for seconds. A TFLite interpreter is not thread-safe,
     * so the SAME model must stay serialised; different models must not
     * block each other.
     */
    private val modelLocks = java.util.concurrent.ConcurrentHashMap<String, Any>()

    /** Guards the `loaded` MAP and `blockedModels` set only. Never held during model construction. */
    private val mapLock = Any()

    /**
     * Models temporarily locked out from loading while an exclusive Tier-2 VLM session holds
     * native memory (e.g. `siglip_probe_int8.tflite`). Guarded by [mapLock].
     */
    private val blockedModels = HashSet<String>()

    /**
     * Detailed result of evicting a model from the interpreter cache, distinguishing cache
     * map removal from successful closure of the underlying native TFLite/delegate/mmap resources.
     */
    data class EvictionReport(
        val modelFile: String,
        val wasCached: Boolean,
        val removedFromCache: Boolean,
        val interpreterClosed: Boolean,
        val delegateClosed: Boolean,
        val mappingClosed: Boolean,
        val closeError: Throwable? = null,
    ) {
        /**
         * True iff the model is no longer cached AND every underlying resource that was open
         * (`interpreter`, `holder`, `mapped.closeable`) closed without throwing an exception.
         */
        val resourcesCleanlyReleased: Boolean
            get() = !wasCached || (
                removedFromCache &&
                    interpreterClosed &&
                    delegateClosed &&
                    mappingClosed &&
                    closeError == null
                )
    }

    /**
     * Explicit ownership lease that evicts specified heavy models (such as `siglip_probe_int8.tflite`)
     * and prevents any concurrent or subsequent inference call from reloading them until [close] is called.
     */
    inner class ExclusiveSessionLease internal constructor(
        val lockedOutModels: Set<String>,
        val evictionReports: Map<String, EvictionReport>,
    ) : AutoCloseable {
        @Volatile
        var isActive: Boolean = true
            private set

        val allResourcesCleanlyReleased: Boolean
            get() = evictionReports.values.all { it.resourcesCleanlyReleased }

        override fun close() {
            if (!isActive) return
            synchronized(mapLock) {
                blockedModels.removeAll(lockedOutModels)
                isActive = false
            }
        }
    }

    private fun lockFor(modelFile: String): Any =
        modelLocks.computeIfAbsent(modelFile) { Any() }

    private fun load(modelFile: String): Interpreter {
        requireModelContract(modelFile.matches(Regex("[a-zA-Z0-9_.-]+\\.tflite"))) { "Invalid model filename" }
        // The per-model lock serialises the same model; the map lock is only
        // held for the map access, so one model's slow first load does not
        // stall another model's first load.
        synchronized(lockFor(modelFile)) {
            synchronized(mapLock) {
                if (modelFile in blockedModels) {
                    throw ModelContractException(
                        "Model '$modelFile' is locked out by an active exclusive VLM session lease"
                    )
                }
                loaded[modelFile]
            }?.let { return it.interpreter }
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
                synchronized(mapLock) { loaded[modelFile] = Loaded(interpreter, holder, mapped) }
                return interpreter
            } catch (e: Throwable) {
                holder.close()
                mapped.closeable?.close()
                if (e is IllegalArgumentException) throw ModelContractException("Unable to load the packaged model interface", e)
                throw e
            }
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

    fun contract(modelFile: String): Contract = synchronized(lockFor(modelFile)) {
        val interpreter = load(modelFile)
        Contract(interpreter.getInputTensor(0).shape().copyOf(), interpreter.getOutputTensor(0).shape().copyOf(), interpreter.getInputTensor(0).numBytes())
    }

    /** Float32 buffers have exact tensor byte size; shaped arrays are validated recursively. */
    fun predict(modelFile: String, input: Any, checkStop: () -> Unit = {}): Float {
        checkStop()
        // The model's own lock: the same interpreter is never entered
        // concurrently, and a different model is never blocked by it.
        return synchronized(lockFor(modelFile)) {
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

    /**
     * Evicts a single cached model interpreter and closes its native delegate and mmap
     * buffers, returning an [EvictionReport] that separates cache removal from native
     * resource closure.
     */
    fun evictModelWithReport(modelFile: String): EvictionReport {
        val lock = lockFor(modelFile)
        synchronized(lock) {
            val removed = synchronized(mapLock) { loaded.remove(modelFile) }
                ?: return EvictionReport(
                    modelFile = modelFile,
                    wasCached = false,
                    removedFromCache = false,
                    interpreterClosed = true,
                    delegateClosed = true,
                    mappingClosed = true,
                    closeError = null,
                )
            var firstError: Throwable? = null
            fun recordError(t: Throwable) {
                if (firstError == null) firstError = t else firstError!!.addSuppressed(t)
            }
            val interpOk = try {
                removed.interpreter.close()
                true
            } catch (t: Throwable) {
                recordError(t)
                false
            }
            val delegateOk = try {
                removed.holder.close()
                true
            } catch (t: Throwable) {
                recordError(t)
                false
            }
            val mappedOk = try {
                removed.mapped.closeable?.close()
                true
            } catch (t: Throwable) {
                recordError(t)
                false
            }
            return EvictionReport(
                modelFile = modelFile,
                wasCached = true,
                removedFromCache = true,
                interpreterClosed = interpOk,
                delegateClosed = delegateOk,
                mappingClosed = mappedOk,
                closeError = firstError,
            )
        }
    }

    /**
     * Evicts [modelFile] and returns true ONLY if it was cached, removed from cache, AND all
     * underlying native resources (`interpreter`, `holder`, `mapped.closeable`) closed cleanly
     * without throwing.
     */
    fun evictModel(modelFile: String): Boolean {
        val report = evictModelWithReport(modelFile)
        return report.removedFromCache && report.resourcesCleanlyReleased
    }

    /**
     * Acquires an exclusive session lease for a Tier-2 on-device VLM invocation.
     * Blocks any concurrent/subsequent reload of [modelsToEvictAndBlock] first, then evicts
     * any cached instances in deterministic lock order.
     */
    fun acquireExclusiveVlmSession(
        modelsToEvictAndBlock: Set<String> = setOf("siglip_probe_int8.tflite"),
    ): ExclusiveSessionLease {
        val ordered = modelsToEvictAndBlock.sortedBy { lockFor(it).hashCode() }
        synchronized(mapLock) {
            blockedModels.addAll(ordered)
        }
        val reports = LinkedHashMap<String, EvictionReport>()
        try {
            for (modelFile in ordered) {
                reports[modelFile] = evictModelWithReport(modelFile)
            }
        } catch (t: Throwable) {
            synchronized(mapLock) {
                blockedModels.removeAll(ordered.toSet())
            }
            throw t
        }
        return ExclusiveSessionLease(ordered.toSet(), reports)
    }

    /** Returns true if [modelFile] is currently locked out by an active [ExclusiveSessionLease]. */
    fun isModelBlocked(modelFile: String): Boolean =
        synchronized(mapLock) { modelFile in blockedModels }

    /** Returns true if [modelFile] currently has an active cached interpreter in memory. */
    fun isModelLoaded(modelFile: String): Boolean =
        synchronized(mapLock) { loaded.containsKey(modelFile) }

    override fun close() {
        // Take every model lock before closing, so an in-flight inference
        // cannot race a closed interpreter.
        val locks = synchronized(mapLock) { modelLocks.values.toList() }
        // A stable order, so two threads closing concurrently cannot
        // deadlock on each other's model locks.
        locks.sortedBy { it.hashCode() }.forEach { lock ->
            synchronized(lock) {
                synchronized(mapLock) {
                    loaded.values.forEach {
                        runCatching { it.interpreter.close() }
                        runCatching { it.holder.close() }
                        runCatching { it.mapped.closeable?.close() }
                    }
                    loaded.clear()
                }
            }
        }
    }
}
