package com.onndetect.chat

import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import android.os.StatFs
import com.onndetect.detection.visual.LlamaCppVlmPrefillBackend
import com.onndetect.inference.LiteRTInferenceEngine
import com.onndetect.ui.screens.scanner.MediaImportException
import com.onndetect.ui.screens.scanner.ProviderMediaImporter
import java.io.File
import java.io.RandomAccessFile
import java.security.MessageDigest
import java.util.UUID
import java.util.zip.ZipInputStream
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONObject

/** User-imported conversation weights. This store never grants forensic model validation. */
object ChatModelStore {
    const val MANIFEST_FILE = "chat_model.json"
    private const val MAX_TOTAL_BYTES = 3L * 1024 * 1024 * 1024
    private const val MIN_LANGUAGE_BYTES = 50L * 1024 * 1024
    private const val MAX_PROJECTOR_BYTES = 768L * 1024 * 1024
    private const val DISK_RESERVE_BYTES = 64L * 1024 * 1024
    private const val MAX_ENTRIES = 4
    private val operation = Mutex()
    private val mutableStatus = MutableStateFlow(Status(null, false, "No conversation model installed.", false))
    val status: StateFlow<Status> = mutableStatus.asStateFlow()

    data class ModelFile(val fileName: String, val bytes: Long, val sha256: String)
    data class Installed(
        val modelId: String,
        val displayName: String,
        val architecture: String,
        val hiddenDim: Int,
        val language: ModelFile,
        val projector: ModelFile?,
        val directory: File,
    ) {
        val totalBytes: Long get() = language.bytes + (projector?.bytes ?: 0)
        val supportsVision: Boolean get() = projector != null
    }
    data class Status(val installed: Installed?, val isReady: Boolean, val message: String, val runtimeAvailable: Boolean) {
        val modelName: String get() = installed?.displayName ?: "Built-in evidence replies"
        val supportsVision: Boolean get() = isReady && installed?.supportsVision == true
    }

    private fun home(context: Context): File = File(context.filesDir, "local_chat_models").apply { mkdirs() }
    private fun active(context: Context): File = File(home(context), "active")

    /** Quick availability check; invocation repeats exact byte-hash verification on an IO worker. */
    fun inspect(context: Context): Status {
        val native = LlamaCppVlmPrefillBackend.isNativeLibraryLoaded()
        val directory = active(context)
        if (!directory.isDirectory) return Status(null, false, "Import a conversation model to enable normal chat and visual questions.", native)
        return try {
            val installed = readManifest(directory, verifyHashes = false)
            Status(installed, native,
                if (native) "Ready · imported on this device · CPU runtime" else "The native model runtime is unavailable on this device.", native)
        } catch (error: Exception) {
            Status(null, false, error.message ?: "Installed chat files are incomplete. Import the pack again.", native)
        }
    }

    suspend fun refresh(context: Context): Status = withContext(Dispatchers.IO) {
        operation.withLock {
            recoverPrevious(context)
            inspect(context).also { mutableStatus.value = it }
        }
    }

    /** Serializes installation/removal with invocation; native residency ends before its lease. */
    suspend fun <T> withRuntime(
        context: Context,
        inferenceEngine: LiteRTInferenceEngine,
        useVision: Boolean = false,
        block: suspend (LlamaCppVlmPrefillBackend, Installed) -> T,
    ): T? = withContext(Dispatchers.IO) {
        operation.withLock {
            recoverPrevious(context)
            val availability = inspect(context)
            mutableStatus.value = availability
            if (!availability.isReady) return@withLock null
            val job = currentCoroutineContext()
            val installed = try { readManifest(active(context), verifyHashes = true, checkStop = { job.ensureActive() }) }
            catch (cancelled: CancellationException) { throw cancelled }
            catch (error: Exception) {
                mutableStatus.value = availability.copy(isReady = false, message = error.message ?: "Conversation model integrity check failed.")
                throw error
            }
            try {
                inferenceEngine.acquireExclusiveVlmSession().use {
                    val info = ActivityManager.MemoryInfo()
                    (context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(info)
                    check(!info.lowMemory) { "This device is low on memory. Close other apps and retry the local model." }
                    check(!useVision || installed.projector != null) { "This conversation model has no vision projector. Import its compatible visual pack first." }
                    // A conservative scheduling budget, not a claim about measured model RAM.
                    // Check after specialist eviction, using the modules this request will load.
                    val estimatedBudget = installed.language.bytes + (if (useVision) installed.projector?.bytes ?: 0 else 0) + 384L * 1024 * 1024
                    check(info.availMem >= estimatedBudget) {
                        "This request needs an estimated ${estimatedBudget / (1024 * 1024)} MB free-memory budget; ${info.availMem / (1024 * 1024)} MB is available. Close other apps or use a smaller model."
                    }
                    LlamaCppVlmPrefillBackend(
                        modelPath = File(installed.directory, installed.language.fileName).absolutePath,
                        mmprojPath = if (useVision) installed.projector?.let { File(installed.directory, it.fileName).absolutePath } else null,
                        hiddenDim = installed.hiddenDim,
                        imageResolutionCapPx = 256,
                    ).use { backend ->
                        job.ensureActive()
                        backend.loadNativeContext()
                        try { block(backend, installed) }
                        finally { backend.unloadNativeContext() }
                    }
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                mutableStatus.value = availability.copy(message = "Local model could not finish: ${error.message.orEmpty().take(180)}")
                throw error
            }
        }
    }

    suspend fun importPackZip(context: Context, uri: Uri, onProgress: (String) -> Unit = {}): Result<Installed> = withContext(Dispatchers.IO) {
        try {
            Result.success(operation.withLock {
                val job = currentCoroutineContext()
                val root = home(context)
                recoverPrevious(context)
                onProgress("Copying conversation pack…")
                val available = StatFs(root.absolutePath).availableBytes
                check(available > 2 * MIN_LANGUAGE_BYTES + DISK_RESERVE_BYTES) { "Not enough free storage to stage a conversation model safely." }
                val maxArchiveBytes = minOf(MAX_TOTAL_BYTES, (available - DISK_RESERVE_BYTES) / 2)
                val archive = try { ProviderMediaImporter(context).import(uri, maxArchiveBytes, 180_000) }
                catch (error: MediaImportException) { throw IllegalArgumentException(context.getString(error.messageResource), error) }
                val staging = File(root, "candidate-${UUID.randomUUID()}")
                try {
                    check(staging.mkdir()) { "Could not create model staging storage." }
                    onProgress("Extracting model files…")
                    ZipInputStream(archive.inputStream().buffered()).use { zip ->
                        val names = mutableSetOf<String>()
                        val buffer = ByteArray(128 * 1024)
                        var total = 0L
                        while (true) {
                            job.ensureActive()
                            val entry = zip.nextEntry ?: break
                            require(!entry.isDirectory && safeFileName(entry.name)) { "Chat packs must contain flat, safe filenames." }
                            require(names.add(entry.name) && names.size <= MAX_ENTRIES) { "Duplicate or excessive files in the chat pack." }
                            val destination = File(staging, entry.name)
                            require(destination.canonicalFile.parentFile == staging.canonicalFile) { "Invalid archive path." }
                            var fileBytes = 0L
                            destination.outputStream().buffered().use { output ->
                                while (true) {
                                    job.ensureActive()
                                    val count = zip.read(buffer)
                                    if (count < 0) break
                                    fileBytes += count
                                    total += count
                                    require(total <= MAX_TOTAL_BYTES) { "This chat pack expands beyond the 3 GB limit." }
                                    require(entry.name != MANIFEST_FILE || fileBytes <= 64 * 1024) { "Chat manifest is too large." }
                                    check(StatFs(root.absolutePath).availableBytes > DISK_RESERVE_BYTES + count) { "Not enough free storage to finish model installation." }
                                    output.write(buffer, 0, count)
                                }
                            }
                            zip.closeEntry()
                        }
                    }
                    onProgress("Checking model architecture and exact file hashes…")
                    val candidate = readManifest(staging, true, checkStop = { job.ensureActive() })
                    job.ensureActive()
                    val previous = File(root, "previous")
                    val current = File(root, "active")
                    removeOwned(previous, root)
                    if (current.exists()) check(current.renameTo(previous)) { "Could not preserve the current chat model." }
                    if (!staging.renameTo(current)) {
                        if (previous.exists()) previous.renameTo(current)
                        error("Could not activate the new chat model; the previous model was retained.")
                    }
                    removeOwned(previous, root)
                    val installed = candidate.copy(directory = current)
                    mutableStatus.value = inspect(context)
                    onProgress("Conversation model ready.")
                    installed
                } finally {
                    archive.delete()
                    removeOwned(staging, root)
                }
            })
        } catch (cancelled: CancellationException) { throw cancelled }
        catch (error: Exception) { Result.failure(error) }
    }

    suspend fun uninstall(context: Context) = withContext(Dispatchers.IO) {
        operation.withLock {
            val root = home(context)
            removeOwned(File(root, "active"), root)
            removeOwned(File(root, "previous"), root)
            mutableStatus.value = inspect(context)
        }
    }

    private fun recoverPrevious(context: Context) {
        val root = home(context)
        val current = File(root, "active")
        val previous = File(root, "previous")
        if (!current.exists() && previous.isDirectory) check(previous.renameTo(current)) { "Could not recover the previous chat model." }
        // An interrupted process cannot own a live import now; remove only this
        // store's explicitly named staging directories after restoring the old model.
        root.listFiles()?.filter { it.isDirectory && it.name.matches(Regex("candidate-[a-f0-9-]{36}")) }
            ?.forEach { removeOwned(it, root) }
    }

    private fun removeOwned(file: File, root: File) {
        require(file.canonicalFile.parentFile == root.canonicalFile) { "Refusing an out-of-store model path." }
        if (file.exists()) check(file.deleteRecursively()) { "Could not clean model staging storage." }
    }

    internal fun safeFileName(name: String): Boolean = name.matches(Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,99}")) && name !in setOf(".", "..")

    internal fun readManifest(directory: File, verifyHashes: Boolean, checkStop: () -> Unit = {}): Installed {
        val manifestFile = File(directory, MANIFEST_FILE)
        require(manifestFile.isFile && manifestFile.length() in 2..65_536) { "This archive has no supported chat_model.json manifest." }
        val json = JSONObject(manifestFile.readText(Charsets.UTF_8))
        require(json.getInt("schema_version") == 1 && json.getString("purpose") == "conversation") { "This pack is not a conversation-model pack." }
        val id = json.getString("model_id").also { require(it.isNotBlank() && it.length <= 200) }
        val name = json.getString("display_name").also { require(it.isNotBlank() && it.length <= 120) }
        val architecture = json.getString("architecture")
        require(architecture == "qwen35") { "This conversation runtime currently supports the tested Qwen3.5 architecture." }
        val hiddenDim = json.getInt("hidden_dim")
        require(hiddenDim in setOf(1024, 2048, 2560)) { "Unsupported Qwen hidden dimension." }
        fun modelFile(obj: JSONObject, language: Boolean): ModelFile {
            val fileName = obj.getString("file")
            require(safeFileName(fileName) && fileName.endsWith(".gguf")) { "Invalid model filename." }
            val bytes = obj.getLong("bytes")
            require(bytes in (if (language) MIN_LANGUAGE_BYTES else 1_048_576L)..(if (language) MAX_TOTAL_BYTES else MAX_PROJECTOR_BYTES)) { "Invalid model file size." }
            val sha = obj.getString("sha256")
            require(sha.matches(Regex("[a-f0-9]{64}"))) { "Invalid model digest." }
            val file = File(directory, fileName)
            require(file.canonicalFile.parentFile == directory.canonicalFile && file.isFile && file.length() == bytes) { "Missing or size-mismatched $fileName." }
            val metadata = readGgufIdentity(file, language)
            if (language) require(metadata.first == architecture && metadata.second == hiddenDim) { "Model architecture or embedding dimension differs from the pack manifest." }
            if (verifyHashes) {
                val digest = MessageDigest.getInstance("SHA-256")
                file.inputStream().buffered().use { stream ->
                    val buffer = ByteArray(128 * 1024)
                    while (true) {
                        checkStop()
                        val count = stream.read(buffer)
                        if (count < 0) break
                        digest.update(buffer, 0, count)
                    }
                }
                require(digest.digest().joinToString("") { "%02x".format(it) } == sha) { "Hash mismatch for $fileName." }
            }
            return ModelFile(fileName, bytes, sha)
        }
        val language = modelFile(json.getJSONObject("language"), true)
        val projector = json.optJSONObject("projector")?.let { modelFile(it, false) }
        require(projector?.fileName != language.fileName) { "Language model and projector must be separate files." }
        require(language.bytes + (projector?.bytes ?: 0) <= MAX_TOTAL_BYTES) { "Pack is too large." }
        val expected = setOfNotNull(MANIFEST_FILE, language.fileName, projector?.fileName)
        require(directory.listFiles()?.map { it.name }?.toSet() == expected) { "Chat pack contains missing or undeclared files." }
        return Installed(id, name, architecture, hiddenDim, language, projector, directory)
    }

    /** Read only bounded metadata, never tensors; model parsing itself is left to the native loader. */
    private fun readGgufIdentity(file: File, language: Boolean): Pair<String?, Int?> = RandomAccessFile(file, "r").use { input ->
        fun uint(): Long = Integer.toUnsignedLong(Integer.reverseBytes(input.readInt()))
        fun ulong(): Long = java.lang.Long.reverseBytes(input.readLong()).also { require(it >= 0) { "Invalid GGUF count." } }
        fun skip(bytes: Long) {
            require(bytes >= 0 && bytes <= file.length() - input.filePointer) { "Truncated GGUF metadata." }
            input.seek(input.filePointer + bytes)
        }
        fun string(max: Long = 1_048_576): String {
            val length = ulong()
            require(length <= max && length <= file.length() - input.filePointer) { "Invalid GGUF string." }
            return ByteArray(length.toInt()).also(input::readFully).toString(Charsets.UTF_8)
        }
        fun skipValue(type: Int, depth: Int = 0) {
            require(depth <= 2)
            when (type) {
                0, 1, 7 -> skip(1); 2, 3 -> skip(2); 4, 5, 6 -> skip(4); 10, 11, 12 -> skip(8)
                8 -> skip(ulong())
                9 -> {
                    val elementType = uint().toInt()
                    val count = ulong().also { require(it <= 2_000_000) }
                    repeat(count.toInt()) { skipValue(elementType, depth + 1) }
                }
                else -> error("Unsupported GGUF metadata type.")
            }
        }
        val magic = ByteArray(4).also(input::readFully)
        require(magic.contentEquals(byteArrayOf(0x47, 0x47, 0x55, 0x46)) && uint() in 2L..3L) { "Model file is not a supported GGUF." }
        require(ulong() > 0) { "GGUF contains no tensors." }
        val count = ulong().also { require(it in 1..2_000_000) }
        if (!language) return@use null to null
        var architecture: String? = null
        var dimension: Int? = null
        for (i in 0 until count) {
            require(input.filePointer <= 64L * 1024 * 1024) { "GGUF identity metadata is too large." }
            val key = string(1024)
            val type = uint().toInt()
            when {
                key == "general.architecture" && type == 8 -> architecture = string(100)
                key.endsWith(".embedding_length") && type in setOf(4, 5) -> dimension = uint().toInt()
                else -> skipValue(type)
            }
            if (architecture != null && dimension != null) break
        }
        require(architecture != null && dimension != null) { "GGUF architecture metadata is incomplete." }
        architecture to dimension
    }
}
