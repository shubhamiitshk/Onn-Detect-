package com.onndetect.chat

import java.io.File
import java.io.InputStream
import java.util.Locale
import java.util.zip.ZipException
import java.util.zip.ZipInputStream

/** Normalizes model archives into a private candidate; never writes into the active installation. */
internal object ChatModelArchive {
    private const val MAX_ENTRIES = 256
    private const val MAX_GGUF_FILES = 8
    private const val MAX_BYTES = 3L * 1024 * 1024 * 1024
    private const val MAX_SIDECAR_BYTES = 16L * 1024 * 1024

    data class Contents(val manifestPresent: Boolean, val ignoredFiles: Int, val rawGguf: Boolean)

    fun extract(
        source: File,
        candidate: File,
        checkStop: () -> Unit = {},
        checkSpace: (Long) -> Unit = {},
    ): Contents {
        require(candidate.isDirectory && candidate.listFiles()?.isEmpty() == true) { "Model staging storage is not empty." }
        val magic = source.inputStream().use { input -> ByteArray(4).also { require(input.read(it) == 4) { "This file is empty or incomplete." } } }
        if (magic.contentEquals("GGUF".toByteArray(Charsets.US_ASCII))) {
            source.inputStream().use { input ->
                File(candidate, "language.gguf").outputStream().buffered().use { output ->
                    copyBounded(input, MAX_BYTES, checkStop, checkSpace) { bytes, count -> output.write(bytes, 0, count) }
                }
            }
            return Contents(false, 0, true)
        }
        require(magic[0] == 0x50.toByte() && magic[1] == 0x4b.toByte()) {
            "Choose a GGUF model or a ZIP containing GGUF files. Safetensors and other model formats need conversion first."
        }
        var count = 0
        var ignored = 0
        var ggufCount = 0
        var manifestPresent = false
        var expanded = 0L
        val paths = mutableSetOf<String>()
        val storedNames = mutableSetOf<String>()
        val modelParents = mutableSetOf<String>()
        try {
            ZipInputStream(source.inputStream().buffered()).use { zip ->
                while (true) {
                    checkStop()
                    val entry = zip.nextEntry ?: break
                    count++
                    require(count <= MAX_ENTRIES) { "This ZIP contains more than 256 entries. Choose a ZIP with one model and its optional vision file." }
                    val path = checkedPath(entry.name, entry.isDirectory)
                    require(paths.add(path.lowercase(Locale.US))) { "The ZIP contains the same path more than once: ${path.take(100)}. Recreate it without duplicate entries." }
                    if (entry.isDirectory) {
                        copyBounded(zip, 0, checkStop, {}) { _, _ -> }
                        zip.closeEntry()
                        continue
                    }
                    val name = path.substringAfterLast('/')
                    val macMetadata = path.startsWith("__MACOSX/") || name.startsWith("._")
                    val isManifest = !macMetadata && name.equals(ChatModelStore.MANIFEST_FILE, true)
                    val isGguf = !macMetadata && name.endsWith(".gguf", true)
                    val install = isManifest || isGguf
                    val storedName = if (isManifest) ChatModelStore.MANIFEST_FILE else name
                    if (install) {
                        require(ChatModelStore.safeFileName(storedName)) { "A model filename is unsupported. Use a short filename with letters, numbers, dots, hyphens or underscores." }
                        require(storedNames.add(storedName.lowercase(Locale.US))) { "Multiple folders contain the same model filename. Choose one model folder instead." }
                        modelParents.add(path.substringBeforeLast('/', ""))
                        if (isManifest) {
                            require(!manifestPresent) { "This ZIP contains multiple chat manifests. Choose one conversation model pack." }
                            manifestPresent = true
                        } else {
                            ggufCount++
                            require(ggufCount <= MAX_GGUF_FILES) { "This ZIP contains too many GGUF files. Include one language model and one optional vision projector." }
                        }
                    } else ignored++
                    val limit = when { isManifest -> 65_536L; isGguf -> MAX_BYTES; else -> MAX_SIDECAR_BYTES }
                    val destination = if (install) File(candidate, storedName) else null
                    destination?.outputStream()?.buffered().use { output ->
                        copyBounded(zip, limit, checkStop, { bytes ->
                            expanded += bytes
                            require(expanded <= MAX_BYTES) { "This ZIP expands beyond the 3 GB import limit." }
                            if (output != null) checkSpace(bytes)
                        }) { bytes, read -> output?.write(bytes, 0, read) }
                    }
                    zip.closeEntry()
                }
            }
        } catch (broken: ZipException) {
            throw IllegalArgumentException("This ZIP is damaged or uses an unsupported compression method. Extract it and choose the GGUF file, or create a new ZIP.", broken)
        }
        require(ggufCount > 0) { "No GGUF model was found in this ZIP. Choose the Qwen3.5 GGUF download, rather than a source-code or Safetensors archive." }
        require(modelParents.size == 1) { "Model files are spread across several folders. ZIP one folder containing the language GGUF and its optional vision GGUF together." }
        return Contents(manifestPresent, ignored, false)
    }

    internal fun checkedPath(name: String, directory: Boolean): String {
        require(name.length in 1..512 && !name.startsWith('/') && '\\' !in name && ':' !in name && name.none { it.code < 32 }) { "This ZIP contains an unsafe file path." }
        val trimmed = if (directory) name.removeSuffix("/") else name
        val segments = trimmed.split('/')
        require(segments.size <= 8 && segments.all { it.isNotEmpty() && it !in setOf(".", "..") && it.length <= 120 }) { "This ZIP contains an unsafe or deeply nested path." }
        return trimmed
    }

    private inline fun copyBounded(
        input: InputStream,
        limit: Long,
        checkStop: () -> Unit,
        checkSpace: (Long) -> Unit,
        write: (ByteArray, Int) -> Unit,
    ) {
        val bytes = ByteArray(128 * 1024)
        var total = 0L
        while (true) {
            checkStop()
            val read = input.read(bytes)
            if (read < 0) break
            total += read
            require(total <= limit) { "An archive entry exceeds its allowed size. Choose a ZIP with GGUF weights and small configuration files." }
            checkSpace(read.toLong())
            write(bytes, read)
        }
    }
}
