package com.onndetect.chat

import java.io.File
import java.io.RandomAccessFile
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/** Archive compatibility and integrity regressions; header fixtures are not runnable neural models. */
class ChatModelArchiveTest {
    @get:Rule val temporary = TemporaryFolder()

    private fun model(name: String = "source.gguf", architecture: String = "qwen35", display: String = "Header fixture"): File {
        val file = File(temporary.newFolder(), name)
        RandomAccessFile(file, "rw").use { out ->
            fun int(value: Int) = out.writeInt(Integer.reverseBytes(value))
            fun long(value: Long) = out.writeLong(java.lang.Long.reverseBytes(value))
            fun string(value: String) { val bytes = value.toByteArray(); long(bytes.size.toLong()); out.write(bytes) }
            out.writeBytes("GGUF"); int(3); long(1); long(3)
            string("general.architecture"); int(8); string(architecture)
            string("general.name"); int(8); string(display)
            string(if (architecture == "clip") "clip.vision.projection_dim" else "$architecture.embedding_length"); int(4); int(1024)
            out.setLength(if (architecture == "clip") 1024L * 1024 else 50L * 1024 * 1024)
        }
        return file
    }

    private fun archive(vararg entries: Pair<String, File?>): File {
        val zip = File(temporary.newFolder(), "download.zip")
        ZipOutputStream(zip.outputStream()).use { out ->
            entries.forEach { (name, file) ->
                out.putNextEntry(ZipEntry(name))
                if (file != null) file.inputStream().use { it.copyTo(out) }
                out.closeEntry()
            }
        }
        return zip
    }

    private fun text(name: String, content: String = "archive notes") = File(temporary.newFolder(), name).apply { writeText(content) }
    private fun prepare(zip: File) = ChatModelStore.prepareImport(zip, temporary.newFolder())

    @Test fun wrapperFolderAndSidecarsDoNotPreventNormalModelImport() {
        val weights = model()
        val installed = prepare(archive("Qwen/" to null, "Qwen/model.gguf" to weights,
            "Qwen/README.md" to text("readme"), "Qwen/LICENSE" to text("license"),
            "Qwen/config.json" to text("config"), "Qwen/tokenizer.json" to text("tokenizer")))
        assertEquals("qwen35", installed.architecture)
        assertEquals(1024, installed.hiddenDim)
        assertFalse(installed.supportsVision)
        assertTrue(installed.importNotes.isNotEmpty())
        assertEquals(setOf("model.gguf", ChatModelStore.MANIFEST_FILE), installed.directory.listFiles()!!.map { it.name }.toSet())
        assertEquals(weights.length(), File(installed.directory, installed.language.fileName).length())
    }

    @Test fun fiveFileForensicArchiveImportsLanguageWithoutRequiringItsHead() {
        val installed = prepare(archive("forensic_vlm_policy.json" to text("policy", "{\"hidden_dim\":8}"),
            "publisher_key.hex" to text("key"), "signed_manifest.json" to text("signed", "{\"backbone\":\"SmolVLM\"}"),
            "qwen-model.gguf" to model(), "qwen-mmproj.gguf" to model("vision.gguf", "clip", "Qwen merged_en_ocr")))
        assertEquals("qwen35", installed.architecture)
        assertNotNull(installed.projector)
        assertFalse(installed.supportsVision)
        assertTrue(installed.visionUnavailableReason!!.contains("OCR"))
        assertFalse(File(installed.directory, "forensic_vlm_policy.json").exists())
    }

    @Test fun standaloneGgufUsesInspectedMetadataAndCreatesByteBoundIdentity() {
        val installed = prepare(model("arbitrary-download-name.bin"))
        assertEquals("Header fixture", installed.displayName)
        assertEquals(64, installed.language.sha256.length)
        assertFalse(installed.supportsVision)
        assertTrue(File(installed.directory, ChatModelStore.MANIFEST_FILE).isFile)
    }

    @Test fun suppliedConversationManifestSurvivesWrappingAndNotes() {
        val first = prepare(model())
        val zip = archive("model/" to null, "model/language.gguf" to File(first.directory, first.language.fileName),
            "model/chat_model.json" to File(first.directory, ChatModelStore.MANIFEST_FILE), "model/README.md" to text("readme"))
        val imported = prepare(zip)
        assertEquals(first.language.sha256, imported.language.sha256)
        assertEquals(first.modelId, imported.modelId)
    }

    @Test fun invalidProvidedManifestIsNotRegeneratedOrAllowedToReplacePreviousModel() {
        val previous = prepare(model())
        val before = File(previous.directory, ChatModelStore.MANIFEST_FILE).readText()
        val manifest = JSONObject(before).apply { getJSONObject("language").put("sha256", "0".repeat(64)) }
        val zip = archive("language.gguf" to File(previous.directory, previous.language.fileName),
            "chat_model.json" to text("manifest", manifest.toString()))
        try { prepare(zip); fail("A supplied mismatched hash must fail") }
        catch (expected: IllegalArgumentException) { assertTrue(expected.message.orEmpty().contains("Hash mismatch")) }
        assertEquals(before, File(previous.directory, ChatModelStore.MANIFEST_FILE).readText())
        assertEquals(previous.language.sha256, ChatModelStore.readManifest(previous.directory, true).language.sha256)
    }

    @Test fun multipleLanguageModelsRequireAnExplicitChoice() {
        try { prepare(archive("first.gguf" to model(), "second.gguf" to model())); fail("Must not choose a model by zip order") }
        catch (expected: IllegalArgumentException) { assertTrue(expected.message.orEmpty().contains("multiple language models")) }
    }

    @Test fun malformedProvidedProjectorIsNotSilentlyConvertedToTextOnly() {
        val first = prepare(model())
        val manifest = JSONObject(File(first.directory, ChatModelStore.MANIFEST_FILE).readText())
            .put("projector", "vision.gguf")
        val zip = archive("language.gguf" to File(first.directory, first.language.fileName),
            "chat_model.json" to text("manifest", manifest.toString()))
        try { prepare(zip); fail("A malformed supplied projector declaration must fail") }
        catch (expected: IllegalArgumentException) { assertTrue(expected.message.orEmpty().contains("projector entry")) }
    }

    @Test fun multipleProjectorsRequireAnExplicitChoice() {
        try { prepare(archive("language.gguf" to model(), "first.gguf" to model("v1.gguf", "clip"), "second.gguf" to model("v2.gguf", "clip"))); fail("Must not choose a projector by zip order") }
        catch (expected: IllegalArgumentException) { assertTrue(expected.message.orEmpty().contains("several vision projectors")) }
    }

    @Test fun duplicateCaseInsensitivePathsAreRejectedSeparatelyFromEntryCount() {
        try { prepare(archive("model.gguf" to model(), "MODEL.GGUF" to model())); fail("Duplicate paths must be rejected") }
        catch (expected: IllegalArgumentException) { assertTrue(expected.message.orEmpty().contains("same path more than once")) }
    }

    @Test fun traversalInAnIgnoredSidecarIsStillRejected() {
        try { prepare(archive("model.gguf" to model(), "../readme.txt" to text("notes"))); fail("Even ignored entries must have safe paths") }
        catch (expected: IllegalArgumentException) { assertTrue(expected.message.orEmpty().contains("unsafe")) }
    }

    @Test fun unsupportedSourceArchiveHasAnActionableError() {
        try { prepare(archive("README.md" to text("readme"), "config.json" to text("config"))); fail("Missing GGUF must fail") }
        catch (expected: IllegalArgumentException) { assertTrue(expected.message.orEmpty().contains("No GGUF model")) }
    }

    @Test fun unverifiedProjectorPairRemainsAvailableForTextWithoutPretendingVisionWasTested() {
        val installed = prepare(archive("language.gguf" to model(), "vision.gguf" to model("vision.gguf", "clip", "Qwen base vision")))
        assertNotNull(installed.projector)
        assertFalse(installed.supportsVision)
        assertTrue(installed.visionUnavailableReason!!.contains("has not been verified"))
    }
}
