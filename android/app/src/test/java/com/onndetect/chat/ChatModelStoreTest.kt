package com.onndetect.chat

import java.io.File
import java.io.RandomAccessFile
import java.security.MessageDigest
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/** Synthetic headers test import contracts, not whether fabricated tensors can run inference. */
class ChatModelStoreTest {
    @get:Rule val temporary = TemporaryFolder()

    private fun model(file: File, architecture: String = "qwen35", dimension: Int = 1024) {
        RandomAccessFile(file, "rw").use { out ->
            fun int(value: Int) = out.writeInt(Integer.reverseBytes(value))
            fun long(value: Long) = out.writeLong(java.lang.Long.reverseBytes(value))
            fun string(value: String) { val bytes = value.toByteArray(); long(bytes.size.toLong()); out.write(bytes) }
            out.writeBytes("GGUF"); int(3); long(1); long(2)
            string("general.architecture"); int(8); string(architecture)
            string("$architecture.embedding_length"); int(4); int(dimension)
            out.setLength(50L * 1024 * 1024)
        }
    }

    private fun fixture(): Pair<File, JSONObject> {
        val folder = temporary.newFolder()
        val model = File(folder, "language.gguf").also { model(it) }
        val digest = MessageDigest.getInstance("SHA-256")
        model.inputStream().use { input -> val buffer = ByteArray(65536); while (true) { val n=input.read(buffer); if(n<0) break; digest.update(buffer,0,n) } }
        val json = JSONObject().put("schema_version",1).put("purpose","conversation")
            .put("model_id","local/test-qwen").put("display_name","Header fixture")
            .put("architecture","qwen35").put("hidden_dim",1024)
            .put("language", JSONObject().put("file","language.gguf").put("bytes",model.length())
                .put("sha256",digest.digest().joinToString("") { "%02x".format(it) }))
        return folder to json
    }

    private fun save(folder: File, json: JSONObject) { File(folder, ChatModelStore.MANIFEST_FILE).writeText(json.toString()) }

    @Test fun validConversationContractNeedsNoForensicHead() {
        val (folder,json)=fixture(); save(folder,json)
        val contract=ChatModelStore.readManifest(folder,true)
        assertEquals(1024,contract.hiddenDim)
        assertFalse(contract.supportsVision)
        assertEquals("local/test-qwen",contract.modelId)
    }

    @Test(expected=IllegalArgumentException::class) fun changedBytesFailEvenWhenLengthMatches() {
        val (folder,json)=fixture(); save(folder,json)
        RandomAccessFile(File(folder,"language.gguf"),"rw").use { it.seek(it.length()-1); it.write(3) }
        ChatModelStore.readManifest(folder,true)
    }

    @Test(expected=IllegalArgumentException::class) fun declaredDimensionCannotOverrideActualGgufMetadata() {
        val (folder,json)=fixture(); json.put("hidden_dim",2048); save(folder,json)
        ChatModelStore.readManifest(folder,false)
    }

    @Test(expected=IllegalArgumentException::class) fun unmanifestedFilesAreRejected() {
        val (folder,json)=fixture(); save(folder,json); File(folder,"injected.txt").writeText("undeclared")
        ChatModelStore.readManifest(folder,false)
    }

    @Test(expected=IllegalArgumentException::class) fun manifestPathCannotEscapePackDirectory() {
        val (folder,json)=fixture(); json.getJSONObject("language").put("file","../language.gguf"); save(folder,json)
        ChatModelStore.readManifest(folder,false)
    }
}
