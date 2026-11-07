package com.onndetect

import android.net.Uri
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.lifecycle.ViewModelProvider
import androidx.test.platform.app.InstrumentationRegistry
import com.onndetect.chat.ChatModelStore
import com.onndetect.ui.screens.chat.ChatRagViewModel
import com.onndetect.ui.screens.modelmanager.ModelManagerViewModel
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Rule
import org.junit.Test

/** Reproduces the user's five-file ZIP through the real provider and native runtime. */
class ChatArchiveImportInstrumentationTest {
    @get:Rule val compose = createAndroidComposeRule<MainActivity>()

    @Test fun cancellingAProviderImportStopsTheActualJob() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assumeTrue("Do not alter an existing installation", ChatModelStore.inspect(context).installed == null)
        lateinit var vm: ModelManagerViewModel
        compose.activityRule.scenario.onActivity {
            vm = ViewModelProvider(it)[ModelManagerViewModel::class.java]
            vm.importChatZip(Uri.parse("content://com.onndetect.test.delayed-media/blocked-pipe"))
        }
        compose.waitUntil(5_000) { vm.uiState.value.isImporting }
        compose.activityRule.scenario.onActivity { vm.cancelImport() }
        compose.waitUntil(10_000) { !vm.uiState.value.isImporting }
        assertNull(vm.uiState.value.errorMessage)
        assertTrue(vm.uiState.value.successMessage.orEmpty().contains("stopped", ignoreCase = true))
        assertNull(ChatModelStore.inspect(context).installed)
    }

    @Test fun fiveFileQwenArchiveImportsAndAnswersWithoutReplacingItOnFailure() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        assumeTrue("Deploy the existing legacy ZIP for this real-weights test", File("/data/local/tmp/onn-legacy-chat-model.zip").isFile)
        assumeTrue("Do not replace an existing user's installation", ChatModelStore.inspect(context).installed == null)
        var installedByTest = false
        var session: ChatRagViewModel? = null
        try {
            val imported = ChatModelStore.importPackZip(context,
                Uri.parse("content://com.onndetect.test.delayed-media/legacy-chat-pack")).getOrThrow()
            installedByTest = true
            assertEquals(1024, imported.hiddenDim)
            assertEquals("57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf", imported.language.sha256)
            assertFalse("The known mismatched OCR projector must not enable visual answers", imported.supportsVision)
            assertTrue(ChatModelStore.inspect(context).isReady)

            val failedReplacement = ChatModelStore.importPackZip(context,
                Uri.parse("content://com.onndetect.test.delayed-media/bad-chat-pack"))
            assertTrue(failedReplacement.isFailure)
            assertEquals(imported.language.sha256, ChatModelStore.inspect(context).installed?.language?.sha256)
            assertTrue("A failed import must preserve the installed working model", ChatModelStore.inspect(context).isReady)

            compose.activityRule.scenario.onActivity {
                session = ViewModelProvider(it)[ChatRagViewModel::class.java]
                session!!.sendMessage("What is six plus seven? Reply only with the number.")
            }
            val vm = checkNotNull(session)
            compose.waitUntil(180_000) { !vm.uiState.value.isGenerating && vm.uiState.value.messages.size >= 2 }
            val answer = vm.uiState.value.messages.last()
            assertFalse(answer.text, answer.isError)
            assertEquals(imported.modelId, answer.runtimeModelId)
            assertEquals(imported.language.sha256, answer.runtimeLanguageSha256)
            assertTrue(answer.text, answer.text.contains("13"))
            assertFalse(answer.visualContextUsed)
        } finally {
            session?.let { vm -> compose.activityRule.scenario.onActivity { vm.newConversation() } }
            if (installedByTest) ChatModelStore.uninstall(context)
        }
    }
}
