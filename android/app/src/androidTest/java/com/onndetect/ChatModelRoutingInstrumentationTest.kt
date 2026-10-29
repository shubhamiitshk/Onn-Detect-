package com.onndetect

import android.net.Uri
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.lifecycle.ViewModelProvider
import androidx.test.platform.app.InstrumentationRegistry
import com.onndetect.chat.ChatModelStore
import com.onndetect.data.local.ScanHistoryDatabase
import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.rag.ForensicRagEngine
import com.onndetect.ui.screens.chat.ChatMessage
import com.onndetect.ui.screens.chat.ChatRagViewModel
import kotlinx.coroutines.runBlocking
import org.junit.AfterClass
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Rule
import org.junit.Test
import java.io.File

/** Exercises the installed conversation pack through the same ViewModel/engine used by the app. */
class ChatModelRoutingInstrumentationTest {
    @get:Rule val compose = createAndroidComposeRule<MainActivity>()

    companion object {
        private var installedByTest = false
        private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

        @BeforeClass @JvmStatic fun installExistingLocalWeights() {
            assumeTrue("Deploy the existing local chat zip to run model routing tests", File("/data/local/tmp/onn-chat-model.zip").isFile)
            assumeTrue("Never replace a user's installed conversation model", ChatModelStore.inspect(context).installed == null)
            runBlocking {
                ChatModelStore.importPackZip(context, Uri.parse("content://com.onndetect.test.delayed-media/chat-pack")).getOrThrow()
            }
            installedByTest = true
            assertTrue(ChatModelStore.inspect(context).isReady)
        }

        @AfterClass @JvmStatic fun removeOnlyTestInstallation() {
            if (installedByTest) runBlocking { ChatModelStore.uninstall(context) }
        }
    }

    private fun session(): ChatRagViewModel {
        lateinit var vm: ChatRagViewModel
        compose.activityRule.scenario.onActivity { vm = ViewModelProvider(it)[ChatRagViewModel::class.java] }
        return vm
    }

    private fun completed(vm: ChatRagViewModel, seconds: Long = 180): ChatMessage {
        compose.waitUntil(seconds * 1000) { !vm.uiState.value.isGenerating && vm.uiState.value.messages.size >= 2 }
        val state = vm.uiState.value
        assertFalse(state.errorMessage.orEmpty(), state.canRetry)
        val answer = state.messages.last()
        assertFalse(answer.text, answer.isError)
        assertEquals("Qwen/Qwen3.5-0.8B", answer.runtimeModelId)
        assertEquals("57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf", answer.runtimeLanguageSha256)
        return answer
    }

    @Test fun normalConversationUsesActualWeightsWithoutAnyMedia() {
        val vm = session()
        try {
            compose.activityRule.scenario.onActivity { vm.sendMessage("What is six plus seven? Reply only with the number.") }
            val answer = completed(vm)
            assertTrue(answer.text, answer.text.contains("13"))
            assertFalse(answer.visualContextUsed)
            assertNull(vm.uiState.value.knowledgeBase)
        } finally { compose.activityRule.scenario.onActivity { vm.newConversation() } }
    }

    @Test fun imageQuestionsActuallyUseProjectorAndPixels() {
        val vm = session()
        try {
            for ((fixture, color) in listOf("chat-red" to "red", "chat-blue" to "blue")) {
                compose.activityRule.scenario.onActivity {
                    vm.newConversation()
                    vm.stageAttachment(Uri.parse("content://com.onndetect.test.delayed-media/$fixture"),
                        ForensicRagEngine.RagMediaKind.IMAGE, "$color.png", "image/png")
                    vm.sendMessage("What is the dominant color of the supplied image? Reply with one color word only.")
                }
                val answer = completed(vm)
                assertTrue("Image questions must pass actual pixels to the model", answer.visualContextUsed)
                assertTrue(answer.text, answer.text.contains(color, ignoreCase = true))
            }
        } finally { compose.activityRule.scenario.onActivity { vm.newConversation() } }
    }

    @Test fun savedForensicReportUsesModelWithoutInventingAValidatedVerdict() {
        val vm = session()
        val dao = ScanHistoryDatabase.create(compose.activity.applicationContext).scanHistoryDao()
        val id = "model-route-report-fixture"
        runBlocking {
            dao.insert(ScanHistoryEntity(id, System.currentTimeMillis(), "AUDIO_ONLY", .8f, true, 0f, 150,
                status = "INCONCLUSIVE", verdict = "SYNTHETIC", modelValidated = false,
                reason = "Fixture: device validation unavailable.",
                evidenceJson = """{"analyzedDurationMs":4000,"totalDurationMs":60000,"timestampsMs":[28000]}"""))
        }
        try {
            compose.waitUntil(10_000) { vm.uiState.value.recentReports.any { it.id == id } }
            compose.activityRule.scenario.onActivity {
                vm.stageScanReport(id)
                vm.sendMessage("Repeat the recorded assessment label only.")
            }
            val answer = completed(vm)
            assertEquals("INCONCLUSIVE", vm.uiState.value.knowledgeBase!!.headlineVerdict)
            assertEquals(60_000L, vm.uiState.value.knowledgeBase!!.durationMs)
            assertFalse(answer.visualContextUsed)
            assertTrue(answer.text, answer.text.contains("INCONCLUSIVE", ignoreCase = true))
        } finally {
            compose.activityRule.scenario.onActivity { vm.newConversation() }
            runBlocking { dao.deleteById(id) }
        }
    }
}
