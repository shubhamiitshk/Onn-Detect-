package com.onndetect

import android.net.Uri
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.lifecycle.ViewModelProvider
import com.onndetect.rag.ForensicRagEngine
import com.onndetect.ui.screens.chat.AttachedMedia
import com.onndetect.ui.screens.chat.ChatRagViewModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

/** Real Hilt ViewModel + provider importer + retrieval + streaming, using synthetic document content. */
class ChatSessionInstrumentationTest {
    @get:Rule val compose = createAndroidComposeRule<MainActivity>()

    private fun session(): ChatRagViewModel {
        lateinit var vm: ChatRagViewModel
        compose.activityRule.scenario.onActivity { activity ->
            vm = ViewModelProvider(activity)[ChatRagViewModel::class.java]
        }
        return vm
    }

    @Test fun attachedTextTraversesImporterAndProducesGroundedReply() {
        val vm = session()
        try {
            compose.activityRule.scenario.onActivity {
                vm.stageAttachment(Uri.parse("content://com.onndetect.test.delayed-media/chat-text"),
                    ForensicRagEngine.RagMediaKind.DOCUMENT, "invoice.txt", "text/plain")
                vm.sendMessage("What is the invoice total and payment status?")
            }
            compose.waitUntil(30_000) { !vm.uiState.value.isGenerating && vm.uiState.value.messages.size >= 2 }
            val state = vm.uiState.value
            assertFalse(state.errorMessage.orEmpty(), state.canRetry)
            assertTrue(state.knowledgeBase!!.chunks.any { it.visualDescription.contains("₹250") })
            val reply = state.messages.last()
            assertFalse(reply.isStreaming)
            assertFalse(reply.isError)
            assertTrue(reply.text.contains("₹250"))
            assertTrue(reply.text.contains("pending"))
            assertEquals("Built-in evidence reply", reply.engineBadge)
        } finally { compose.activityRule.scenario.onActivity { vm.newConversation() } }
    }

    @Test fun busySendIsIgnoredAndNewConversationClearsAttachmentsAndContext() {
        val vm = session()
        try {
            compose.activityRule.scenario.onActivity {
                vm.sendMessage("Hello")
                vm.sendMessage("This duplicate must be ignored while busy")
                assertEquals(1, vm.uiState.value.messages.count { it.isUser })
                vm.cancelGeneration()
                assertFalse(vm.uiState.value.isGenerating)
                vm.newConversation()
                vm.stageAttachments((1..5).map { AttachedMedia(Uri.parse("content://preview/$it"), ForensicRagEngine.RagMediaKind.DOCUMENT, "$it.txt") })
                assertEquals(4, vm.uiState.value.stagedAttachments.size)
                vm.newConversation()
                assertTrue(vm.uiState.value.messages.isEmpty())
                assertTrue(vm.uiState.value.stagedAttachments.isEmpty())
                assertEquals(null, vm.uiState.value.knowledgeBase)
                vm.sendMessage("Hello")
            }
            compose.waitUntil(15_000) { !vm.uiState.value.isGenerating && vm.uiState.value.messages.size == 2 }
            assertEquals(1, vm.uiState.value.messages.count { it.isUser })
            assertFalse(vm.uiState.value.messages.last().isError)
        } finally { compose.activityRule.scenario.onActivity { vm.newConversation() } }
    }
}
