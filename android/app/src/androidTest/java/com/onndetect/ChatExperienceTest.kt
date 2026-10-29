package com.onndetect

import android.graphics.Bitmap
import android.net.Uri
import androidx.activity.compose.setContent
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.Modifier
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertExists
import androidx.compose.ui.test.captureToImage
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onRoot
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.unit.Density
import com.onndetect.rag.ForensicRagEngine
import com.onndetect.ui.screens.chat.AttachedMedia
import com.onndetect.ui.screens.chat.ChatMessage
import com.onndetect.ui.screens.chat.ChatRagContent
import com.onndetect.ui.screens.chat.ChatRagUiState
import com.onndetect.ui.theme.OnnDetectTheme
import com.onndetect.ui.components.LiquidGlassScaffold
import com.onndetect.ui.components.GlassBottomTab
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import java.io.File

/** Real Compose interactions; preview messages are synthetic fixtures, not detector evidence. */
class ChatExperienceTest {
    @get:Rule val compose = createAndroidComposeRule<MainActivity>()

    private fun show(
        state: ChatRagUiState = ChatRagUiState(),
        largeText: Boolean = false,
        onSend: (String) -> Unit = {},
        onStop: () -> Unit = {},
        onRemove: (Uri) -> Unit = {},
        onRetry: () -> Unit = {},
    ) {
        compose.activityRule.scenario.onActivity { activity ->
            activity.setContent {
                OnnDetectTheme {
                    val density = LocalDensity.current
                    CompositionLocalProvider(LocalDensity provides Density(density.density, if (largeText) 2f else density.fontScale)) {
                        LiquidGlassScaffold(activeTab = GlassBottomTab.CHAT_RAG, onSelectTab = {}) { padding ->
                          ChatRagContent(
                            uiState = state, onSend = onSend, onStop = onStop,
                            onAttach = {}, onRemoveAttachment = onRemove, onNewConversation = {},
                            onRetry = onRetry, onDismissError = {}, onSelectChunk = {},
                            onSelectTimestamp = {}, onOpenModels = {}, onCopy = {},
                            modifier = Modifier.fillMaxSize().padding(padding),
                        )
                        }
                    }
                }
            }
        }
    }

    @Test fun composerDisablesEmptySendAndSubmitsTypedMessage() {
        var submitted: String? = null
        show(onSend = { submitted = it })
        compose.onNodeWithTag("chat_send").assertIsNotEnabled()
        compose.onNodeWithTag("chat_input").performTextInput("What evidence suggests an image is AI generated?")
        compose.onNodeWithTag("chat_send").assertIsEnabled().performClick()
        compose.runOnIdle { assertEquals("What evidence suggests an image is AI generated?", submitted) }
    }

    @Test fun streamingShowsRunningIndicatorAndStopControl() {
        var stopped = false
        show(ChatRagUiState(
            isGenerating = true, streamingStatus = "Writing response…",
            messages = listOf(ChatMessage("response", false, "A detector score needs", isStreaming = true)),
        ), onStop = { stopped = true })
        compose.onNodeWithTag("chat_working").assertExists()
        compose.onNodeWithTag("chat_stop").assertIsDisplayed().performClick()
        compose.runOnIdle { assertTrue(stopped) }
    }

    @Test fun attachmentOnlyMessageCanSendAndAttachmentCanBeRemoved() {
        val first = AttachedMedia(Uri.parse("content://preview/image"), ForensicRagEngine.RagMediaKind.IMAGE, "photo.jpg", "image/jpeg")
        val second = AttachedMedia(Uri.parse("content://preview/document"), ForensicRagEngine.RagMediaKind.DOCUMENT, "notes.pdf", "application/pdf")
        var sent = false
        var removed: Uri? = null
        show(ChatRagUiState(stagedAttachments = listOf(first, second)), onSend = { sent = true }, onRemove = { removed = it })
        compose.onNodeWithText("photo.jpg").assertExists()
        compose.onNodeWithContentDescription("Remove photo.jpg").performClick()
        compose.runOnIdle { assertEquals(first.uri, removed) }
        compose.onNodeWithTag("chat_send").assertIsEnabled().performClick()
        compose.runOnIdle { assertTrue(sent) }
    }

    @Test fun failedResponseProvidesRetryAction() {
        var retried = false
        show(ChatRagUiState(
            canRetry = true,
            messages = listOf(ChatMessage("failed", false, "Couldn't read this file. Please try again.", isError = true)),
        ), onRetry = { retried = true })
        compose.onNodeWithTag("chat_retry").performScrollTo().performClick()
        compose.runOnIdle { assertTrue(retried) }
    }

    @Test fun composerControlsRemainReachableWithLargeText() {
        show(largeText = true)
        compose.onNodeWithTag("chat_attach").assertIsDisplayed()
        compose.onNodeWithTag("chat_input").assertIsDisplayed()
        compose.onNodeWithTag("chat_send").assertIsDisplayed()
        compose.onNodeWithTag("chat_new").assertIsDisplayed()
    }

    @Test fun saveEmptyChatDesignPreview() {
        show()
        savePreview("chat-empty.png")
    }

    @Test fun saveStreamingConversationDesignPreview() {
        val attachment = AttachedMedia(Uri.parse("content://preview/photo"), ForensicRagEngine.RagMediaKind.IMAGE, "street-photo.jpg", "image/jpeg", "2.4 MB")
        show(ChatRagUiState(
            isGenerating = true,
            streamingStatus = "Writing response…",
            messages = listOf(
                ChatMessage("question", true, "Does this look AI generated?", attachment = attachment, attachments = listOf(attachment)),
                ChatMessage("response", false,
                    "**The assessment is inconclusive.**\n\nThe available signals need to be considered together:\n\n• Fine texture and image structure\n• The semantic detector's result\n• Compression and image quality\n\nA detector score alone cannot prove where a photo came from.",
                    isStreaming = true),
            ),
        ))
        savePreview("chat-streaming.png")
    }

    private fun savePreview(name: String) {
        compose.waitForIdle()
        val bitmap = compose.onRoot().captureToImage().asAndroidBitmap()
        val directory = File(compose.activity.getExternalFilesDir(null), "chat-previews").apply { mkdirs() }
        File(directory, name).outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
    }
}
