package com.onndetect

import android.content.ContentValues
import android.os.Build
import android.provider.MediaStore
import androidx.activity.compose.setContent
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.captureToImage
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onRoot
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.unit.Density
import com.onndetect.chat.ChatModelStore
import com.onndetect.detection.visual.VlmPackManager
import com.onndetect.ui.screens.modelmanager.ModelManagerContent
import com.onndetect.ui.screens.modelmanager.ModelManagerUiState
import com.onndetect.ui.theme.OnnDetectTheme
import java.io.File
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

/** Rendered UI fixtures are for layout inspection, not native inference evidence. */
class ModelManagerExperienceTest {
    @get:Rule val compose = createAndroidComposeRule<MainActivity>()

    private fun emptyState() = ModelManagerUiState(
        deviceProfile = VlmPackManager.DeviceHardwareProfile(7.1f, 2.3f, "arm64-v8a", "fixture"),
        installedPack = null,
        chatStatus = ChatModelStore.Status(null, false, "No model added.", true),
    )

    private fun show(state: ModelManagerUiState = emptyState(), largeText: Boolean = false,
                     choose: () -> Unit = {}, cancel: () -> Unit = {}, chat: () -> Unit = {}) {
        compose.activityRule.scenario.onActivity { activity ->
            activity.setContent { OnnDetectTheme {
                val density = LocalDensity.current
                CompositionLocalProvider(LocalDensity provides Density(density.density, if (largeText) 2f else density.fontScale)) {
                    ModelManagerContent(state, onBack = {}, onChooseModel = choose, onOpenChat = chat,
                        onRemove = {}, onCancelImport = cancel, onDismissMessage = {}, onRefresh = {},
                        onResearchImport = {}, onAllowCustomKey = {})
                }
            } }
        }
    }

    @Test fun emptySetupShowsTheImportActionWithoutExpandedTechnicalCards() {
        var chosen = false
        show(choose = { chosen = true })
        compose.onNodeWithTag("chat_model_import").assertIsDisplayed().performClick()
        compose.runOnIdle { assertTrue(chosen) }
        compose.onNodeWithText("How the models work together").assertDoesNotExist()
        savePreview("models-empty.png")
    }

    @Test fun failureNamesTheFileAndOffersAnotherSelection() {
        var chosen = false
        show(emptyState().copy(selectedFileName = "qwen35-0.8b-v1.zip",
            errorMessage = "This ZIP contains more than one language model. Keep one Qwen GGUF in the ZIP."), choose = { chosen = true })
        compose.onNodeWithTag("chat_model_notice").performScrollTo().assertIsDisplayed()
        compose.onNodeWithText("Choose another file").performScrollTo().performClick()
        compose.runOnIdle { assertTrue(chosen) }
        savePreview("models-error.png")
    }

    @Test fun importingHasVisibleProgressAndAStopAction() {
        var stopped = false
        show(emptyState().copy(isImporting = true, selectedFileName = "qwen35-0.8b-v1.zip",
            importProgressText = "Checking model files…"), cancel = { stopped = true })
        compose.onNodeWithTag("chat_model_progress").assertIsDisplayed()
        compose.onNodeWithTag("chat_model_cancel").performClick()
        compose.runOnIdle { assertTrue(stopped) }
        savePreview("models-importing.png")
    }

    @Test fun installedModelOffersChatDirectly() {
        val file = ChatModelStore.ModelFile("language.gguf", 563036064, "0".repeat(64))
        val installed = ChatModelStore.Installed("fixture/preview", "Qwen 3.5 · 0.8B", "qwen35", 1024,
            file, null, File(compose.activity.cacheDir, "render-fixture"))
        var opened = false
        show(emptyState().copy(chatStatus = ChatModelStore.Status(installed, true, "Ready", true)), chat = { opened = true })
        compose.onNodeWithTag("chat_model_open_chat").assertIsDisplayed().performClick()
        compose.runOnIdle { assertTrue(opened) }
        savePreview("models-ready.png")
    }

    @Test fun setupRemainsUsableWithTwiceTheTextSize() {
        show(largeText = true)
        compose.onNodeWithTag("chat_model_import").performScrollTo().assertIsDisplayed()
        savePreview("models-large-text.png")
    }

    private fun savePreview(name: String) {
        compose.waitForIdle()
        if (Build.VERSION.SDK_INT < 29) return
        val bitmap = compose.onRoot().captureToImage().asAndroidBitmap()
        val resolver = compose.activity.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/ONN-Import-UI-Preview")
        }
        val uri = checkNotNull(resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values))
        checkNotNull(resolver.openOutputStream(uri)).use { bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it) }
    }
}
