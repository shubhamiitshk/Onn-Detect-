package com.onndetect

import android.content.ContentValues
import android.accessibilityservice.AccessibilityServiceInfo
import android.graphics.Rect
import android.os.Build
import android.os.SystemClock
import android.provider.MediaStore
import android.view.MotionEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo
import androidx.activity.compose.setContent
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import com.onndetect.ui.components.GlassBottomTab
import com.onndetect.ui.components.LiquidGlassScaffold
import com.onndetect.ui.screens.chat.ChatRagContent
import com.onndetect.ui.screens.chat.ChatRagUiState
import com.onndetect.ui.theme.OnnDetectTheme
import androidx.compose.ui.Modifier
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.*
import org.junit.Test

/** Uses Android input directly: Compose's semantics test harness intercepts native IME sessions. */
class ChatKeyboardInstrumentationTest {
    @Test fun realKeyboardKeepsComposerAboveItAndHidesTheBottomDock() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val automation = instrumentation.uiAutomation
        automation.serviceInfo = automation.serviceInfo.apply {
            flags = flags or AccessibilityServiceInfo.FLAG_RETRIEVE_INTERACTIVE_WINDOWS or
                AccessibilityServiceInfo.FLAG_INCLUDE_NOT_IMPORTANT_VIEWS or AccessibilityServiceInfo.FLAG_REPORT_VIEW_IDS
        }
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity { activity ->
                activity.setContent { OnnDetectTheme {
                    LiquidGlassScaffold(GlassBottomTab.CHAT_RAG, {}) { padding ->
                        ChatRagContent(ChatRagUiState(), onSend = {}, onStop = {}, onAttach = {},
                            onRemoveAttachment = {}, onNewConversation = {}, onRetry = {}, onDismissError = {},
                            onSelectChunk = {}, onSelectTimestamp = {}, onOpenModels = {}, onCopy = {},
                            modifier = Modifier.fillMaxSize().padding(padding))
                    }
                } }
            }
            fun appRoot() = automation.windows.asSequence().filter { it.type == AccessibilityWindowInfo.TYPE_APPLICATION }
                .mapNotNull { it.root }.firstOrNull { it.packageName?.toString() == instrumentation.targetContext.packageName }
            fun node(description: String) = find(appRoot()) { it.contentDescription?.toString() == description ||
                (description == "Message" && it.isEditable) }
            waitUntil { node("Message") != null }
            val input = Rect().also { checkNotNull(node("Message")).getBoundsInScreen(it) }
            assertFalse(input.isEmpty)
            val now = SystemClock.uptimeMillis()
            MotionEvent.obtain(now, now, MotionEvent.ACTION_DOWN, input.exactCenterX(), input.exactCenterY(), 0).let {
                try { instrumentation.sendPointerSync(it) } finally { it.recycle() }
            }
            MotionEvent.obtain(now, now + 40, MotionEvent.ACTION_UP, input.exactCenterX(), input.exactCenterY(), 0).let {
                try { instrumentation.sendPointerSync(it) } finally { it.recycle() }
            }
            waitUntil { node("Message")?.isFocused == true }
            val imeVisible = AtomicBoolean(false)
            val keyboardTop = AtomicInteger(0)
            waitUntil {
                scenario.onActivity { activity ->
                    val view = activity.window.decorView
                    val insets = ViewCompat.getRootWindowInsets(view)
                    imeVisible.set(insets?.isVisible(WindowInsetsCompat.Type.ime()) == true)
                    val origin = IntArray(2).also { view.getLocationOnScreen(it) }
                    keyboardTop.set(origin[1] + view.height - (insets?.getInsets(WindowInsetsCompat.Type.ime())?.bottom ?: 0))
                }
                imeVisible.get()
            }
            instrumentation.sendStringSync("Hello ONN")
            waitUntil { node("Message")?.text?.toString().orEmpty().contains("Hello ONN") }
            waitUntil {
                appRoot() != null && find(appRoot()) { it.text?.toString() == "Voice Vault" && it.isVisibleToUser } == null &&
                    node("Send message")?.let { button -> Rect().also(button::getBoundsInScreen).bottom <= keyboardTop.get() } == true
            }
            val send = Rect().also { checkNotNull(node("Send message")).getBoundsInScreen(it) }
            val typedInput = Rect().also { checkNotNull(node("Message")).getBoundsInScreen(it) }
            assertFalse(send.isEmpty)
            assertTrue("Send must remain above the keyboard: $send vs ${keyboardTop.get()}", send.bottom <= keyboardTop.get())
            assertTrue("The input must remain above the keyboard", typedInput.bottom <= keyboardTop.get())
            assertTrue("Actual Android input must reach the composer", node("Message")?.text?.toString().orEmpty().contains("Hello ONN"))

            if (Build.VERSION.SDK_INT >= 29) {
                val bitmap = checkNotNull(automation.takeScreenshot())
                val resolver = instrumentation.targetContext.contentResolver
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, "chat-keyboard.png")
                    put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/ONN-Import-UI-Preview")
                }
                val uri = checkNotNull(resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values))
                checkNotNull(resolver.openOutputStream(uri)).use { bitmap.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it) }
            }
            scenario.onActivity { WindowInsetsControllerCompat(it.window, it.window.decorView).hide(WindowInsetsCompat.Type.ime()) }
        }
    }

    private fun find(root: AccessibilityNodeInfo?, predicate: (AccessibilityNodeInfo) -> Boolean): AccessibilityNodeInfo? {
        if (root == null) return null
        if (predicate(root)) return root
        for (i in 0 until root.childCount) find(root.getChild(i), predicate)?.let { return it }
        return null
    }

    private fun waitUntil(condition: () -> Boolean) {
        val end = SystemClock.uptimeMillis() + 15_000
        while (!condition() && SystemClock.uptimeMillis() < end) SystemClock.sleep(50)
        assertTrue("Android UI condition did not settle within 15 seconds", condition())
    }
}
