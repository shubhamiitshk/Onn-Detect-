package com.onndetect

import androidx.activity.compose.setContent
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.lifecycle.ViewModelProvider
import androidx.navigation.compose.rememberNavController
import com.onndetect.data.local.ScanHistoryDatabase
import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.rag.ForensicRagEngine
import com.onndetect.ui.navigation.ONNNavGraph
import com.onndetect.ui.navigation.Routes
import com.onndetect.ui.screens.chat.ChatIntent
import com.onndetect.ui.screens.chat.ChatRagViewModel
import com.onndetect.ui.theme.OnnDetectTheme
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

/** Saved reports are evidence records, not provider files or newly inferred authenticity. */
class ChatReportFlowInstrumentationTest {
    @get:Rule val compose = createAndroidComposeRule<MainActivity>()
    private val dao get() = ScanHistoryDatabase.create(compose.activity.applicationContext).scanHistoryDao()

    private fun record(id: String, mode: String) = ScanHistoryEntity(
        id, System.currentTimeMillis(), mode, 0.8f, true, 0f, 125,
        status = "INCONCLUSIVE", verdict = "SYNTHETIC", modelValidated = false,
        reason = "Fixture: independent device validation unavailable.",
        evidenceJson = """{"analyzedDurationMs":4000,"totalDurationMs":60000,"timestampsMs":[28000]}""",
    )

    private fun session(): ChatRagViewModel {
        lateinit var vm: ChatRagViewModel
        compose.activityRule.scenario.onActivity { vm = ViewModelProvider(it)[ChatRagViewModel::class.java] }
        return vm
    }

    @Test fun explicitSavedReportKeepsRecordedUncertainty() {
        val id = "report-audio-fixture"
        runBlocking { dao.insert(record(id, "AUDIO_ONLY")) }
        val vm = session()
        try {
            compose.activityRule.scenario.onActivity { vm.loadFromScanHistory(id) }
            compose.waitUntil(10_000) { !vm.uiState.value.isIndexing && vm.uiState.value.knowledgeBase != null }
            assertEquals(ChatIntent.REPORT, vm.uiState.value.intent)
            assertEquals("INCONCLUSIVE", vm.uiState.value.knowledgeBase!!.headlineVerdict)
            assertEquals(ForensicRagEngine.RagMediaKind.SCAN_HISTORY_RECORD, vm.uiState.value.knowledgeBase!!.mediaKind)
        } finally {
            compose.activityRule.scenario.onActivity { vm.newConversation() }
            runBlocking { dao.deleteById(id) }
        }
    }

    @Test fun severalReportsStageWithoutProviderUrisAndNormalChatClearsSources() {
        val ids = listOf("report-image-fixture", "report-video-fixture")
        runBlocking { ids.forEachIndexed { index, id -> dao.insert(record(id, if (index == 0) "IMAGE_ONLY" else "VIDEO_ONLY")) } }
        val vm = session()
        try {
            compose.waitUntil(10_000) { vm.uiState.value.recentReports.map { it.id }.containsAll(ids) }
            compose.activityRule.scenario.onActivity {
                ids.forEach(vm::stageScanReport)
                assertEquals(ids, vm.uiState.value.stagedAttachments.map { it.scanId })
                assertTrue(vm.uiState.value.stagedAttachments.all { it.uri.scheme == "onn-report" })
                assertEquals(ChatIntent.REPORT, vm.uiState.value.intent)
                vm.setChatIntent(ChatIntent.NORMAL)
                assertTrue(vm.uiState.value.stagedAttachments.isEmpty())
                assertEquals(null, vm.uiState.value.knowledgeBase)
            }
        } finally {
            compose.activityRule.scenario.onActivity { vm.newConversation() }
            runBlocking { ids.forEach { dao.deleteById(it) } }
        }
    }

    @Test fun reportPickerAddsActualHistoryRowsToComposer() {
        val ids = listOf("report-one-picker", "report-two-picker")
        runBlocking { ids.forEach { dao.insert(record(it, "IMAGE_ONLY")) } }
        try {
            compose.activityRule.scenario.onActivity { activity ->
                activity.setContent { OnnDetectTheme {
                    val navigation = rememberNavController()
                    ONNNavGraph(navigation)
                    LaunchedEffect(Unit) { navigation.navigate(Routes.CHAT_RAG_BASE) }
                } }
            }
            compose.onNodeWithTag("chat_mode_report").performClick()
            compose.waitUntil(10_000) { compose.onAllNodes(androidx.compose.ui.test.hasTestTag("chat_report_${ids.first()}")).fetchSemanticsNodes().isNotEmpty() }
            ids.forEach { compose.onNodeWithTag("chat_report_$it").performClick() }
            compose.onNodeWithText("Done").performClick()
            compose.onNodeWithTag("chat_send").assertExists()
            compose.onNodeWithTag("chat_mode_normal").performClick()
        } finally { runBlocking { ids.forEach { dao.deleteById(it) } } }
    }
}
