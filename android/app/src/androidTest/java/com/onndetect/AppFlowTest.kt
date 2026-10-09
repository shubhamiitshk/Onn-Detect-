package com.onndetect

import androidx.activity.compose.setContent
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Density
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.navigation.compose.rememberNavController
import com.onndetect.data.local.ScanHistoryDatabase
import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.ui.navigation.ONNNavGraph
import com.onndetect.ui.theme.OnnDetectTheme
import org.junit.Rule
import org.junit.Test
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.json.JSONArray

class AppFlowTest {
    @get:Rule val compose = createAndroidComposeRule<MainActivity>()
    private fun open(route: String? = null, largeText: Boolean = false) {
        compose.activityRule.scenario.onActivity { activity ->
            activity.setContent {
                OnnDetectTheme {
                    val density = LocalDensity.current
                    CompositionLocalProvider(LocalDensity provides Density(density.density,if(largeText) 2f else density.fontScale)) {
                        val navigation = rememberNavController()
                        ONNNavGraph(navigation)
                        LaunchedEffect(route) { route?.let { navigation.navigate(it) } }
                    }
                }
            }
        }
    }
    @Test fun historyRemainsReachableAtLargeFont() {
        open(largeText=true)
        compose.onNodeWithText(compose.activity.getString(R.string.history_title)).performScrollTo().performClick()
        compose.onNodeWithText(compose.activity.getString(R.string.history_policy)).assertExists()
        compose.onNodeWithText("IDENTITY SECURED").assertDoesNotExist()
    }
    @Test fun missingResultTerminatesLoading() {
        open("result/missing-test-id")
        compose.waitUntil(10_000) { compose.onAllNodesWithText(compose.activity.getString(R.string.result_not_found)).fetchSemanticsNodes().isNotEmpty() }
        compose.onNodeWithText(compose.activity.getString(R.string.result_not_found)).assertIsDisplayed()
        compose.onNodeWithText(compose.activity.getString(R.string.done)).performScrollTo().assertIsDisplayed()
    }
    @Test fun unreadableMediaDisplaysFailureWithoutInventingVerdict() {
        open("scanner?uri=content%3A%2F%2Fcom.onndetect.invalid%2Fmissing&mode=IMAGE_ONLY")
        compose.waitUntil(20_000) { compose.onAllNodesWithText(compose.activity.getString(R.string.scan_failed)).fetchSemanticsNodes().isNotEmpty() }
        compose.onNodeWithText(compose.activity.getString(R.string.scan_failed)).assertExists()
        compose.onNodeWithText(compose.activity.getString(R.string.synthetic)).assertDoesNotExist()
        compose.onNodeWithText(compose.activity.getString(R.string.no_strong_evidence)).assertDoesNotExist()
    }
    @Test fun savedAudioCoverageNeverUsesVideoFrameTerminology() {
        val context = compose.activity.applicationContext
        val dao = ScanHistoryDatabase.create(context).scanHistoryDao()
        val id = "audio-coverage-ui-regression"
        runBlocking {
            dao.insert(ScanHistoryEntity(id, 1, "AUDIO_ONLY", .2f, false, 0f, 10,
                evidenceJson = JSONObject().put("analyzedDurationMs", 4000).put("totalDurationMs", 4000)
                    .put("timestampsMs", JSONArray("[0]")).toString()))
        }
        try {
            open("result/$id")
            val audio = context.getString(R.string.coverage_audio, 4f, 4f)
            compose.waitUntil(10_000) { compose.onAllNodesWithText(audio).fetchSemanticsNodes().isNotEmpty() }
            compose.onNodeWithText(audio).performScrollTo().assertIsDisplayed()
            compose.onNodeWithText(context.resources.getQuantityString(R.plurals.coverage_frames, 1, 1)).assertDoesNotExist()
        } finally { runBlocking { dao.deleteById(id) } }
    }
}
