package com.onndetect.ui.screens.result

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.unit.dp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import com.onndetect.R
import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.data.local.SignalBreakdownJson
import com.onndetect.ui.components.*
import org.json.JSONObject
import java.text.DateFormat
import java.util.Date

@Composable
fun ResultScreen(navController: NavController, viewModel: ResultViewModel = hiltViewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    Column(Modifier.fillMaxSize().safeDrawingPadding().verticalScroll(rememberScrollState()).padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Text(stringResource(R.string.result_title), style = MaterialTheme.typography.headlineSmall)
        when (val current = state) {
            ResultState.Loading -> CircularProgressIndicator()
            ResultState.NotFound -> Text(stringResource(R.string.result_not_found))
            is ResultState.Error -> { Text(stringResource(R.string.history_read_failed)); Button(onClick = viewModel::reload) { Text(stringResource(R.string.retry)) } }
            is ResultState.Ready -> AssessmentDetails(current.entry)
        }
        Button(onClick = { navController.popBackStack() }) { Text(stringResource(R.string.done)) }
    }
}
@Composable
fun AssessmentDetails(e: ScanHistoryEntity) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(assessmentLabel(e.verdict, e.modelValidated), style = MaterialTheme.typography.headlineMedium)
            Text(DateFormat.getDateTimeInstance().format(Date(e.timestamp)), color = MaterialTheme.colorScheme.onSurfaceVariant)
            e.reason?.let { Text(it) }
            Text(stringResource(R.string.result_limitations), style = MaterialTheme.typography.bodyMedium)
        }
    }
    ConfidenceMeter(e.score)
    SignalBreakdownJson.decode(e.signalsJson)?.let { SignalBreakdownCard(it) }
    val evidence = remember(e.evidenceJson) { runCatching { JSONObject(e.evidenceJson ?: "{}") }.getOrDefault(JSONObject()) }
    if (e.scanMode == "AUDIO_ONLY" && evidence.optLong("totalDurationMs") > 0) Text(stringResource(R.string.coverage_audio,
        evidence.optLong("analyzedDurationMs") / 1000f, evidence.optLong("totalDurationMs") / 1000f))
    if (e.scanMode == "VIDEO_ONLY" && evidence.optLong("analyzedDurationMs") > 0) Text(stringResource(R.string.coverage_video_audio, evidence.optLong("analyzedDurationMs") / 1000f))
    if (e.scanMode == "VIDEO_ONLY") evidence.optJSONArray("timestampsMs")?.takeIf { it.length() > 0 }?.let {
        Text(pluralStringResource(R.plurals.coverage_frames, it.length(), it.length()))
    }
    if (evidence.optInt("resolutionMinEdge") > 0) Text(stringResource(R.string.source_resolution, evidence.optInt("resolutionMinEdge")))
    if (evidence.optBoolean("blurry") || evidence.optBoolean("dark")) Text(stringResource(R.string.quality_warning))
    Text(stringResource(R.string.provenance_unverified), color = MaterialTheme.colorScheme.onSurfaceVariant)
    Text(stringResource(R.string.processing_duration, e.durationMs), style = MaterialTheme.typography.bodySmall)
}
