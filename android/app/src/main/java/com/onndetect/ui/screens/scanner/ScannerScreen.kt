package com.onndetect.ui.screens.scanner

import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import com.onndetect.R
import com.onndetect.data.repository.ScanHistoryRepository
import com.onndetect.ui.components.ScanProgressIndicator
import com.onndetect.ui.screens.result.AssessmentDetails

@Composable
fun ScannerScreen(navController: NavController, uri: String?, mode: String?, viewModel: ScannerViewModel = hiltViewModel()) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    LaunchedEffect(uri,mode) { uri?.let { viewModel.start(Uri.parse(it),mode) } }
    BackHandler { viewModel.cancel(); navController.popBackStack() }
    LaunchedEffect(state.completedScanId) {
        state.completedScanId?.let { id -> navController.navigate("result/$id") { popUpTo("home") }; viewModel.consumeCompletion() }
    }
    Column(Modifier.fillMaxSize().safeDrawingPadding().verticalScroll(rememberScrollState()).padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Text(stringResource(R.string.scanner_title), style = MaterialTheme.typography.headlineSmall)
        if (uri == null) Text(stringResource(R.string.invalid_input))
        state.error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
        state.unsavedResult?.let { result ->
            AssessmentDetails(ScanHistoryRepository.toEntry(result))
            Button(onClick = viewModel::retrySave) { Text(stringResource(R.string.retry_save)) }
        }
        if (state.scanning) {
            ScanProgressIndicator(state.stage,state.percent,state.indeterminate)
            Text(stringResource(R.string.progress_notice), style = MaterialTheme.typography.bodySmall)
            Button(onClick = viewModel::cancel) { Text(stringResource(R.string.cancel)) }
        } else if (state.error != null && state.unsavedResult == null && uri != null) {
            Button(onClick = { viewModel.start(Uri.parse(uri),mode,retry = true) }) { Text(stringResource(R.string.retry)) }
        }
        TextButton(onClick = { viewModel.cancel(); navController.popBackStack() }) { Text(stringResource(R.string.back)) }
    }
}
