package com.onndetect.ui.screens.history

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import com.onndetect.R
import com.onndetect.ui.components.assessmentLabel
import java.text.DateFormat
import java.util.Date

@Composable
fun HistoryScreen(navController: NavController, viewModel: HistoryViewModel = hiltViewModel()) {
    val entries by viewModel.entries.collectAsStateWithLifecycle()
    val error by viewModel.error.collectAsStateWithLifecycle()
    var deleteId by rememberSaveable { mutableStateOf<String?>(null) }
    var clearRequested by rememberSaveable { mutableStateOf(false) }
    Column(Modifier.fillMaxSize().safeDrawingPadding().padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(stringResource(R.string.history_title), style = MaterialTheme.typography.headlineSmall)
        Text(stringResource(R.string.history_policy), style = MaterialTheme.typography.bodySmall)
        error?.let { Text(stringResource(R.string.history_update_failed), color = MaterialTheme.colorScheme.error) }
        Row {
            TextButton(onClick = { navController.popBackStack() }) { Text(stringResource(R.string.back)) }
            if (entries.isNotEmpty()) TextButton(onClick = { clearRequested = true }) { Text(stringResource(R.string.clear_history)) }
        }
        if (entries.isEmpty()) Text(stringResource(R.string.no_scans))
        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(entries, key = { it.id }) { e ->
                Card(onClick = { navController.navigate("result/${e.id}") }, modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text(assessmentLabel(e.verdict, e.modelValidated), style = MaterialTheme.typography.titleMedium)
                        Text(e.scanMode.replace('_', ' '), color = MaterialTheme.colorScheme.onSurfaceVariant)
                        if (e.score.isFinite() && e.score in 0f..1f) Text(stringResource(R.string.detector_score_value, e.score))
                        Text(DateFormat.getDateTimeInstance().format(Date(e.timestamp)), color = MaterialTheme.colorScheme.onSurfaceVariant)
                        TextButton(onClick = { deleteId = e.id }) { Text(stringResource(R.string.delete)) }
                    }
                }
            }
        }
    }
    if (clearRequested || deleteId != null) AlertDialog(onDismissRequest = { clearRequested = false; deleteId = null },
        title = { Text(stringResource(R.string.delete_title)) }, text = { Text(stringResource(R.string.delete_notice)) },
        confirmButton = { TextButton(onClick = {
            if (clearRequested) viewModel.clearAll() else deleteId?.let(viewModel::delete)
            clearRequested = false; deleteId = null
        }) { Text(stringResource(R.string.delete)) } },
        dismissButton = { TextButton(onClick = { clearRequested = false; deleteId = null }) { Text(stringResource(R.string.cancel)) } })
}
