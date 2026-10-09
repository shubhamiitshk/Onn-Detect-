package com.onndetect.ui.screens.home

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import com.onndetect.R

@Composable
fun HomeScreen(navController: NavController, viewModel: HomeViewModel = hiltViewModel()) {
    val readiness by viewModel.readiness.collectAsStateWithLifecycle()
    val modelsVerified = if (readiness.image == null || readiness.audio == null) null else readiness.image == true && readiness.audio == true
    fun open(uri: Uri?, mode: String) {
        if (uri != null) navController.navigate("scanner?uri=${Uri.encode(uri.toString())}&mode=$mode")
    }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { open(it, "IMAGE_ONLY") }
    val audioPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { open(it, "AUDIO_ONLY") }
    val videoPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { open(it, "VIDEO_ONLY") }
    Column(Modifier.fillMaxSize().safeDrawingPadding().verticalScroll(rememberScrollState()).padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Text(stringResource(R.string.app_name), style = MaterialTheme.typography.headlineLarge)
        Text(stringResource(R.string.offline_subtitle), color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(stringResource(when (modelsVerified) { true -> R.string.assets_ready; false -> R.string.assets_failed; null -> R.string.assets_checking }),
            color = if (modelsVerified == false) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
        Text(stringResource(R.string.experimental_notice), style = MaterialTheme.typography.bodyMedium)
        ScannerCard(stringResource(R.string.image_title), stringResource(R.string.image_subtitle), readiness.image == true) {
            picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        ScannerCard(stringResource(R.string.audio_title), stringResource(R.string.audio_subtitle), readiness.audio == true) { audioPicker.launch(arrayOf("audio/*")) }
        ScannerCard(stringResource(R.string.video_title), stringResource(R.string.video_subtitle), readiness.image == true) { videoPicker.launch(arrayOf("video/*")) }
        Text(stringResource(R.string.local_provider_notice), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        OutlinedButton(onClick = { navController.navigate("history") }, modifier = Modifier.fillMaxWidth()) { Text(stringResource(R.string.history_title)) }
    }
}
@Composable
fun ScannerCard(title: String, subtitle: String, enabled: Boolean = true, onClick: () -> Unit) {
    Card(onClick = onClick, enabled = enabled, modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(subtitle, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}
