package com.onndetect.ui.screens.scanner

import android.net.Uri
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavController
import com.onndetect.ui.components.ScanProgressIndicator

@Composable
fun ScannerScreen(
    navController: NavController,
    uri: String?,
    mode: String?,
    viewModel: ScannerViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()

    LaunchedEffect(uri, mode) {
        if (uri != null && state.completedScanId == null && !state.scanning) {
            val parsed = Uri.parse(uri)
            when (mode) {
                "AUDIO_ONLY" -> viewModel.scanAudio(parsed)
                "VIDEO_ONLY", "VIDEO_WITH_AUDIO", "CROSS_MODAL" -> viewModel.scanVideo(parsed)
                else -> viewModel.scanImage(parsed)
            }
        }
    }

    LaunchedEffect(state.completedScanId) {
        state.completedScanId?.let { id ->
            navController.navigate("result/$id") {
                popUpTo("home")
            }
            viewModel.consumeCompletion()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Scanner", style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.height(24.dp))

        when {
            state.error != null -> {
                Text(
                    "Error: ${state.error}",
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyLarge,
                )
                Spacer(Modifier.height(16.dp))
                Button(onClick = { navController.popBackStack() }) { Text("Back") }
            }

            state.scanning || state.stage.isNotEmpty() ->
                ScanProgressIndicator(
                    stage = state.stage.ifEmpty { "Preparing…" },
                    percent = state.percent,
                    indeterminate = state.indeterminate,
                )

            else -> {
                Text("Pick an image on the Home screen to start a scan.")
                Button(
                    onClick = { navController.popBackStack() },
                    modifier = Modifier.padding(top = 16.dp),
                ) { Text("Back") }
            }
        }
    }
}
