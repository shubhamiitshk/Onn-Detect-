package com.onndetect.ui.screens.home

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.compose.runtime.collectAsState
import androidx.navigation.NavController

@Composable
fun HomeScreen(
    navController: NavController,
    viewModel: HomeViewModel = hiltViewModel(),
) {
    val modelsVerified by viewModel.modelsVerified.collectAsState()

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { picked: Uri? ->
        if (picked != null) {
            navController.navigate("scanner?uri=${Uri.encode(picked.toString())}&mode=IMAGE_ONLY")
        }
    }

    // Photo picker on many Android versions can't handle audio/*; the
    // system document picker is universally available. Same for video.
    val audioPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { picked: Uri? ->
        if (picked != null) {
            navController.navigate("scanner?uri=${Uri.encode(picked.toString())}&mode=AUDIO_ONLY")
        }
    }

    val videoPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { picked: Uri? ->
        if (picked != null) {
            navController.navigate("scanner?uri=${Uri.encode(picked.toString())}&mode=VIDEO_ONLY")
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text("ONN-DETECT", style = MaterialTheme.typography.headlineLarge)
        Text(
            "Offline AI-media forensics",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.outline,
        )

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 8.dp),
            horizontalArrangement = Arrangement.Center,
        ) {
            val (label, color) = when (modelsVerified) {
                true -> "Models verified" to MaterialTheme.colorScheme.primary
                false -> "Model integrity failure!" to MaterialTheme.colorScheme.error
                null -> "Checking model integrity…" to MaterialTheme.colorScheme.outline
            }
            Text(label, color = color, style = MaterialTheme.typography.labelMedium)
        }

        Spacer(Modifier.height(32.dp))

        Button(
            onClick = {
                picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Scan an Image")
        }

        Spacer(Modifier.height(12.dp))

        OutlinedButton(
            onClick = {
                audioPicker.launch(arrayOf("audio/*"))
            },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Scan an Audio File")
        }

        Spacer(Modifier.height(12.dp))

        OutlinedButton(
            onClick = {
                videoPicker.launch(arrayOf("video/*"))
            },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Scan a Video")
        }

        Spacer(Modifier.height(12.dp))

        OutlinedButton(
            onClick = { navController.navigate("history") },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("Scan History")
        }
    }
}
