package com.onndetect.ui.screens.home

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
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
            .background(MaterialTheme.colorScheme.background)
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        
        Text(
            "ONN-DETECT", 
            style = MaterialTheme.typography.headlineLarge,
            color = MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.Bold,
            letterSpacing = 4.dp
        )
        Spacer(Modifier.height(4.dp))
        Text(
            "Zero-Trust Offline Forensics",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            letterSpacing = 1.dp
        )

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 16.dp, bottom = 32.dp),
            horizontalArrangement = Arrangement.Center,
        ) {
            val (label, color) = when (modelsVerified) {
                true -> "SYSTEMS ONLINE" to MaterialTheme.colorScheme.tertiary
                false -> "INTEGRITY COMPROMISED" to MaterialTheme.colorScheme.error
                null -> "INITIALIZING CORE..." to MaterialTheme.colorScheme.secondary
            }
            Card(
                colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.1f)),
                shape = RoundedCornerShape(16.dp)
            ) {
                Text(
                    label, 
                    color = color, 
                    style = MaterialTheme.typography.labelMedium,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.dp
                )
            }
        }

        ScannerCard(
            title = "Visual Forensics",
            subtitle = "Scan JPG/PNG for generative artifacts",
            onClick = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
        )
        
        Spacer(Modifier.height(16.dp))
        
        ScannerCard(
            title = "Audio Forensics",
            subtitle = "Analyze Voice & Speech (WAV/FLAC)",
            onClick = { audioPicker.launch(arrayOf("audio/*")) }
        )
        
        Spacer(Modifier.height(16.dp))

        ScannerCard(
            title = "Cross-Modal Forensics",
            subtitle = "Deepfake Video & Lip-Sync (MP4)",
            onClick = { videoPicker.launch(arrayOf("video/*")) }
        )
        
        Spacer(Modifier.height(32.dp))

        OutlinedButton(
            onClick = { navController.navigate("history") },
            modifier = Modifier.fillMaxWidth().height(56.dp),
            shape = RoundedCornerShape(12.dp)
        ) {
            Text("Forensic Log", letterSpacing = 1.dp)
        }
    }
}

@Composable
fun ScannerCard(title: String, subtitle: String, onClick: () -> Unit) {
    Card(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.onSurface)
            Spacer(Modifier.height(4.dp))
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}
