package com.onndetect.ui.screens.enrollment

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import kotlinx.coroutines.delay

@Composable
fun VoiceEnrollmentScreen(navController: NavController) {
    var recording by remember { mutableStateOf(false) }
    var progress by remember { mutableStateOf(0f) }
    var complete by remember { mutableStateOf(false) }

    LaunchedEffect(recording) {
        if (recording) {
            progress = 0f
            while (progress < 1f) {
                delay(100) // 10 seconds total (100 * 100)
                progress += 0.01f
            }
            recording = false
            complete = true
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            "VOICE LIVENESS",
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.Bold,
            letterSpacing = 2.dp
        )
        Spacer(Modifier.height(8.dp))
        Text(
            "Register a trusted identity.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        
        Spacer(Modifier.height(48.dp))

        if (complete) {
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.tertiary.copy(alpha = 0.2f)),
                shape = RoundedCornerShape(16.dp)
            ) {
                Text(
                    "IDENTITY SECURED",
                    color = MaterialTheme.colorScheme.tertiary,
                    style = MaterialTheme.typography.titleMedium,
                    modifier = Modifier.padding(24.dp)
                )
            }
            Spacer(Modifier.height(32.dp))
            Button(onClick = { navController.popBackStack() }) {
                Text("Return to Dashboard")
            }
        } else {
            Button(
                onClick = { recording = true },
                enabled = !recording,
                colors = ButtonDefaults.buttonColors(containerColor = if (recording) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary),
                modifier = Modifier.size(120.dp),
                shape = RoundedCornerShape(60.dp)
            ) {
                Text(if (recording) "RECORDING..." else "HOLD TO RECORD", style = MaterialTheme.typography.labelSmall)
            }

            Spacer(Modifier.height(32.dp))

            if (recording) {
                LinearProgressIndicator(
                    progress = { progress },
                    modifier = Modifier.fillMaxWidth().height(8.dp),
                    color = MaterialTheme.colorScheme.error,
                    trackColor = MaterialTheme.colorScheme.surfaceVariant
                )
            }
        }
    }
}
