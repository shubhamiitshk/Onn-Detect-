package com.onndetect.ui.screens.enrollment

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.onndetect.R

/** Reserved feature: no simulated recording, identity or liveness success. */
@Composable
fun VoiceEnrollmentScreen(navController: NavController) {
    Column(Modifier.fillMaxSize().safeDrawingPadding().padding(24.dp)) {
        Text(stringResource(R.string.enrollment_unavailable))
        Button(onClick = { navController.popBackStack() }) { Text(stringResource(R.string.back)) }
    }
}
