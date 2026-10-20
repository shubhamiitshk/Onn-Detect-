package com.onndetect.ui.screens.scanner

import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import com.onndetect.R
import com.onndetect.data.repository.ScanHistoryRepository
import com.onndetect.ui.components.ScanProgressIndicator
import com.onndetect.ui.screens.result.AssessmentDetails
import com.onndetect.ui.theme.*

@Composable
fun ScannerScreen(
    navController: NavController,
    uri: String?,
    mode: String?,
    viewModel: ScannerViewModel = hiltViewModel()
) {
    val state by viewModel.state.collectAsStateWithLifecycle()

    LaunchedEffect(uri, mode) {
        uri?.let { viewModel.start(Uri.parse(it), mode) }
    }

    BackHandler {
        viewModel.cancel()
        navController.popBackStack()
    }

    LaunchedEffect(state.completedScanId) {
        state.completedScanId?.let { id ->
            navController.navigate("result/$id") { popUpTo("home") }
            viewModel.consumeCompletion()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BackgroundVoid)
            .safeDrawingPadding()
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp)
    ) {
        // Header Bar
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = stringResource(R.string.scanner_title),
                    style = MaterialTheme.typography.headlineSmall,
                    color = TextPrimary,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = mode?.replace('_', ' ') ?: "IMAGE SCREENING",
                    style = MaterialTheme.typography.labelSmall,
                    color = CyberCyan,
                    letterSpacing = 1.sp
                )
            }

            // Pulsing Scan Active Pill
            if (state.scanning) {
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(12.dp))
                        .background(CyberCyan.copy(alpha = 0.15f))
                        .border(1.dp, CyberCyan.copy(alpha = 0.5f), RoundedCornerShape(12.dp))
                        .padding(horizontal = 10.dp, vertical = 6.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(6.dp)
                                .clip(CircleShape)
                                .background(CyberCyan)
                        )
                        Text(
                            text = "SCANNING",
                            style = MaterialTheme.typography.labelSmall,
                            color = CyberCyan,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 0.8.sp
                        )
                    }
                }
            }
        }

        // Error message if any
        state.error?.let { err ->
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(SyntheticRedContainer.copy(alpha = 0.4f))
                    .border(1.dp, SyntheticRed.copy(alpha = 0.6f), RoundedCornerShape(16.dp))
                    .padding(16.dp)
            ) {
                Text(
                    text = err,
                    color = SyntheticRedLight,
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }

        // Active Scanning HUD
        if (state.scanning) {
            ScanProgressIndicator(
                stage = state.stage,
                percent = state.percent,
                indeterminate = state.indeterminate
            )

            // Diagnostic Pipeline Architecture Steps
            PipelineStepsCard(percent = state.percent)

            // Cancel Scan Button
            Button(
                onClick = viewModel::cancel,
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = SurfaceElevated,
                    contentColor = SyntheticRed
                ),
                border = androidx.compose.foundation.BorderStroke(
                    1.dp,
                    androidx.compose.ui.graphics.SolidColor(SyntheticRed.copy(alpha = 0.4f))
                ),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp)
            ) {
                Text(
                    text = stringResource(R.string.cancel),
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold
                )
            }
        } else if (state.error != null && state.unsavedResult == null && uri != null) {
            Button(
                onClick = { viewModel.start(Uri.parse(uri), mode, retry = true) },
                shape = RoundedCornerShape(14.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = CyberCyan,
                    contentColor = BackgroundVoid
                ),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp)
            ) {
                Text(
                    text = stringResource(R.string.retry),
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        // Unsaved result preview fallback
        state.unsavedResult?.let { result ->
            AssessmentDetails(ScanHistoryRepository.toEntry(result))
            Button(
                onClick = viewModel::retrySave,
                shape = RoundedCornerShape(14.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp)
            ) {
                Text(stringResource(R.string.retry_save))
            }
        }

        // Back action button
        OutlinedButton(
            onClick = {
                viewModel.cancel()
                navController.popBackStack()
            },
            shape = RoundedCornerShape(14.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(stringResource(R.string.back))
        }
    }
}

@Composable
private fun PipelineStepsCard(percent: Float) {
    val steps = listOf(
        "1. Input Normalization & Spatial Formatting" to 15f,
        "2. 16-Channel SRM + NPR Residual Filterbank" to 40f,
        "3. 2D Spectral Slope & Pixel Co-occurrence" to 70f,
        "4. Shortcut-Free XGBoost Decision Tree Fusion" to 95f
    )

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .padding(16.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(
                text = "PIPELINE EXECUTION STAGES",
                style = MaterialTheme.typography.labelSmall,
                color = TextTertiary,
                letterSpacing = 1.sp
            )

            steps.forEach { (name, threshold) ->
                val isCompleted = percent >= threshold
                val isCurrent = percent < threshold && (percent >= threshold - 30f || threshold == 15f)

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Box(
                        contentAlignment = Alignment.Center,
                        modifier = Modifier
                            .size(18.dp)
                            .clip(CircleShape)
                            .background(
                                when {
                                    isCompleted -> AuthenticGreen
                                    isCurrent -> CyberCyan
                                    else -> SurfaceHighlight
                                }
                            )
                    ) {
                        if (isCompleted) {
                            Text("✓", color = BackgroundVoid, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                        } else if (isCurrent) {
                            Box(
                                modifier = Modifier
                                    .size(6.dp)
                                    .clip(CircleShape)
                                    .background(BackgroundVoid)
                            )
                        }
                    }

                    Text(
                        text = name,
                        style = MaterialTheme.typography.bodySmall,
                        color = when {
                            isCompleted -> TextPrimary
                            isCurrent -> CyberCyanLight
                            else -> TextDisabled
                        },
                        fontWeight = if (isCurrent) FontWeight.SemiBold else FontWeight.Normal
                    )
                }
            }
        }
    }
}
