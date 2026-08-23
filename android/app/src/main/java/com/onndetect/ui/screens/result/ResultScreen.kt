package com.onndetect.ui.screens.result

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavController
import com.onndetect.ui.components.ConfidenceMeter
import com.onndetect.domain.models.SignalBreakdown
import com.onndetect.ui.components.HeatmapViewer
import com.onndetect.ui.components.SignalBreakdownCard
import java.text.DateFormat
import java.util.Date

@Composable
fun ResultScreen(
    navController: NavController,
    scanId: String?,
    viewModel: ResultViewModel = hiltViewModel(),
) {
    val entry by viewModel.entry.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Scan Result", style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.height(16.dp))

        when (val e = entry) {
            null -> Text("Loading result…")
            else -> {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                    ) {
                        Text(
                            text = if (e.isAiGenerated) "Likely AI-Generated"
                            else "Likely Authentic",
                            style = MaterialTheme.typography.headlineMedium,
                            color = if (e.isAiGenerated) MaterialTheme.colorScheme.error
                            else MaterialTheme.colorScheme.primary,
                        )
                        Spacer(Modifier.height(8.dp))
                        Text(
                            DateFormat.getDateTimeInstance()
                                .format(Date(e.timestamp)),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.outline,
                        )
                    }
                }

                Spacer(Modifier.height(12.dp))
                ConfidenceMeter(score = e.score, isAiGenerated = e.isAiGenerated)

                // Historical rows store the fused score only; per-signal values
                // live on the transient ScanResult of the just-finished scan.
                SignalBreakdownCard(signals = placeholderSignals(e.score))
                HeatmapViewer(heatmap = null)

                Text(
                    "Duration ${e.durationMs} ms",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.outline,
                    modifier = Modifier.padding(top = 8.dp),
                )
            }
        }

        Spacer(Modifier.height(20.dp))
        Button(onClick = { navController.popBackStack() }) { Text("Done") }
    }
}

private fun placeholderSignals(fused: Float): SignalBreakdown =
    SignalBreakdown(
        mobileClipScore = -1f, srmForensicScore = -1f, fftSpectralScore = -1f,
        klDivergenceScore = -1f, perturbationScore = -1f, audioDspScore = 0f,
        audioModelScore = 0f, temporalScore = 0f, provenanceScore = fused,
    )
