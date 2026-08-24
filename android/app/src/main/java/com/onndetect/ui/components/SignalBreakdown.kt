package com.onndetect.ui.components

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.onndetect.domain.models.SignalBreakdown

private val SignalBreakdown.available: List<Pair<String, Float?>>
    get() = listOf(
        "SRM forensic (CNN)" to srmForensicScore,
        "FFT spectral slope" to fftSpectralScore,
        "KL co-occurrence" to klDivergenceScore,
        "MobileCLIP semantic" to mobileClipScore.takeIf { it >= 0f },
        "Frequency perturbation" to perturbationScore.takeIf { it >= 0f },
        "Audio model (LCNN)" to audioModelScore.takeIf { it != 0f },
        "Audio DSP" to audioDspScore.takeIf { it != 0f },
        "Temporal consistency" to temporalScore.takeIf { it != 0f },
        "PRNU sensor noise" to prnuScore.takeIf { it >= 0f },
    )

/** Per-signal readout; signals marked unavailable are hidden or dashed. */
@Composable
fun SignalBreakdownCard(
    signals: SignalBreakdown,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier.fillMaxWidth().padding(horizontal = 16.dp)) {
        Text("Signal breakdown", style = MaterialTheme.typography.titleMedium)
        val rows = signals.available
        rows.forEachIndexed { index, pair ->
            val label = pair.first
            val value = pair.second
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp),
                horizontalArrangement = androidx.compose.foundation.layout.Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(label, style = MaterialTheme.typography.bodyMedium)
                Text(
                    text = value?.let { "%.3f".format(it) } ?: "n/a",
                    style = MaterialTheme.typography.bodyMedium,
                    fontFamily = FontFamily.Monospace,
                    color = if (value == null) MaterialTheme.colorScheme.outline
                    else MaterialTheme.colorScheme.onSurface,
                )
            }
            if (index < rows.lastIndex) Divider(thickness = 0.5.dp)
        }
    }
}
