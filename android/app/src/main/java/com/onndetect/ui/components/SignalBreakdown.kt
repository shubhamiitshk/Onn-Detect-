package com.onndetect.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.onndetect.R
import com.onndetect.domain.models.SignalBreakdown

@Composable
fun SignalBreakdownCard(signals: SignalBreakdown, modifier: Modifier = Modifier) {
    val rows = listOf(R.string.signal_srm to signals.srmForensicScore, R.string.signal_fft to signals.fftSpectralScore,
        R.string.signal_kl to signals.klDivergenceScore, R.string.signal_audio to signals.audioModelScore,
        R.string.signal_temporal to signals.temporalScore, R.string.signal_residual to signals.prnuScore)
        .filter { it.second.isFinite() && it.second in 0f..1f }
    if (rows.isEmpty()) return
    Column(modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(stringResource(R.string.signals_title), style = MaterialTheme.typography.titleMedium)
        rows.forEach { (label, value) ->
            Text(stringResource(R.string.signal_value, stringResource(label), value), style = MaterialTheme.typography.bodyMedium)
        }
    }
}
