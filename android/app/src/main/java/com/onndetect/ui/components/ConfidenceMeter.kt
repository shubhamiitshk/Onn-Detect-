package com.onndetect.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.unit.dp
import com.onndetect.R

@Composable
fun ConfidenceMeter(score: Float, modifier: Modifier = Modifier) {
    if (!score.isFinite() || score !in 0f..1f) return
    val description = stringResource(R.string.detector_score_value, score)
    Column(modifier.fillMaxWidth().padding(16.dp).semantics { stateDescription = description }) {
        Text(description, style = MaterialTheme.typography.titleMedium)
        LinearProgressIndicator(progress = { score }, modifier = Modifier.fillMaxWidth().padding(top = 8.dp))
        Text(stringResource(R.string.score_notice), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
