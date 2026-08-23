package com.onndetect.ui.components

import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

/**
 * Horizontal score gauge: 0 = authentic, 1 = AI-generated, with the numeric
 * probability and a color verdict.
 */
@Composable
fun ConfidenceMeter(
    score: Float,
    isAiGenerated: Boolean,
    modifier: Modifier = Modifier,
) {
    val barColor by animateColorAsState(
        targetValue = if (isAiGenerated) Color(0xFFEF4444) else Color(0xFF22C55E),
        label = "verdictColor",
    )

    Column(modifier = modifier.fillMaxWidth().padding(16.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("AI-generation probability", style = MaterialTheme.typography.labelLarge)
            Text("%.1f%%".format(score * 100f), style = MaterialTheme.typography.titleMedium)
        }
        LinearProgressIndicator(
            progress = score.coerceIn(0f, 1f),
            color = barColor,
            trackColor = MaterialTheme.colorScheme.surfaceVariant,
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 10.dp),
        )
    }
}
