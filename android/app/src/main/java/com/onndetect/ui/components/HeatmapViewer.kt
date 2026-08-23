package com.onndetect.ui.components

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * Displays XAI saliency overlays. Grad-CAM extraction from the TFLite graphs
 * is planned (see xai/ package); until a heatmap is produced we render an
 * explicit placeholder instead of pretending.
 */
@Composable
fun HeatmapViewer(
    heatmap: FloatArray?,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier.fillMaxWidth().padding(16.dp)) {
        Text("Explainability", style = MaterialTheme.typography.titleMedium)
        Text(
            text = if (heatmap != null) {
                "Saliency map (${heatmap.size} cells)"
            } else {
                "XAI heatmap unavailable for this scan type."
            },
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.outline,
            modifier = Modifier.padding(top = 6.dp),
        )
    }
}
