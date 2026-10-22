package com.onndetect.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.onndetect.ui.theme.*
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Renders spatial high-pass residual energy overlays across a 32x32 grid.
 *
 * In natural camera captures, sensor noise is spatially uniform. In generative or
 * spliced images, in-painted text, faces, or pasted regions exhibit sharp localized
 * variance discontinuities compared to background noise.
 */
@Composable
fun HeatmapViewer(
    heatmap: FloatArray?,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Spatial Residual Energy",
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                color = TextPrimary
            )
            if (heatmap != null && heatmap.isNotEmpty()) {
                val side = sqrt(heatmap.size.toFloat()).roundToInt()
                Text(
                    text = "${side}×${side} Grid",
                    style = MaterialTheme.typography.labelSmall,
                    color = CyberCyan
                )
            }
        }

        if (heatmap != null && heatmap.isNotEmpty()) {
            val side = sqrt(heatmap.size.toFloat()).roundToInt()
            require(side * side == heatmap.size) { "Heatmap must be square" }

            Text(
                text = "Spatial noise variance across the image canvas. Concentrated hot clusters highlight localized pixel splicing or generative resampling.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
                lineHeight = 16.sp
            )

            // Heatmap Canvas
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f)
                    .clip(RoundedCornerShape(8.dp))
                    .border(1.dp, BorderSubtle, RoundedCornerShape(8.dp)),
                contentAlignment = Alignment.Center
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val cellW = size.width / side
                    val cellH = size.height / side

                    for (r in 0 until side) {
                        for (c in 0 until side) {
                            val v = heatmap[r * side + c].coerceIn(0f, 1f)
                            val cellColor = colorForResidualEnergy(v)
                            drawRect(
                                color = cellColor,
                                topLeft = Offset(c * cellW, r * cellH),
                                size = Size(cellW + 0.5f, cellH + 0.5f) // avoid subpixel seams
                            )
                        }
                    }
                }
            }

            // Legend Gradient Bar
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp)
                        .clip(RoundedCornerShape(4.dp))
                        .background(
                            Brush.horizontalGradient(
                                listOf(
                                    Color(0xFF0F172A),
                                    Color(0xFF0284C7),
                                    Color(0xFF06B6D4),
                                    Color(0xFFFBBF24),
                                    Color(0xFFEF4444)
                                )
                            )
                        )
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "Uniform (no localised anomaly)",
                        style = MaterialTheme.typography.labelSmall,
                        color = TextTertiary
                    )
                    Text(
                        text = "Discontinuity (localised anomaly)",
                        style = MaterialTheme.typography.labelSmall,
                        color = TextTertiary
                    )
                }
            }
        } else {
            Text(
                text = "Spatial XAI heatmap is unavailable for this media format.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary
            )
        }
    }
}

/**
 * Maps normalized residual energy [0.0, 1.0] to a thermal heatmap color.
 */
private fun colorForResidualEnergy(v: Float): Color {
    return when {
        v < 0.20f -> {
            // Navy to Deep Blue
            val t = v / 0.20f
            lerpColor(Color(0xFF0A0F1D), Color(0xFF0369A1), t)
        }
        v < 0.45f -> {
            // Deep Blue to Cyan
            val t = (v - 0.20f) / 0.25f
            lerpColor(Color(0xFF0369A1), Color(0xFF06B6D4), t)
        }
        v < 0.70f -> {
            // Cyan to Amber
            val t = (v - 0.45f) / 0.25f
            lerpColor(Color(0xFF06B6D4), Color(0xFFFBBF24), t)
        }
        else -> {
            // Amber to Crimson Red
            val t = (v - 0.70f) / 0.30f
            lerpColor(Color(0xFFFBBF24), Color(0xFFEF4444), t)
        }
    }
}

private fun lerpColor(c1: Color, c2: Color, t: Float): Color {
    val clampedT = t.coerceIn(0f, 1f)
    return Color(
        red = c1.red + (c2.red - c1.red) * clampedT,
        green = c1.green + (c2.green - c1.green) * clampedT,
        blue = c1.blue + (c2.blue - c1.blue) * clampedT,
        alpha = 1f
    )
}
