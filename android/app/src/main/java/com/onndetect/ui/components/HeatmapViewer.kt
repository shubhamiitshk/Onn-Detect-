package com.onndetect.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.onndetect.detection.visual.ResidualHeatmapExtractor
import com.onndetect.ui.theme.*
import java.util.Locale
import kotlin.math.roundToInt
import kotlin.math.sqrt

private enum class XRayMode(val label: String) {
    RETICLE("X-Ray Reticle"),
    THERMAL("Thermal Map"),
    SPIKE_ISOLATE("Spike Isolate"),
}

/**
 * Interactive Forensic X-Ray & Spatial Residual Energy Viewer across a 32x32 grid.
 *
 * In natural camera captures, sensor noise is spatially uniform. In generative or
 * spliced images, in-painted text, swapped faces, or pasted regions exhibit sharp localized
 * variance discontinuities compared to background noise.
 */
@Composable
fun HeatmapViewer(
    heatmap: FloatArray?,
    imageAspectRatio: Float = 1f,
    roiProbeScore: Float = -1f,
    modifier: Modifier = Modifier,
) {
    var mode by remember { mutableStateOf(XRayMode.RETICLE) }
    var selectedCell by remember { mutableStateOf<Pair<Int, Int>?>(null) }

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
                text = "Tampering X-Ray & Residual Map",
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

            val hotspot = remember(heatmap, side) {
                ResidualHeatmapExtractor.extractHotspotBoundingBox(heatmap, side, windowCells = (side / 4).coerceAtLeast(2))
            }

            // Interactive X-Ray Mode Selector Chips
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                XRayMode.entries.forEach { item ->
                    val active = mode == item
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .clip(RoundedCornerShape(8.dp))
                            .background(if (active) CyberCyan.copy(alpha = 0.16f) else SurfaceElevated)
                            .border(
                                1.dp,
                                if (active) CyberCyan else BorderSubtle,
                                RoundedCornerShape(8.dp)
                            )
                            .clickable { mode = item }
                            .padding(vertical = 6.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = item.label,
                            style = MaterialTheme.typography.labelSmall,
                            color = if (active) CyberCyan else TextSecondary,
                            fontWeight = if (active) FontWeight.Bold else FontWeight.Medium
                        )
                    }
                }
            }

            Text(
                text = "Tap any cell to inspect local high-pass residual energy. Target reticle locks onto the highest spatial variance cluster.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
                lineHeight = 16.sp
            )

            // Heatmap Canvas with tap gesture & target reticle
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(imageAspectRatio.coerceIn(0.4f, 2.5f))
                    .clip(RoundedCornerShape(8.dp))
                    .border(1.dp, BorderSubtle, RoundedCornerShape(8.dp))
                    .pointerInput(side) {
                        detectTapGestures { offset ->
                            val c = ((offset.x / size.width) * side).toInt().coerceIn(0, side - 1)
                            val r = ((offset.y / size.height) * side).toInt().coerceIn(0, side - 1)
                            selectedCell = Pair(r, c)
                        }
                    },
                contentAlignment = Alignment.Center
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val cellW = size.width / side
                    val cellH = size.height / side

                    for (r in 0 until side) {
                        for (c in 0 until side) {
                            val v = heatmap[r * side + c].coerceIn(0f, 1f)
                            val displayV = when (mode) {
                                XRayMode.SPIKE_ISOLATE -> if (v >= 0.62f) v else v * 0.18f
                                else -> v
                            }
                            val cellColor = colorForResidualEnergy(displayV)
                            drawRect(
                                color = cellColor,
                                topLeft = Offset(c * cellW, r * cellH),
                                size = Size(cellW + 0.5f, cellH + 0.5f)
                            )
                        }
                    }

                    // Draw X-Ray Hotspot Target Reticle when in RETICLE mode
                    if (mode == XRayMode.RETICLE && hotspot != null) {
                        val x0 = hotspot.gridCol0 * cellW
                        val y0 = hotspot.gridRow0 * cellH
                        val w = (hotspot.gridCol1 - hotspot.gridCol0) * cellW
                        val h = (hotspot.gridRow1 - hotspot.gridRow0) * cellH
                        val reticleColor = if (hotspot.zScore >= 1.85f) Color(0xFFEF4444) else Color(0xFF06B6D4)

                        drawRect(
                            color = reticleColor.copy(alpha = 0.28f),
                            topLeft = Offset(x0, y0),
                            size = Size(w, h),
                            style = Stroke(width = 2f)
                        )

                        // Corner brackets
                        val bracketLen = minOf(w, h) * 0.28f
                        val strokeW = 5f
                        // Top-left
                        drawLine(reticleColor, Offset(x0, y0), Offset(x0 + bracketLen, y0), strokeW)
                        drawLine(reticleColor, Offset(x0, y0), Offset(x0, y0 + bracketLen), strokeW)
                        // Top-right
                        drawLine(reticleColor, Offset(x0 + w, y0), Offset(x0 + w - bracketLen, y0), strokeW)
                        drawLine(reticleColor, Offset(x0 + w, y0), Offset(x0 + w, y0 + bracketLen), strokeW)
                        // Bottom-left
                        drawLine(reticleColor, Offset(x0, y0 + h), Offset(x0 + bracketLen, y0 + h), strokeW)
                        drawLine(reticleColor, Offset(x0, y0 + h), Offset(x0, y0 + h - bracketLen), strokeW)
                        // Bottom-right
                        drawLine(reticleColor, Offset(x0 + w, y0 + h), Offset(x0 + w - bracketLen, y0 + h), strokeW)
                        drawLine(reticleColor, Offset(x0 + w, y0 + h), Offset(x0 + w, y0 + h - bracketLen), strokeW)
                    }

                    // Draw selected cell crosshair
                    selectedCell?.let { (sr, sc) ->
                        if (sr in 0 until side && sc in 0 until side) {
                            drawRect(
                                color = Color.White,
                                topLeft = Offset(sc * cellW, sr * cellH),
                                size = Size(cellW, cellH),
                                style = Stroke(width = 3f)
                            )
                        }
                    }
                }
            }

            // Selected Cell Telemetry Banner or Hotspot Summary
            val sel = selectedCell
            if (sel != null && sel.first in 0 until side && sel.second in 0 until side) {
                val cellVal = heatmap[sel.first * side + sel.second].coerceIn(0f, 1f)
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(SurfaceElevated)
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Cell [Row ${sel.first}, Col ${sel.second}]",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary
                    )
                    Text(
                        text = "Residual Energy: ${(cellVal * 100f).roundToInt()}%",
                        style = MaterialTheme.typography.bodySmall,
                        color = if (cellVal >= 0.65f) SyntheticRed else CyberCyan,
                        fontWeight = FontWeight.Bold
                    )
                }
            } else if (hotspot != null) {
                val localizedSpike = hotspot.zScore >= 1.85f
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(if (localizedSpike) SyntheticRedContainer.copy(alpha = 0.3f) else SurfaceElevated)
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = if (localizedSpike) {
                            "Localized Hotspot [${hotspot.gridRow0}..${hotspot.gridRow1}, ${hotspot.gridCol0}..${hotspot.gridCol1}]"
                        } else {
                            "Peak Window [${hotspot.gridRow0}..${hotspot.gridRow1}, ${hotspot.gridCol0}..${hotspot.gridCol1}]"
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = if (localizedSpike) SyntheticRed else TextSecondary
                    )
                    val roiText = if (roiProbeScore in 0f..1f) {
                        " • ROI SRM ${(roiProbeScore * 100).roundToInt()}%"
                    } else ""
                    Text(
                        text = String.format(Locale.US, "Z-Score %.2fσ%s", hotspot.zScore, roiText),
                        style = MaterialTheme.typography.bodySmall,
                        color = if (localizedSpike) SyntheticRed else CyberCyan,
                        fontWeight = FontWeight.Bold
                    )
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
            val t = v / 0.20f
            lerpColor(Color(0xFF0A0F1D), Color(0xFF0369A1), t)
        }
        v < 0.45f -> {
            val t = (v - 0.20f) / 0.25f
            lerpColor(Color(0xFF0369A1), Color(0xFF06B6D4), t)
        }
        v < 0.70f -> {
            val t = (v - 0.45f) / 0.25f
            lerpColor(Color(0xFF06B6D4), Color(0xFFFBBF24), t)
        }
        else -> {
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
