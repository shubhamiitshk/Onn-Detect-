package com.onndetect.ui.components

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.onndetect.ui.theme.*
import kotlin.math.cos
import kotlin.math.sin

/**
 * High-Tech Cyber-Forensic Radar and Scanning Progress Indicator
 */
@Composable
fun ScanProgressIndicator(
    stage: String,
    percent: Float,
    indeterminate: Boolean,
    modifier: Modifier = Modifier,
) {
    val cleanPercent = (percent.takeIf { it.isFinite() } ?: 0f).coerceIn(0f, 100f)

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(20.dp),
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(24.dp))
            .background(SurfaceCard)
            .border(1.dp, BorderCyanGlow, RoundedCornerShape(24.dp))
            .padding(24.dp)
    ) {
        // Centerpiece Animated Radar Reticle
        ForensicRadarVisualization()

        // Stage Title & Live Percentage
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(
                text = if (indeterminate) "ANALYZING TENSORS…" else "${cleanPercent.toInt()}%",
                style = MaterialTheme.typography.displaySmall,
                color = CyberCyan,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = stage,
                style = MaterialTheme.typography.titleMedium,
                color = TextPrimary,
                fontWeight = FontWeight.Medium
            )
            Text(
                text = "100% On-Device Neural + DSP Diagnostics",
                style = MaterialTheme.typography.labelSmall,
                color = TextTertiary,
                letterSpacing = 1.sp
            )
        }

        // Precision Progress Bar with Cyan Glow
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(CircleShape)
                .background(SurfaceElevated)
        ) {
            val progressFraction = cleanPercent / 100f
            Box(
                modifier = Modifier
                    .fillMaxWidth(if (indeterminate) 0.4f else progressFraction)
                    .fillMaxHeight()
                    .clip(CircleShape)
                    .background(
                        Brush.horizontalGradient(
                            colors = listOf(CyberCyan, DeepIndigo, NeonPurple)
                        )
                    )
            )
        }
    }
}

@Composable
fun ForensicRadarVisualization(modifier: Modifier = Modifier) {
    val infiniteTransition = rememberInfiniteTransition(label = "RadarTransition")

    // Rotation angle for the sweeping radar line
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 2200, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "RadarRotation"
    )

    // Pulsing alpha for concentric rings
    val pulseAlpha by infiniteTransition.animateFloat(
        initialValue = 0.25f,
        targetValue = 0.75f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1100, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "RadarPulse"
    )

    // Forensic blip blink alpha
    val blipAlpha by infiniteTransition.animateFloat(
        initialValue = 0.1f,
        targetValue = 0.9f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 800, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "BlipPulse"
    )

    Box(
        contentAlignment = Alignment.Center,
        modifier = modifier
            .size(160.dp)
            .clip(CircleShape)
            .background(SurfaceElevated)
            .border(1.5.dp, BorderCyanGlow, CircleShape)
    ) {
        Canvas(modifier = Modifier.size(150.dp)) {
            val center = Offset(size.width / 2, size.height / 2)
            val maxRadius = size.minDimension / 2

            // Sweeping radar fan trail (arc)
            drawArc(
                brush = Brush.sweepGradient(
                    0.0f to Color.Transparent,
                    0.85f to Color.Transparent,
                    0.95f to CyberCyan.copy(alpha = 0.12f),
                    1.0f to CyberCyan.copy(alpha = 0.35f),
                    center = center
                ),
                startAngle = rotation - 45f,
                sweepAngle = 45f,
                useCenter = true,
                topLeft = Offset(center.x - maxRadius, center.y - maxRadius),
                size = androidx.compose.ui.geometry.Size(maxRadius * 2, maxRadius * 2)
            )

            // Concentric range rings
            drawCircle(
                color = CyberCyan.copy(alpha = 0.15f),
                radius = maxRadius * 0.35f,
                center = center,
                style = Stroke(width = 1.dp.toPx())
            )
            drawCircle(
                color = NeonPurple.copy(alpha = 0.20f),
                radius = maxRadius * 0.68f,
                center = center,
                style = Stroke(width = 1.dp.toPx())
            )
            drawCircle(
                color = CyberCyan.copy(alpha = pulseAlpha),
                radius = maxRadius * 0.98f,
                center = center,
                style = Stroke(width = 1.5.dp.toPx())
            )

            // Crosshairs
            drawLine(
                color = BorderLight,
                start = Offset(center.x, center.y - maxRadius),
                end = Offset(center.x, center.y + maxRadius),
                strokeWidth = 1.dp.toPx()
            )
            drawLine(
                color = BorderLight,
                start = Offset(center.x - maxRadius, center.y),
                end = Offset(center.x + maxRadius, center.y),
                strokeWidth = 1.dp.toPx()
            )

            // Simulated forensic telemetry blips
            val blip1 = Offset(center.x + maxRadius * 0.45f, center.y - maxRadius * 0.30f)
            val blip2 = Offset(center.x - maxRadius * 0.50f, center.y + maxRadius * 0.40f)
            drawCircle(color = CyberCyan.copy(alpha = blipAlpha), radius = 3.dp.toPx(), center = blip1)
            drawCircle(color = NeonPurple.copy(alpha = 1f - blipAlpha), radius = 2.5.dp.toPx(), center = blip2)

            // Sweeping lead beam line
            val angleRad = Math.toRadians(rotation.toDouble())
            val targetX = center.x + (maxRadius * 0.95f * cos(angleRad)).toFloat()
            val targetY = center.y + (maxRadius * 0.95f * sin(angleRad)).toFloat()

            drawLine(
                brush = Brush.linearGradient(
                    colors = listOf(CyberCyan, CyberCyanLight),
                    start = center,
                    end = Offset(targetX, targetY)
                ),
                start = center,
                end = Offset(targetX, targetY),
                strokeWidth = 2.5.dp.toPx()
            )

            // Center glowing core dot
            drawCircle(
                color = CyberCyanLight,
                radius = 4.dp.toPx(),
                center = center
            )
        }
    }
}
