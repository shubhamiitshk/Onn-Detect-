package com.onndetect.ui.components

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.onndetect.R
import com.onndetect.ui.theme.*

/**
 * Modern Cyber-Forensic Confidence Gauge featuring animated progress,
 * operating threshold marker, and risk interpretation.
 */
@Composable
fun ConfidenceMeter(
    score: Float,
    threshold: Float = 0.499f,
    modifier: Modifier = Modifier
) {
    if (!score.isFinite() || score !in 0f..1f) return

    val animatedScore by animateFloatAsState(
        targetValue = score,
        animationSpec = tween(durationMillis = 900),
        label = "ConfidenceMeterProgress"
    )

    val description = stringResource(R.string.detector_score_value, score)
    val percentage = (score * 100f).coerceIn(0f, 100f)

    val (meterColor, riskText, riskBg) = when {
        score >= 0.70f -> Triple(SyntheticRed, "HIGH SYNTHETIC PROBABILITY", SyntheticRedContainer)
        score >= threshold -> Triple(WarningAmber, "ELEVATED SYNTHETIC RISK", WarningAmberContainer)
        else -> Triple(AuthenticGreen, "LOW SYNTHETIC EVIDENCE", AuthenticGreenContainer)
    }

    Box(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(SurfaceCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(20.dp))
            .padding(20.dp)
            .semantics { stateDescription = description }
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            // Header: Title & Risk Pill
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "Confidence Score",
                        style = MaterialTheme.typography.titleMedium,
                        color = TextPrimary,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "Decision Threshold: ${(threshold * 100f).toInt()}%",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextTertiary
                    )
                }

                // Risk Pill
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(riskBg.copy(alpha = 0.6f))
                        .border(1.dp, meterColor.copy(alpha = 0.6f), RoundedCornerShape(8.dp))
                        .padding(horizontal = 10.dp, vertical = 5.dp)
                ) {
                    Text(
                        text = riskText,
                        style = MaterialTheme.typography.labelSmall,
                        color = meterColor,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 0.5.sp
                    )
                }
            }

            // Big Score Display & Gauge Bar
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.Bottom,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(verticalAlignment = Alignment.Bottom) {
                    Text(
                        text = String.format("%.1f", percentage),
                        style = MaterialTheme.typography.displayMedium,
                        color = meterColor,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "%",
                        style = MaterialTheme.typography.titleLarge,
                        color = meterColor.copy(alpha = 0.7f),
                        modifier = Modifier.padding(bottom = 6.dp, start = 2.dp)
                    )
                }

                Text(
                    text = String.format("Score: %.4f", score),
                    style = MaterialTheme.typography.bodySmall,
                    color = TextTertiary,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
            }

            // Precision Gauge Track with Threshold Indicator
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(14.dp)
                        .clip(CircleShape)
                        .background(SurfaceElevated)
                ) {
                    // Animated Fill Bar
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(animatedScore)
                            .fillMaxHeight()
                            .clip(CircleShape)
                            .background(
                                Brush.horizontalGradient(
                                    colors = listOf(CyberCyan, meterColor)
                                )
                            )
                    )

                    // Threshold Marker Tick
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(threshold)
                            .fillMaxHeight()
                            .wrapContentWidth(Alignment.End)
                            .width(2.5.dp)
                            .fillMaxHeight()
                            .background(TextPrimary.copy(alpha = 0.8f))
                    )
                }

                // Range Labels
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text("0% Natural", style = MaterialTheme.typography.labelSmall, color = TextTertiary)
                    Text("Threshold (${(threshold * 100).toInt()}%)", style = MaterialTheme.typography.labelSmall, color = TextSecondary)
                    Text("100% Synthetic", style = MaterialTheme.typography.labelSmall, color = TextTertiary)
                }
            }

            // Explanatory footer notice
            Text(
                text = stringResource(R.string.score_notice),
                style = MaterialTheme.typography.bodySmall,
                color = TextTertiary,
                lineHeight = 16.sp
            )
        }
    }
}
