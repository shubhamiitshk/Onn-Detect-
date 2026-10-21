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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.onndetect.R
import com.onndetect.domain.models.SignalBreakdown
import com.onndetect.ui.theme.*

data class SignalMeta(
    val titleRes: Int,
    val subtitle: String,
    val tag: String,
    val value: Float
)

@Composable
fun SignalBreakdownCard(signals: SignalBreakdown, modifier: Modifier = Modifier) {
    val signalList = listOf(
        SignalMeta(R.string.signal_semantic, "Engine C \u00b7 SigLIP-B/16 vision tower + linear head", "SEMANTIC", signals.vfmProbeScore),
        SignalMeta(R.string.signal_srm, "Engine A 20-ch SRM+NPR+RGB hybrid", "RESIDUAL", signals.srmForensicScore),
        SignalMeta(R.string.signal_fft, "2D power spectrum radial slope", "SPECTRAL", signals.fftSpectralScore),
        SignalMeta(R.string.signal_kl, "Pixel co-occurrence divergence", "CO-OCCURRENCE", signals.klDivergenceScore),
        SignalMeta(R.string.signal_audio, "LFCC 16kHz speech model", "ACOUSTIC", signals.audioModelScore),
        SignalMeta(R.string.signal_temporal, "Frame-to-frame cosine variance", "TEMPORAL", signals.temporalScore),
        SignalMeta(R.string.signal_residual, "Sensor pattern noise proxy", "PRNU", signals.prnuScore)
    ).filter { it.value.isFinite() && it.value in 0f..1f }

    if (signalList.isEmpty()) return

    Box(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(SurfaceCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(20.dp))
            .padding(20.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
            // Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = stringResource(R.string.signals_title),
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = "${signalList.size} SIGNALS ACTIVE",
                    style = MaterialTheme.typography.labelSmall,
                    color = CyberCyan,
                    letterSpacing = 1.sp
                )
            }

            // Signal rows
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                signalList.forEach { meta ->
                    SignalRow(meta)
                }
            }
        }
    }
}

@Composable
private fun SignalRow(meta: SignalMeta) {
    val animatedProgress by animateFloatAsState(
        targetValue = meta.value,
        animationSpec = tween(durationMillis = 800),
        label = "SignalRowProgress"
    )

    val (barColor, statusLabel) = when {
        meta.value >= 0.70f -> Pair(SyntheticRed, "STRONG ANOMALY")
        meta.value >= 0.45f -> Pair(WarningAmber, "MODERATE TRACE")
        else -> Pair(AuthenticGreen, "NORMAL")
    }

    Column(
        verticalArrangement = Arrangement.spacedBy(6.dp),
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(SurfaceElevated)
            .padding(12.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    val tagColor = when (meta.tag) {
                        "SEMANTIC" -> NeonPurple
                        "RESIDUAL" -> CyberCyan
                        "SPECTRAL" -> ElectricBlue
                        "CO-OCCURRENCE" -> CyberCyanDark
                        else -> TextTertiary
                    }
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(4.dp))
                            .background(tagColor.copy(alpha = 0.15f))
                            .padding(horizontal = 6.dp, vertical = 2.dp)
                    ) {
                        Text(
                            text = meta.tag,
                            style = MaterialTheme.typography.labelSmall,
                            color = tagColor,
                            fontWeight = FontWeight.Bold,
                            fontSize = 9.sp
                        )
                    }
                    Text(
                        text = stringResource(meta.titleRes),
                        style = MaterialTheme.typography.titleSmall,
                        color = TextPrimary,
                        fontWeight = FontWeight.Medium
                    )
                }
                Text(
                    text = meta.subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextTertiary,
                    modifier = Modifier.padding(top = 2.dp)
                )
            }

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = statusLabel,
                    style = MaterialTheme.typography.labelSmall,
                    color = barColor,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = String.format("%.3f", meta.value),
                    style = MaterialTheme.typography.labelLarge,
                    color = TextPrimary,
                    fontWeight = FontWeight.SemiBold
                )
            }
        }

        // Horizontal visual bar
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(6.dp)
                .clip(CircleShape)
                .background(SurfaceHighlight)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(animatedProgress)
                    .fillMaxHeight()
                    .clip(CircleShape)
                    .background(barColor)
            )
        }
    }
}
