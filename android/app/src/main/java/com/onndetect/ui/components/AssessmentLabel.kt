package com.onndetect.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.onndetect.R
import com.onndetect.ui.theme.*

fun assessmentStringId(verdict: String, validated: Boolean): Int = if (!validated) R.string.inconclusive else when (verdict) {
    "SYNTHETIC" -> R.string.synthetic
    "AUTHENTIC" -> R.string.no_strong_evidence
    else -> R.string.inconclusive
}

@Composable
fun assessmentLabel(verdict: String, validated: Boolean): String = stringResource(assessmentStringId(verdict, validated))

/**
 * Premium Forensic Verdict Banner with ambient gradient glow, status badge,
 * and clear diagnostic summary.
 */
@Composable
fun ForensicVerdictCard(
    verdict: String,
    validated: Boolean,
    score: Float,
    reason: String? = null,
    /**
     * Whether an independent ON-DEVICE evaluation verified the release.
     *
     * The release line says which. A release that is only desktop-verified
     * is usable - its packaged bytes are hash-bound, measured and calibrated
     * - but it is not device-verified, and the card must not imply that it
     * is.
     */
    deviceValidated: Boolean = false,
    releaseId: String = "unknown",
    /**
     * The scan mode, because the engines that ran depend on it. An
     * AUDIO-only scan runs the speech model and NOTHING else, so a global
     * "SigLIP-B/16 TOWER + 16-CH SRM/NPR" line would name two engines that
     * never executed.
     */
    scanMode: String = "IMAGE_ONLY",
    modifier: Modifier = Modifier
) {
    val isSynthetic = validated && verdict == "SYNTHETIC"
    val isAuthentic = validated && verdict == "AUTHENTIC"

    val (accentColor, containerGradient, statusIcon, verdictTitle, verdictDesc) = when {
        isSynthetic -> Quintuple(
            SyntheticRed,
            SyntheticBannerGradient,
            "⚠️",
            stringResource(R.string.synthetic),
            "High-frequency sensor and synthesis residuals match known generative model patterns."
        )
        isAuthentic -> Quintuple(
            AuthenticGreen,
            AuthenticBannerGradient,
            "✓",
            stringResource(R.string.no_strong_evidence),
            "Noise residuals and spectral frequencies are consistent with natural camera captures."
        )
        else -> Quintuple(
            WarningAmber,
            InconclusiveBannerGradient,
            "ℹ",
            stringResource(R.string.inconclusive),
            reason ?: "Signal confidence below operational threshold. No definitive origin established."
        )
    }

    Box(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(containerGradient)
            .border(1.dp, accentColor.copy(alpha = 0.45f), RoundedCornerShape(20.dp))
            .padding(20.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            // Top row: Status icon, verdict text, and verified badge
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                // Pulsing-style icon circle
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier
                        .size(44.dp)
                        .clip(CircleShape)
                        .background(accentColor.copy(alpha = 0.2f))
                        .border(1.5.dp, accentColor.copy(alpha = 0.8f), CircleShape)
                ) {
                    Text(
                        text = statusIcon,
                        color = accentColor,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold
                    )
                }

                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = verdictTitle,
                        style = MaterialTheme.typography.headlineSmall,
                        color = TextPrimary,
                        fontWeight = FontWeight.Bold
                    )
                    // The final percentage, next to the verdict. The score is
                    // the AI-EVIDENCE score of the calibrated fusion at its
                    // operating point; the meter states what it is and is not.
                    if (validated && score.isFinite() && score in 0f..1f) {
                        Text(
                            text = stringResource(R.string.ai_evidence_percentage, (score * 100f).toInt()),
                            style = MaterialTheme.typography.titleMedium,
                            color = accentColor,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    Text(
                        text = if (validated) stringResource(engineLineId(scanMode)) else "DIAGNOSTIC MATH ONLY",
                        style = MaterialTheme.typography.labelSmall,
                        color = accentColor,
                        letterSpacing = 1.sp
                    )
                }
            }

            // Explanatory description
            Text(
                text = verdictDesc,
                style = MaterialTheme.typography.bodyMedium,
                color = TextSecondary,
                lineHeight = 20.sp
            )

            // Security & Offline verification pill
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier
                    .clip(RoundedCornerShape(8.dp))
                    .background(BackgroundVoid.copy(alpha = 0.5f))
                    .padding(horizontal = 10.dp, vertical = 6.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(6.dp)
                        .clip(CircleShape)
                        .background(if (validated) AuthenticGreen else WarningAmber)
                )
                Text(
                    text = when {
                        !validated -> "Unverified Model State"
                        deviceValidated -> stringResource(R.string.release_line_device_verified, releaseId)
                        else -> stringResource(R.string.release_line_desktop_verified, releaseId)
                    },
                    style = MaterialTheme.typography.labelSmall,
                    color = TextTertiary
                )
            }
        }
    }
}

/**
 * The engine line for a scan mode.
 *
 * The line must name the engines that ACTUALLY ran:
 *  - an image scan runs the SigLIP-B/16 tower (slot 6) and the 16-channel
 *    SRM/NPR residual CNN (Engine A);
 *  - an audio scan runs the LCNN speech model and NOTHING else;
 *  - a video scan runs the tower and the residual CNN per frame, plus the
 *    audio model.
 */
fun engineLineId(scanMode: String): Int = when (scanMode) {
    "AUDIO_ONLY" -> R.string.verdict_engine_line_audio
    "VIDEO_ONLY", "VIDEO_WITH_AUDIO", "CROSS_MODAL" -> R.string.verdict_engine_line_video
    else -> R.string.verdict_engine_line
}

private data class Quintuple<A, B, C, D, E>(
    val first: A,
    val second: B,
    val third: C,
    val fourth: D,
    val fifth: E
)
