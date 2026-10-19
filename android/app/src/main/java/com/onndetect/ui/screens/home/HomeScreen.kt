package com.onndetect.ui.screens.home

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import com.onndetect.R
import com.onndetect.ui.theme.*

@Composable
fun HomeScreen(navController: NavController, viewModel: HomeViewModel = hiltViewModel()) {
    val readiness by viewModel.readiness.collectAsStateWithLifecycle()
    val modelsVerified = if (readiness.image == null || readiness.audio == null) null else readiness.image == true && readiness.audio == true

    fun open(uri: Uri?, mode: String) {
        if (uri != null) navController.navigate("scanner?uri=${Uri.encode(uri.toString())}&mode=$mode")
    }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { open(it, "IMAGE_ONLY") }
    val audioPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { open(it, "AUDIO_ONLY") }
    val videoPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { open(it, "VIDEO_ONLY") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BackgroundVoid)
            .safeDrawingPadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp)
    ) {
        // Top App Bar / Hero Header
        HomeTopBar(
            onHistoryClick = { navController.navigate("history") }
        )

        // Engine Status & Integrity Banner
        EngineStatusBanner(modelsVerified = modelsVerified)

        // Category Cards Section Title
        Text(
            text = "FORENSIC SCREENING SUITE",
            style = MaterialTheme.typography.labelSmall,
            color = TextTertiary,
            letterSpacing = 1.2.sp,
            modifier = Modifier.padding(start = 4.dp, top = 4.dp)
        )

        // Image Screening Hero Card
        MediaScannerCard(
            title = stringResource(R.string.image_title),
            subtitle = stringResource(R.string.image_subtitle),
            accentGradient = Brush.horizontalGradient(listOf(CyberCyan, DeepIndigo)),
            iconSymbol = "📷",
            badges = listOf("16-ch SRM+NPR", "2D FFT", "KL Divergence"),
            enabled = readiness.image == true,
            onClick = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
        )

        // Speech Screening Hero Card
        MediaScannerCard(
            title = stringResource(R.string.audio_title),
            subtitle = stringResource(R.string.audio_subtitle),
            accentGradient = Brush.horizontalGradient(listOf(NeonPurple, DeepIndigo)),
            iconSymbol = "🎙️",
            badges = listOf("16kHz LFCC", "MFM-LCNN", "Voiceprints"),
            enabled = readiness.audio == true,
            onClick = { audioPicker.launch(arrayOf("audio/*")) }
        )

        // Video Screening Hero Card
        MediaScannerCard(
            title = stringResource(R.string.video_title),
            subtitle = stringResource(R.string.video_subtitle),
            accentGradient = Brush.horizontalGradient(listOf(DeepIndigo, CyberCyanLight)),
            iconSymbol = "🎬",
            badges = listOf("Frame Cosine", "Speech Sample", "Temporal"),
            enabled = readiness.image == true,
            onClick = { videoPicker.launch(arrayOf("video/*")) }
        )

        // History Quick Action Button
        OutlinedButton(
            onClick = { navController.navigate("history") },
            shape = RoundedCornerShape(14.dp),
            colors = ButtonDefaults.outlinedButtonColors(
                containerColor = SurfaceCard,
                contentColor = TextPrimary
            ),
            border = ButtonDefaults.outlinedButtonBorder.copy(
                brush = Brush.horizontalGradient(listOf(BorderSubtle, BorderLight))
            ),
            modifier = Modifier
                .fillMaxWidth()
                .height(52.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text("⏱️", fontSize = 16.sp)
                Text(
                    text = stringResource(R.string.history_title),
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold
                )
            }
        }

        // Privacy Guarantee Card
        PrivacyGuaranteeCard()
    }
}

@Composable
private fun HomeTopBar(onHistoryClick: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(10.dp)
                        .clip(CircleShape)
                        .background(CyberCyan)
                )
                Text(
                    text = stringResource(R.string.app_name),
                    style = MaterialTheme.typography.headlineLarge,
                    color = TextPrimary,
                    fontWeight = FontWeight.Bold
                )
            }
            Text(
                text = "Autonomous On-Device Deepfake Detection",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary
            )
        }

        // Offline Verified Chip
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(20.dp))
                .background(SurfaceElevated)
                .border(1.dp, BorderSubtle, RoundedCornerShape(20.dp))
                .padding(horizontal = 12.dp, vertical = 6.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(6.dp)
                        .clip(CircleShape)
                        .background(AuthenticGreen)
                )
                Text(
                    text = "100% OFFLINE",
                    style = MaterialTheme.typography.labelSmall,
                    color = AuthenticGreen,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 0.5.sp
                )
            }
        }
    }
}

@Composable
private fun EngineStatusBanner(modelsVerified: Boolean?) {
    val (statusColor, statusBg, statusTitle, statusSub) = when (modelsVerified) {
        true -> Quadruple(
            AuthenticGreen,
            AuthenticGreenContainer.copy(alpha = 0.35f),
            "Engine Ready (onn-20261019-v4)",
            "16-Channel SRM+NPR Visual • LCNN Audio • Shortcut-Free XGBoost"
        )
        false -> Quadruple(
            SyntheticRed,
            SyntheticRedContainer.copy(alpha = 0.35f),
            "Model Verification Failed",
            stringResource(R.string.assets_failed)
        )
        null -> Quadruple(
            WarningAmber,
            WarningAmberContainer.copy(alpha = 0.35f),
            "Validating Checksums…",
            stringResource(R.string.assets_checking)
        )
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(statusBg)
            .border(1.dp, statusColor.copy(alpha = 0.3f), RoundedCornerShape(16.dp))
            .padding(14.dp)
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .clip(CircleShape)
                    .background(statusColor)
            )
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = statusTitle,
                    style = MaterialTheme.typography.titleSmall,
                    color = TextPrimary,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = statusSub,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary,
                    lineHeight = 16.sp
                )
            }
        }
    }
}

@Composable
fun MediaScannerCard(
    title: String,
    subtitle: String,
    accentGradient: Brush,
    iconSymbol: String,
    badges: List<String>,
    enabled: Boolean = true,
    onClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(SurfaceCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(20.dp))
            .clickable(enabled = enabled, onClick = onClick)
    ) {
        // Gradient left accent strip
        Box(
            modifier = Modifier
                .width(5.dp)
                .fillMaxHeight()
                .background(accentGradient)
        )

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Text(iconSymbol, fontSize = 22.sp)
                    Text(
                        text = title,
                        style = MaterialTheme.typography.titleMedium,
                        color = if (enabled) TextPrimary else TextDisabled,
                        fontWeight = FontWeight.Bold
                    )
                }

                Text(
                    text = "SELECT →",
                    style = MaterialTheme.typography.labelSmall,
                    color = if (enabled) CyberCyan else TextDisabled,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.sp
                )
            }

            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodyMedium,
                color = if (enabled) TextSecondary else TextDisabled,
                lineHeight = 19.sp
            )

            // Technology Pill Badges
            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.padding(top = 2.dp)
            ) {
                badges.forEach { badge ->
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(6.dp))
                            .background(SurfaceElevated)
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    ) {
                        Text(
                            text = badge,
                            style = MaterialTheme.typography.labelSmall,
                            color = TextTertiary,
                            fontSize = 10.sp
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun PrivacyGuaranteeCard() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceElevated.copy(alpha = 0.5f))
            .border(1.dp, BorderLight.copy(alpha = 0.5f), RoundedCornerShape(16.dp))
            .padding(14.dp)
    ) {
        Row(
            verticalAlignment = Alignment.Top,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text("🛡️", fontSize = 16.sp)
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = "Zero Cloud Dependencies",
                    style = MaterialTheme.typography.titleSmall,
                    color = TextPrimary,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = stringResource(R.string.local_provider_notice),
                    style = MaterialTheme.typography.bodySmall,
                    color = TextTertiary,
                    lineHeight = 16.sp
                )
            }
        }
    }
}

private data class Quadruple<A, B, C, D>(
    val first: A,
    val second: B,
    val third: C,
    val fourth: D
)
