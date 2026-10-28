package com.onndetect.ui.screens.home

import android.content.Context
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import com.onndetect.R
import com.onndetect.inference.ModelIntegrityGate
import com.onndetect.ui.components.GlassBottomTab
import com.onndetect.ui.components.LiquidGlassCard
import com.onndetect.ui.components.LiquidGlassScaffold
import com.onndetect.ui.navigation.Routes
import com.onndetect.ui.theme.*

@Composable
fun HomeScreen(navController: NavController, viewModel: HomeViewModel = hiltViewModel()) {
    val context = LocalContext.current
    val readiness by viewModel.readiness.collectAsStateWithLifecycle()
    val modelsVerified = if (readiness.image == null || readiness.audio == null) null else readiness.image == true && readiness.audio == true

    fun open(uri: Uri?, mode: String) {
        if (uri != null) navController.navigate("scanner?uri=${Uri.encode(uri.toString())}&mode=$mode")
    }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { open(it, "IMAGE_ONLY") }
    val audioPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { open(it, "AUDIO_ONLY") }
    val videoPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { open(it, "VIDEO_ONLY") }

    LiquidGlassScaffold(
        activeTab = GlassBottomTab.DETECT,
        onSelectTab = { tab ->
            if (tab != GlassBottomTab.DETECT) {
                navController.navigate(tab.route) {
                    launchSingleTop = true
                }
            }
        }
    ) { bottomBarPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .safeDrawingPadding()
                .verticalScroll(rememberScrollState())
                .padding(bottomBarPadding)
                .padding(horizontal = 20.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(18.dp)
        ) {
            // Top App Bar / Hero Header
            HomeTopBar(
                onHistoryClick = { navController.navigate(Routes.HISTORY) }
            )

            // Engine Status & Integrity Banner
            EngineStatusBanner(context, modelsVerified = modelsVerified)

            // Category Cards Section Title
            Text(
                text = "ON-DEVICE FORENSIC DETECTION SUITE",
                style = MaterialTheme.typography.labelSmall,
                color = TextTertiary,
                letterSpacing = 1.2.sp,
                modifier = Modifier.padding(start = 4.dp, top = 2.dp)
            )

            // 1. Image Screening Liquid Glass Card
            MediaScannerCard(
                title = stringResource(R.string.image_title),
                subtitle = stringResource(R.string.image_subtitle),
                accentGradient = Brush.horizontalGradient(listOf(CyberCyan, DeepIndigo)),
                accentColor = CyberCyan,
                iconSymbol = "📷",
                badges = listOf("20-ch SRM", "SigLIP-TAP", "XGBoost v10", "ROI Heatmap"),
                enabled = readiness.image == true,
                onClick = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }
            )

            // 2. Audio Screening Liquid Glass Card
            MediaScannerCard(
                title = stringResource(R.string.audio_title),
                subtitle = stringResource(R.string.audio_subtitle),
                accentGradient = Brush.horizontalGradient(listOf(NeonPurple, DeepIndigo)),
                accentColor = NeonPurple,
                iconSymbol = "🎙️",
                badges = listOf("16kHz LFCC v3", "LCNN-MFM", "Voice Vault"),
                enabled = readiness.audio == true,
                onClick = { audioPicker.launch(arrayOf("audio/*")) }
            )

            // 3. Video Screening Liquid Glass Card
            MediaScannerCard(
                title = stringResource(R.string.video_title),
                subtitle = stringResource(R.string.video_subtitle),
                accentGradient = Brush.horizontalGradient(listOf(DeepIndigo, CyberCyanLight)),
                accentColor = CyberCyanLight,
                iconSymbol = "🎬",
                badges = listOf("Keyframe Reuse", "BioLip rPPG", "AV Sync"),
                enabled = readiness.image == true,
                onClick = { videoPicker.launch(arrayOf("video/*")) }
            )

            // Compact Pill Row Beneath Video (Matches Hand-Drawn UI Layout: [History] + [VLM Packs])
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                LiquidGlassCard(
                    accentColor = CyberCyan,
                    cornerRadius = 16.dp,
                    modifier = Modifier
                        .weight(1f)
                        .height(50.dp),
                    onClick = { navController.navigate(Routes.HISTORY) }
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(horizontal = 14.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center
                    ) {
                        Text("⏱️", fontSize = 15.sp)
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text = stringResource(R.string.history_title),
                            style = MaterialTheme.typography.titleSmall,
                            color = TextPrimary,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                LiquidGlassCard(
                    accentColor = NeonPurple,
                    cornerRadius = 16.dp,
                    modifier = Modifier
                        .weight(1f)
                        .height(50.dp),
                    onClick = { navController.navigate(Routes.MODEL_MANAGER) }
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(horizontal = 14.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center
                    ) {
                        Text("🧠", fontSize = 15.sp)
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text = "Qwen / VLM Packs",
                            style = MaterialTheme.typography.titleSmall,
                            color = NeonPurpleLight,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            // Privacy Guarantee Card
            PrivacyGuaranteeCard()
        }
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
                text = "Autonomous On-Device Deepfake & RAG Suite",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary
            )
        }

        // Offline Verified Chip
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(20.dp))
                .background(SurfaceElevated.copy(alpha = 0.8f))
                .border(1.dp, BorderSubtle, RoundedCornerShape(20.dp))
                .clickable { onHistoryClick() }
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
private fun EngineStatusBanner(context: Context, modelsVerified: Boolean?) {
    val (statusColor, statusBg, statusTitle, statusSub) = when (modelsVerified) {
        true -> Quadruple(
            AuthenticGreen,
            AuthenticGreenContainer.copy(alpha = 0.35f),
            "Dual-Engine Active (${ModelIntegrityGate.releaseId(context)})",
            "Fast Specialist Cascade (<180ms) • Optional Qwen 0.8B RAG"
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
    accentColor: Color = CyberCyan,
    iconSymbol: String,
    badges: List<String>,
    enabled: Boolean = true,
    onClick: () -> Unit
) {
    LiquidGlassCard(
        modifier = Modifier.fillMaxWidth(),
        cornerRadius = 22.dp,
        accentColor = accentColor,
        enabled = enabled,
        onClick = onClick
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
                    text = "SCAN →",
                    style = MaterialTheme.typography.labelSmall,
                    color = if (enabled) accentColor else TextDisabled,
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
                            .background(SurfaceElevated.copy(alpha = 0.85f))
                            .border(0.5.dp, accentColor.copy(alpha = 0.3f), RoundedCornerShape(6.dp))
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    ) {
                        Text(
                            text = badge,
                            style = MaterialTheme.typography.labelSmall,
                            color = TextSecondary,
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
    LiquidGlassCard(
        modifier = Modifier.fillMaxWidth(),
        cornerRadius = 16.dp,
        accentColor = AuthenticGreen
    ) {
        Row(
            modifier = Modifier.padding(14.dp),
            verticalAlignment = Alignment.Top,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text("🛡️", fontSize = 16.sp)
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = "Zero Cloud Dependencies • 3-Window Liquid Glass",
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
