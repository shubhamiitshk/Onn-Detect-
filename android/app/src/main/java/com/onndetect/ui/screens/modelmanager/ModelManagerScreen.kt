package com.onndetect.ui.screens.modelmanager

import android.content.Intent
import android.net.Uri
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import com.onndetect.detection.visual.VlmPackManager
import com.onndetect.ui.theme.*

@Composable
fun ModelManagerScreen(
    navController: NavController,
    viewModel: ModelManagerViewModel = hiltViewModel(),
) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    val zipPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) {
            viewModel.importZip(uri)
        }
    }

    LaunchedEffect(uiState.successMessage) {
        uiState.successMessage?.let {
            Toast.makeText(context, it, Toast.LENGTH_LONG).show()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BackgroundVoid)
            .safeDrawingPadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp)
    ) {
        // Top Header
        ModelManagerTopBar(onBackClick = { navController.popBackStack() })

        // Device Hardware Profile Card
        HardwareProfileCard(uiState.deviceProfile)

        // Import In-Progress or Alert Banner
        if (uiState.isImporting) {
            ImportProgressCard(progressText = uiState.importProgressText)
        } else if (uiState.errorMessage != null) {
            ErrorAlertCard(
                message = uiState.errorMessage!!,
                onDismiss = { viewModel.clearMessages() }
            )
        } else if (uiState.successMessage != null) {
            SuccessAlertCard(
                message = uiState.successMessage!!,
                onDismiss = { viewModel.clearMessages() }
            )
        }

        // Active Installed Pack Status Card
        InstalledPackCard(
            installedPack = uiState.installedPack,
            cascadeEnabled = uiState.cascadeEnabled,
            onUninstall = { viewModel.uninstall() },
            onToggleCascade = { viewModel.toggleCascade(it) }
        )

        // Section Title: Available Tiers
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                text = "ON-DEVICE VLM TIERS",
                style = MaterialTheme.typography.labelSmall,
                color = TextTertiary,
                letterSpacing = 1.2.sp,
                modifier = Modifier.padding(start = 4.dp)
            )
            Text(
                text = "Download packs from Google Drive and import them with 1 tap. Completely offline.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary,
                modifier = Modifier.padding(start = 4.dp)
            )
        }

        // Catalog Cards
        uiState.catalog.forEach { entry ->
            VlmCatalogCard(
                entry = entry,
                onDownloadDriveClick = {
                    val intent = Intent(Intent.ACTION_VIEW, Uri.parse(entry.defaultDriveUrl))
                    context.startActivity(intent)
                },
                onImportClick = {
                    zipPicker.launch(arrayOf("application/zip", "application/x-zip-compressed", "application/octet-stream", "*/*"))
                }
            )
        }

        // Open Full Drive Folder Card
        PublicDriveFolderCard(
            onOpenFolder = {
                val intent = Intent(Intent.ACTION_VIEW, Uri.parse(VlmPackManager.DRIVE_FOLDER_URL))
                context.startActivity(intent)
            }
        )

        // Developer Settings Card
        DeveloperSettingsCard(
            allowCustomKey = uiState.allowCustomKey,
            onToggleAllowCustomKey = { viewModel.toggleAllowCustomKey(it) }
        )

        // Privacy Guarantee Card
        PrivacyAssuranceCard()
    }
}

@Composable
private fun ModelManagerTopBar(onBackClick: () -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            IconButton(
                onClick = onBackClick,
                modifier = Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(SurfaceCard)
            ) {
                Text("←", color = TextPrimary, fontSize = 20.sp, fontWeight = FontWeight.Bold)
            }

            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = "VLM Pack Manager",
                    style = MaterialTheme.typography.headlineSmall,
                    color = TextPrimary,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "On-Device Multimodal Escalation",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary
                )
            }
        }

        // 100% Offline Chip
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(20.dp))
                .background(SurfaceElevated)
                .border(1.dp, BorderSubtle, RoundedCornerShape(20.dp))
                .padding(horizontal = 10.dp, vertical = 6.dp)
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
                    text = "OFFLINE",
                    style = MaterialTheme.typography.labelSmall,
                    color = AuthenticGreen,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
private fun HardwareProfileCard(profile: VlmPackManager.DeviceHardwareProfile) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .padding(16.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "DEVICE HARDWARE SPECS",
                    style = MaterialTheme.typography.labelSmall,
                    color = CyberCyan,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.sp
                )
                Text(
                    text = profile.abi,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextTertiary
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Total RAM", style = MaterialTheme.typography.bodySmall, color = TextSecondary)
                    Text("${profile.totalRamGb} GB", style = MaterialTheme.typography.titleMedium, color = TextPrimary, fontWeight = FontWeight.Bold)
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text("Available RAM", style = MaterialTheme.typography.bodySmall, color = TextSecondary)
                    Text("${profile.availableRamGb} GB", style = MaterialTheme.typography.titleMedium, color = AuthenticGreen, fontWeight = FontWeight.Bold)
                }
                Column(modifier = Modifier.weight(1.2f)) {
                    Text("Recommended", style = MaterialTheme.typography.bodySmall, color = TextSecondary)
                    Text(
                        text = when (profile.recommendedPresetId) {
                            "qwen3.5-2b" -> "Qwen 3.5 2B"
                            "qwen3.5-0.8b" -> "Qwen 3.5 0.8B"
                            else -> "SmolVLM2 500M"
                        },
                        style = MaterialTheme.typography.titleSmall,
                        color = CyberCyan,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}

@Composable
private fun InstalledPackCard(
    installedPack: VlmPackManager.InstalledVlmPackDetails?,
    cascadeEnabled: Boolean,
    onUninstall: () -> Unit,
    onToggleCascade: (Boolean) -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceCard)
            .border(
                1.dp,
                if (installedPack != null) AuthenticGreen.copy(alpha = 0.5f) else BorderSubtle,
                RoundedCornerShape(16.dp)
            )
            .padding(16.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Box(
                        modifier = Modifier
                            .size(10.dp)
                            .clip(CircleShape)
                            .background(if (installedPack != null) AuthenticGreen else WarningAmber)
                    )
                    Text(
                        text = if (installedPack != null) "INSTALLED RESIDENT PACK" else "NO VLM PACK INSTALLED",
                        style = MaterialTheme.typography.labelSmall,
                        color = if (installedPack != null) AuthenticGreen else WarningAmber,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.sp
                    )
                }

                if (installedPack != null) {
                    Text(
                        text = "${installedPack.totalSizeBytes / (1024 * 1024)} MB",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            if (installedPack != null) {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        text = installedPack.packId,
                        style = MaterialTheme.typography.titleMedium,
                        color = TextPrimary,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Model: ${installedPack.modelId} • Projector: ${installedPack.visionFileName}",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary
                    )
                    Text(
                        text = if (installedPack.isOfficialPublisher) "✓ Verified Official Ed25519 Signature" else "⚠️ Self-Signed / Custom Developer Key",
                        style = MaterialTheme.typography.bodySmall,
                        color = if (installedPack.isOfficialPublisher) AuthenticGreen else WarningAmber
                    )
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Tier-2 Escalation Active",
                        style = MaterialTheme.typography.bodyMedium,
                        color = TextPrimary
                    )
                    Switch(
                        checked = cascadeEnabled,
                        onCheckedChange = onToggleCascade,
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = BackgroundVoid,
                            checkedTrackColor = CyberCyan,
                        )
                    )
                }

                OutlinedButton(
                    onClick = onUninstall,
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = SyntheticRed),
                    border = androidx.compose.foundation.BorderStroke(1.dp, SyntheticRed.copy(alpha = 0.5f)),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Uninstall & Free Storage", fontWeight = FontWeight.SemiBold)
                }
            } else {
                Text(
                    text = "The app is currently using the fast Tier-1 detectors (SRM CNN + SigLIP probe). Download and import a VLM pack below to unlock multimodal diagnostic reasoning for ambiguous images.",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary
                )
            }
        }
    }
}

@Composable
private fun VlmCatalogCard(
    entry: VlmPackManager.VlmCatalogEntry,
    onDownloadDriveClick: () -> Unit,
    onImportClick: () -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceCard)
            .border(
                1.dp,
                if (entry.isRecommended) CyberCyan.copy(alpha = 0.6f) else BorderSubtle,
                RoundedCornerShape(16.dp)
            )
            .padding(16.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = entry.name,
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary,
                    fontWeight = FontWeight.Bold
                )

                if (entry.isRecommended) {
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(12.dp))
                            .background(CyberCyan.copy(alpha = 0.15f))
                            .border(1.dp, CyberCyan.copy(alpha = 0.5f), RoundedCornerShape(12.dp))
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    ) {
                        Text(
                            text = "RECOMMENDED",
                            style = MaterialTheme.typography.labelSmall,
                            color = CyberCyan,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            Text(
                text = entry.description,
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary
            )

            Row(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "📦 ${entry.downloadSizeStr}",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextTertiary
                )
                Text(
                    text = "•",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextTertiary
                )
                Text(
                    text = "Min: ${entry.minRamGb} GB RAM",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextTertiary
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                OutlinedButton(
                    onClick = onDownloadDriveClick,
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.outlinedButtonColors(
                        containerColor = SurfaceElevated,
                        contentColor = CyberCyan
                    ),
                    border = androidx.compose.foundation.BorderStroke(1.dp, BorderSubtle),
                    modifier = Modifier.weight(1f)
                ) {
                    Text("🔗 Drive Link", fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                }

                Button(
                    onClick = onImportClick,
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = CyberCyan, contentColor = BackgroundVoid),
                    modifier = Modifier.weight(1f)
                ) {
                    Text("📥 Import .zip", fontSize = 13.sp, fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}

@Composable
private fun PublicDriveFolderCard(onOpenFolder: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceElevated)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .clickable { onOpenFolder() }
            .padding(16.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(
                    text = "📁 Public Google Drive Repository",
                    style = MaterialTheme.typography.titleSmall,
                    color = TextPrimary,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Browse all pre-built forensic model packs in Drive",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextSecondary
                )
            }
            Text("→", fontSize = 20.sp, color = CyberCyan, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
private fun DeveloperSettingsCard(
    allowCustomKey: Boolean,
    onToggleAllowCustomKey: (Boolean) -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .padding(16.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                text = "DEVELOPER OPTIONS",
                style = MaterialTheme.typography.labelSmall,
                color = TextTertiary,
                fontWeight = FontWeight.Bold,
                letterSpacing = 1.sp
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Allow Developer / Custom Signed Packs",
                        style = MaterialTheme.typography.bodyMedium,
                        color = TextPrimary,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "Accepts packs signed with --test-key or your own keypair",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary
                    )
                }
                Switch(
                    checked = allowCustomKey,
                    onCheckedChange = onToggleAllowCustomKey,
                    colors = SwitchDefaults.colors(
                        checkedThumbColor = BackgroundVoid,
                        checkedTrackColor = WarningAmber,
                    )
                )
            }
        }
    }
}

@Composable
private fun ImportProgressCard(progressText: String) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceCard)
            .border(1.dp, CyberCyan, RoundedCornerShape(16.dp))
            .padding(20.dp)
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            CircularProgressIndicator(color = CyberCyan, modifier = Modifier.size(32.dp))
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(
                    text = "Importing Model Pack...",
                    style = MaterialTheme.typography.titleSmall,
                    color = TextPrimary,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = progressText,
                    style = MaterialTheme.typography.bodySmall,
                    color = CyberCyan
                )
            }
        }
    }
}

@Composable
private fun ErrorAlertCard(message: String, onDismiss: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(SyntheticRedContainer.copy(alpha = 0.3f))
            .border(1.dp, SyntheticRed, RoundedCornerShape(16.dp))
            .padding(16.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = "Import Error",
                    style = MaterialTheme.typography.titleSmall,
                    color = SyntheticRed,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = message,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextPrimary
                )
            }
            TextButton(onClick = onDismiss) {
                Text("Dismiss", color = SyntheticRed)
            }
        }
    }
}

@Composable
private fun SuccessAlertCard(message: String, onDismiss: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(AuthenticGreenContainer.copy(alpha = 0.3f))
            .border(1.dp, AuthenticGreen, RoundedCornerShape(16.dp))
            .padding(16.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = "Pack Verified & Active",
                    style = MaterialTheme.typography.titleSmall,
                    color = AuthenticGreen,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = message,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextPrimary
                )
            }
            TextButton(onClick = onDismiss) {
                Text("OK", color = AuthenticGreen)
            }
        }
    }
}

@Composable
private fun PrivacyAssuranceCard() {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceElevated)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .padding(16.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text("🔒", fontSize = 16.sp)
                Text(
                    text = "Zero-Network Guarantee",
                    style = MaterialTheme.typography.titleSmall,
                    color = AuthenticGreen,
                    fontWeight = FontWeight.Bold
                )
            }
            Text(
                text = "Onn-Detect never connects to the internet. Model packs are downloaded via your phone's browser and imported through Android's secure sandbox. Your photos and media never leave this device.",
                style = MaterialTheme.typography.bodySmall,
                color = TextSecondary
            )
        }
    }
}
