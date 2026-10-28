package com.onndetect.ui.screens.enrollment

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.navigation.NavController
import com.onndetect.data.local.VoiceProfileEntity
import com.onndetect.ui.components.GlassBottomTab
import com.onndetect.ui.components.LiquidGlassCard
import com.onndetect.ui.components.LiquidGlassScaffold
import com.onndetect.ui.theme.*
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun VoiceEnrollmentScreen(
    navController: NavController,
    viewModel: VoiceEnrollmentViewModel = hiltViewModel(),
) {
    val profiles by viewModel.profiles.collectAsState()
    val uiState by viewModel.uiState.collectAsState()

    var showEnrollDialog by remember { mutableStateOf(false) }
    var selectedAudioUri by remember { mutableStateOf<Uri?>(null) }
    var contactNameInput by remember { mutableStateOf("") }
    var relationshipInput by remember { mutableStateOf("Family") }

    val enrollAudioPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) {
            selectedAudioUri = uri
            contactNameInput = ""
            relationshipInput = "Family"
            showEnrollDialog = true
        }
    }

    val verifyAudioPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) {
            viewModel.verifySuspiciousAudio(uri)
        }
    }

    LiquidGlassScaffold(
        activeTab = GlassBottomTab.VOICE_VAULT,
        onSelectTab = { tab ->
            if (tab != GlassBottomTab.VOICE_VAULT) {
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
                .padding(bottomBarPadding)
                .padding(horizontal = 20.dp, vertical = 14.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            // Top Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text("🛡️", fontSize = 20.sp)
                    Column {
                        Text(
                            text = "Family Safe-Voice Vault",
                            style = MaterialTheme.typography.headlineSmall,
                            fontWeight = FontWeight.Bold,
                            color = TextPrimary
                        )
                        Text(
                            text = "AES-256-GCM Voiceprints • Instant Clone Guard",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary,
                            fontSize = 11.sp
                        )
                    }
                }
            }

            // Explanatory Liquid Glass Card
            LiquidGlassCard(
                accentColor = AuthenticGreen,
                cornerRadius = 16.dp,
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Text(
                        text = "ZERO-NETWORK BIOMETRIC VAULT",
                        style = MaterialTheme.typography.labelSmall,
                        color = AuthenticGreen,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.sp
                    )
                    Text(
                        text = "Enroll clean voice samples (8s+ voiced) of family members. Test any suspicious WhatsApp voice note or call recording below to check if it matches their genuine voiceprint or is an AI clone.",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary,
                        lineHeight = 17.sp
                    )
                }
            }

            // Dual Action Row: [Enroll Family Voice] + [Verify Voice Note]
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Button(
                    onClick = { enrollAudioPicker.launch(arrayOf("audio/*")) },
                    modifier = Modifier
                        .weight(1f)
                        .height(48.dp),
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = CyberCyan,
                        contentColor = BackgroundVoid
                    ),
                    enabled = !uiState.isEnrolling && !uiState.isVerifying
                ) {
                    Text(
                        text = if (uiState.isEnrolling) "Enrolling..." else "🎙️ Enroll Voice",
                        fontWeight = FontWeight.Bold,
                        fontSize = 13.sp
                    )
                }

                Button(
                    onClick = { verifyAudioPicker.launch(arrayOf("audio/*")) },
                    modifier = Modifier
                        .weight(1f)
                        .height(48.dp),
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = AuthenticGreen,
                        contentColor = BackgroundVoid
                    ),
                    enabled = !uiState.isEnrolling && !uiState.isVerifying
                ) {
                    Text(
                        text = if (uiState.isVerifying) "Checking..." else "🔍 Verify Clip",
                        fontWeight = FontWeight.Bold,
                        fontSize = 13.sp
                    )
                }
            }

            // Quick Verification Result Card
            uiState.verifyResult?.let { vr ->
                val cardAccent = when {
                    vr.isSyntheticClone -> SyntheticRed
                    vr.matchReport.isMatch -> AuthenticGreen
                    else -> WarningAmber
                }
                LiquidGlassCard(
                    accentColor = cardAccent,
                    cornerRadius = 14.dp,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier.padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Text(
                            text = vr.headlineSummary,
                            style = MaterialTheme.typography.bodySmall,
                            color = TextPrimary,
                            fontWeight = FontWeight.SemiBold,
                            lineHeight = 17.sp
                        )
                        Text(
                            text = "LCNN Spoof Score: ${(vr.spoofScore * 100f).toInt()}% (thr ${(vr.spoofThreshold * 100f).toInt()}%) • Max Vault Match: ${(vr.matchReport.maxSimilarity * 100f).toInt()}%",
                            style = MaterialTheme.typography.labelSmall,
                            color = cardAccent
                        )
                    }
                }
            }

            // Status / Error Messages
            uiState.statusMessage?.let { msg ->
                Card(
                    colors = CardDefaults.cardColors(containerColor = CyberCyan.copy(alpha = 0.15f)),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Text(
                        text = msg,
                        color = CyberCyan,
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(12.dp)
                    )
                }
            }

            uiState.errorMessage?.let { err ->
                Card(
                    colors = CardDefaults.cardColors(containerColor = SyntheticRed.copy(alpha = 0.15f)),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Text(
                        text = err,
                        color = SyntheticRed,
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(12.dp)
                    )
                }
            }

            // Profiles Section Title
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "ENROLLED PROFILES (${profiles.size})",
                    style = MaterialTheme.typography.labelSmall,
                    color = TextTertiary,
                    letterSpacing = 1.sp
                )
            }

            // Profiles List
            if (profiles.isEmpty()) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text("🛡️", fontSize = 38.sp)
                        Text(
                            text = "No Voiceprints Enrolled",
                            style = MaterialTheme.typography.titleSmall,
                            color = TextPrimary,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            text = "Tap 'Enroll Voice' above with a genuine 8s+ voice note to protect your family.",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextTertiary,
                            fontSize = 12.sp
                        )
                    }
                }
            } else {
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    items(profiles, key = { it.id }) { profile ->
                        VoiceProfileCard(
                            profile = profile,
                            onDelete = { viewModel.deleteProfile(profile.id) }
                        )
                    }
                }
            }
        }
    }

    // Enrollment Dialog
    if (showEnrollDialog && selectedAudioUri != null) {
        AlertDialog(
            onDismissRequest = { showEnrollDialog = false },
            title = {
                Text(
                    text = "Enroll Family Member",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
            },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text(
                        text = "Enter profile details for this voice sample:",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary
                    )
                    OutlinedTextField(
                        value = contactNameInput,
                        onValueChange = { contactNameInput = it },
                        label = { Text("Contact Name (e.g. Mom)") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth()
                    )
                    OutlinedTextField(
                        value = relationshipInput,
                        onValueChange = { relationshipInput = it },
                        label = { Text("Relationship (e.g. Mother, Father, Child)") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        val uri = selectedAudioUri ?: return@Button
                        viewModel.enrollVoiceprint(uri, contactNameInput, relationshipInput)
                        showEnrollDialog = false
                    },
                    enabled = contactNameInput.isNotBlank()
                ) {
                    Text("Save Voiceprint")
                }
            },
            dismissButton = {
                TextButton(onClick = { showEnrollDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }
}

@Composable
private fun VoiceProfileCard(
    profile: VoiceProfileEntity,
    onDelete: () -> Unit,
) {
    val dateStr = remember(profile.enrolledAtMs) {
        SimpleDateFormat("MMM d, yyyy", Locale.US).format(Date(profile.enrolledAtMs))
    }

    LiquidGlassCard(
        accentColor = CyberCyan,
        cornerRadius = 14.dp,
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(42.dp)
                        .clip(CircleShape)
                        .background(CyberCyan.copy(alpha = 0.15f)),
                    contentAlignment = Alignment.Center
                ) {
                    Text("👤", fontSize = 18.sp)
                }

                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(
                        text = profile.contactName,
                        style = MaterialTheme.typography.titleSmall,
                        color = TextPrimary,
                        fontWeight = FontWeight.Bold
                    )
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Text(
                            text = profile.relationship,
                            style = MaterialTheme.typography.labelSmall,
                            color = CyberCyan
                        )
                        Text("•", style = MaterialTheme.typography.labelSmall, color = TextTertiary)
                        Text(
                            text = dateStr,
                            style = MaterialTheme.typography.labelSmall,
                            color = TextTertiary
                        )
                    }
                }
            }

            IconButton(onClick = onDelete) {
                Text("🗑️", fontSize = 16.sp)
            }
        }
    }
}
