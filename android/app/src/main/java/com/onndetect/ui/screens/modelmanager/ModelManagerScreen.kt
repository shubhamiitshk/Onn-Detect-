package com.onndetect.ui.screens.modelmanager

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import com.onndetect.chat.ChatModelStore
import com.onndetect.ui.components.LiquidGlassCard
import com.onndetect.ui.theme.*

@Composable
fun ModelManagerScreen(navController: NavController, viewModel: ModelManagerViewModel = hiltViewModel()) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    var showResearch by remember { mutableStateOf(false) }
    var confirmRemoval by remember { mutableStateOf(false) }
    val chatPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? -> uri?.let(viewModel::importChatZip) }
    val researchPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? -> uri?.let(viewModel::importZip) }
    val types = arrayOf("application/zip", "application/x-zip-compressed", "application/octet-stream")

    Column(Modifier.fillMaxSize().background(BackgroundVoid).safeDrawingPadding().verticalScroll(rememberScrollState())
        .padding(horizontal = 20.dp, vertical = 16.dp), verticalArrangement = Arrangement.spacedBy(18.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = { navController.popBackStack() }) { Text("←", color = TextPrimary, style = MaterialTheme.typography.titleLarge) }
            Column(Modifier.weight(1f)) {
                Text("Local models", color = TextPrimary, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
                Text("Choose what powers your conversations", color = TextSecondary, style = MaterialTheme.typography.bodySmall)
            }
        }

        ChatModelCard(state.chatStatus, state.isImporting,
            onImport = { chatPicker.launch(types) }, onRemove = { confirmRemoval = true })

        if (state.isImporting) LiquidGlassCard(Modifier.fillMaxWidth()) {
            Row(Modifier.padding(18.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                CircularProgressIndicator(Modifier.size(25.dp), color = CyberCyan, strokeWidth = 2.dp)
                Text(state.importProgressText, color = TextPrimary, style = MaterialTheme.typography.bodyMedium)
            }
        }
        state.errorMessage?.let { message -> NoticeCard("Could not complete this action", message, true, viewModel::clearMessages) }
        state.successMessage?.let { message -> NoticeCard("Ready", message, false, viewModel::clearMessages) }

        LiquidGlassCard(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("How the models work together", color = TextPrimary, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                Text("Conversation model: answers normal questions and discusses attached media and reports.", color = TextSecondary, style = MaterialTheme.typography.bodyMedium)
                Text("Bundled detectors: SRM, SigLIP and the speech model measure forensic signals. A chat answer does not replace their validation or turn uncertainty into proof.", color = TextSecondary, style = MaterialTheme.typography.bodyMedium)
                Text("Embedding models organize search results; they do not generate chat replies.", color = TextSecondary, style = MaterialTheme.typography.bodyMedium)
            }
        }

        LiquidGlassCard(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("This device", color = TextPrimary, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                Text("${state.deviceProfile.totalRamGb} GB total · ${state.deviceProfile.availableRamGb} GB currently available", color = TextSecondary, style = MaterialTheme.typography.bodyMedium)
                Text("CPU inference · ${state.deviceProfile.abi}", color = CyberCyan, style = MaterialTheme.typography.bodySmall)
                Text("Speed and memory use depend on your phone, input size and model. The app does not claim GPU/NPU acceleration or a fixed token rate.", color = TextSecondary, style = MaterialTheme.typography.bodySmall)
                OutlinedButton(onClick = viewModel::refreshInstalledPack, enabled = !state.isImporting) { Text("Refresh availability") }
            }
        }

        TextButton(onClick = { showResearch = !showResearch }) { Text(if (showResearch) "Hide forensic research options" else "Forensic research options", color = TextSecondary) }
        if (showResearch) LiquidGlassCard(Modifier.fillMaxWidth(), accentColor = WarningAmber) {
            Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Separate research pack", color = TextPrimary, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                Text("These packs include an experimental forensic head and policy. They are separate from conversation weights. Importing one does not validate detector accuracy.", color = TextSecondary, style = MaterialTheme.typography.bodySmall)
                state.installedPack?.let { Text("Recorded pack: ${it.packId}\nDeclared model: ${it.modelId}", color = TextSecondary, style = MaterialTheme.typography.bodySmall) }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Allow custom research publisher", color = TextSecondary, modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
                    Switch(state.allowCustomKey, onCheckedChange = viewModel::toggleAllowCustomKey, enabled = !state.isImporting)
                }
                OutlinedButton(onClick = { researchPicker.launch(types) }, enabled = !state.isImporting) { Text("Import forensic research pack") }
            }
        }
        Text("Model files are imported from your device. The app has no internet permission and sends no conversations or media to a server.",
            color = TextTertiary, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(bottom = 12.dp))
    }
    if (confirmRemoval) AlertDialog(
        onDismissRequest = { confirmRemoval = false },
        title = { Text("Remove conversation model?") },
        text = { Text("This removes the app's imported conversation copy. Original model files and bundled detectors are kept.") },
        confirmButton = { TextButton(onClick = { confirmRemoval = false; viewModel.removeChatModel() }) { Text("Remove") } },
        dismissButton = { TextButton(onClick = { confirmRemoval = false }) { Text("Keep model") } },
    )
}

@Composable
private fun ChatModelCard(status: ChatModelStore.Status, busy: Boolean, onImport: () -> Unit, onRemove: () -> Unit) {
    LiquidGlassCard(Modifier.fillMaxWidth().testTag("chat_model_status")) {
        Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("CONVERSATION MODEL", color = CyberCyan, style = MaterialTheme.typography.labelSmall)
            Text(status.modelName, color = TextPrimary, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            Text(status.message, color = if (status.isReady) AuthenticGreenLight else TextSecondary, style = MaterialTheme.typography.bodyMedium)
            status.installed?.let { installed ->
                Text("Language: ${installed.language.fileName}\nVision: ${installed.projector?.fileName ?: "No projector · text only"}\nStorage: ${installed.totalBytes / (1024 * 1024)} MB",
                    color = TextSecondary, style = MaterialTheme.typography.bodySmall)
                Text("SHA-256 checked before each use · imported by you", color = TextTertiary, style = MaterialTheme.typography.bodySmall)
                Text("A projector enables visual input; it does not establish detection accuracy or validate this pairing's answers.", color = TextSecondary, style = MaterialTheme.typography.bodySmall)
            }
            if (status.installed == null) Text("Use a chat pack containing Qwen3.5 language weights and, for visual questions, its compatible vision projector. A forensic head is not needed for normal chat.",
                color = TextSecondary, style = MaterialTheme.typography.bodyMedium)
            Button(onClick = onImport, enabled = !busy, modifier = Modifier.fillMaxWidth().testTag("chat_model_import"), shape = RoundedCornerShape(16.dp)) {
                Text(if (status.installed == null) "Import conversation pack" else "Replace conversation model")
            }
            if (status.installed != null) TextButton(onClick = onRemove, enabled = !busy, modifier = Modifier.align(Alignment.End)) { Text("Remove imported model", color = SyntheticRedLight) }
        }
    }
}

@Composable
private fun NoticeCard(title: String, message: String, error: Boolean, onDismiss: () -> Unit) {
    LiquidGlassCard(Modifier.fillMaxWidth(), accentColor = if (error) SyntheticRed else AuthenticGreen) {
        Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text(title, color = TextPrimary, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            Text(message, color = TextSecondary, style = MaterialTheme.typography.bodyMedium)
            TextButton(onClick = onDismiss, modifier = Modifier.align(Alignment.End)) { Text("Dismiss") }
        }
    }
}
