package com.onndetect.ui.screens.modelmanager

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import com.onndetect.ui.components.LiquidGlassCard
import com.onndetect.ui.navigation.Routes
import com.onndetect.ui.theme.*

@Composable
fun ModelManagerScreen(navController: NavController, viewModel: ModelManagerViewModel = hiltViewModel()) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    var confirmRemoval by remember { mutableStateOf(false) }
    val chatPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? -> uri?.let(viewModel::importChatZip) }
    val researchPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? -> uri?.let(viewModel::importZip) }
    // GGUF providers use several MIME types; the importer checks actual contents.
    val chooseModel = { chatPicker.launch(arrayOf("*/*")) }
    ModelManagerContent(
        state = state,
        onBack = { navController.popBackStack() },
        onChooseModel = chooseModel,
        onOpenChat = { navController.navigate(Routes.CHAT_RAG_BASE) {
            popUpTo(Routes.MODEL_MANAGER) { inclusive = true }
            launchSingleTop = true
        } },
        onRemove = { confirmRemoval = true },
        onCancelImport = viewModel::cancelImport,
        onDismissMessage = viewModel::clearMessages,
        onRefresh = viewModel::refreshInstalledPack,
        onResearchImport = { researchPicker.launch(arrayOf("application/zip", "application/x-zip-compressed", "application/octet-stream")) },
        onAllowCustomKey = viewModel::toggleAllowCustomKey,
    )
    if (confirmRemoval) AlertDialog(
        onDismissRequest = { confirmRemoval = false },
        title = { Text("Remove this model?") },
        text = { Text("Only the app's imported copy is removed. Your original file stays on your phone.") },
        confirmButton = { TextButton(onClick = { confirmRemoval = false; viewModel.removeChatModel() }) { Text("Remove", color = SyntheticRedLight) } },
        dismissButton = { TextButton(onClick = { confirmRemoval = false }) { Text("Keep model") } },
    )
}

/** State-only content also permits device screenshots and interaction verification. */
@Composable
fun ModelManagerContent(
    state: ModelManagerUiState,
    onBack: () -> Unit,
    onChooseModel: () -> Unit,
    onOpenChat: () -> Unit,
    onRemove: () -> Unit,
    onCancelImport: () -> Unit,
    onDismissMessage: () -> Unit,
    onRefresh: () -> Unit,
    onResearchImport: () -> Unit,
    onAllowCustomKey: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    var showDetails by remember { mutableStateOf(false) }
    var showResearch by remember { mutableStateOf(false) }
    val status = state.chatStatus
    val installed = status.installed
    Column(modifier.fillMaxSize().background(BackgroundVoid).safeDrawingPadding()) {
        Row(Modifier.fillMaxWidth().padding(start = 8.dp, end = 20.dp, top = 6.dp, bottom = 10.dp),
            verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack, modifier = Modifier.size(48.dp).semantics { contentDescription = "Back" }) {
                ModelIcon(ModelSymbol.BACK, TextPrimary)
            }
            Column(Modifier.weight(1f).padding(start = 4.dp)) {
                Text("Chat model", color = TextPrimary, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                Text("Private. On your phone.", color = TextTertiary, style = MaterialTheme.typography.bodySmall)
            }
        }
        Column(Modifier.weight(1f).fillMaxWidth().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp),
            horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Spacer(Modifier.height(6.dp))
            LiquidGlassCard(Modifier.widthIn(max = 600.dp).fillMaxWidth().testTag("chat_model_status"), cornerRadius = 24.dp) {
                Column(Modifier.padding(22.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        Box(Modifier.size(46.dp).background(CyberCyan.copy(alpha = .10f), RoundedCornerShape(15.dp)), contentAlignment = Alignment.Center) {
                            ModelIcon(if (installed == null) ModelSymbol.ADD else ModelSymbol.FILE, CyberCyan, Modifier.size(24.dp))
                        }
                        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                            Text("On-device conversation",
                                color = TextTertiary, style = MaterialTheme.typography.labelMedium)
                            Text(if (installed == null) "Add your chat model" else installed.displayName,
                                color = TextPrimary, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                        }
                    }
                    if (installed == null) {
                        Text("Choose a Qwen3.5 GGUF file or ZIP to start chatting. Add a matching vision model for image and video questions.",
                            color = TextSecondary, style = MaterialTheme.typography.bodyMedium)
                    } else {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                            ModelPill(if (status.isReady) "Ready" else "Needs attention", if (status.isReady) AuthenticGreenLight else WarningAmberLight)
                            ModelPill(if (installed.supportsVision) "Text + vision" else "Text chat", CyberCyan)
                        }
                        if (!status.isReady) Text(status.message, color = WarningAmberLight, style = MaterialTheme.typography.bodySmall)
                        installed.visionUnavailableReason?.let { note ->
                            Text(note, color = TextSecondary, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                    if (state.isImporting) {
                        ImportProgress(state.selectedFileName, state.importProgressText, onCancelImport)
                    } else if (installed != null && status.isReady) {
                        Button(onClick = onOpenChat, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp).testTag("chat_model_open_chat"),
                            shape = RoundedCornerShape(16.dp)) { Text("Start chatting", fontWeight = FontWeight.SemiBold) }
                        TextButton(onClick = onChooseModel, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp).testTag("chat_model_import")) {
                            Text("Choose another model", color = TextSecondary)
                        }
                    } else {
                        Button(onClick = onChooseModel, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp).testTag("chat_model_import"),
                            shape = RoundedCornerShape(16.dp)) {
                            ModelIcon(ModelSymbol.ADD, BackgroundVoid, Modifier.size(19.dp))
                            Spacer(Modifier.width(8.dp))
                            Text(if (installed == null) "Choose model file" else "Replace model", fontWeight = FontWeight.SemiBold)
                        }
                    }
                    if (installed == null && !state.isImporting) Text("GGUF or ZIP · Imported once · No account needed",
                        color = TextTertiary, style = MaterialTheme.typography.labelSmall)
                }
            }

            state.errorMessage?.let { message ->
                ModelImportNotice(state.selectedFileName, message, true, onDismissMessage,
                    onRetry = if (state.isImporting) null else onChooseModel)
            }
            state.successMessage?.takeIf { state.errorMessage == null }?.let { message ->
                Row(Modifier.widthIn(max = 600.dp).fillMaxWidth().padding(start = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(message, color = TextSecondary, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                    IconButton(onClick = onDismissMessage, modifier = Modifier.size(48.dp).semantics { contentDescription = "Dismiss message" }) {
                        ModelIcon(ModelSymbol.CLOSE, TextTertiary, Modifier.size(16.dp))
                    }
                }
            }

            Row(Modifier.widthIn(max = 600.dp).fillMaxWidth().padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Box(Modifier.size(6.dp).background(CyberCyan, RoundedCornerShape(3.dp)))
                Text("${state.deviceProfile.availableRamGb} GB free memory", color = TextSecondary, style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.weight(1f))
                Text("On-device CPU", color = TextTertiary, style = MaterialTheme.typography.labelSmall)
            }
            TextButton(onClick = { showDetails = !showDetails }, modifier = Modifier.widthIn(max = 600.dp).fillMaxWidth().heightIn(min = 48.dp)
                .testTag("chat_model_details")) {
                Text(if (showDetails) "Hide model details" else "Model details & help", color = TextSecondary)
                Spacer(Modifier.width(8.dp))
                ModelIcon(if (showDetails) ModelSymbol.UP else ModelSymbol.DOWN, TextTertiary, Modifier.size(15.dp))
            }
            if (showDetails) {
                LiquidGlassCard(Modifier.widthIn(max = 600.dp).fillMaxWidth(), cornerRadius = 18.dp) {
                    Column(Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Text("Supported files", color = TextPrimary, style = MaterialTheme.typography.titleSmall)
                        Text("Qwen3.5 language weights in GGUF format, or a ZIP containing them. A compatible vision projector adds image and sampled-video input.",
                            color = TextSecondary, style = MaterialTheme.typography.bodySmall)
                        installed?.let {
                            Text("Language: ${it.language.fileName}\nVision: ${it.projector?.fileName ?: "Not installed"}\nStorage: ${it.totalBytes / (1024 * 1024)} MB",
                                color = TextSecondary, style = MaterialTheme.typography.bodySmall)
                            it.importNotes.forEach { note -> Text(note, color = TextSecondary, style = MaterialTheme.typography.bodySmall) }
                            Text("File integrity is checked before each use. Model answers can make mistakes; a chat reply does not validate an authenticity result.",
                                color = TextTertiary, style = MaterialTheme.typography.bodySmall)
                            TextButton(onClick = onRemove, enabled = !state.isImporting, modifier = Modifier.heightIn(min = 48.dp)) {
                                Text("Remove imported model", color = SyntheticRedLight)
                            }
                        }
                        Text("This phone: ${state.deviceProfile.totalRamGb} GB RAM · ${state.deviceProfile.abi}. Speed depends on the model and your phone.",
                            color = TextTertiary, style = MaterialTheme.typography.bodySmall)
                        OutlinedButton(onClick = onRefresh, enabled = !state.isImporting, modifier = Modifier.heightIn(min = 48.dp)) { Text("Refresh status") }
                        HorizontalDivider(color = BorderSubtle)
                        TextButton(onClick = { showResearch = !showResearch }, modifier = Modifier.heightIn(min = 48.dp)) {
                            Text(if (showResearch) "Hide advanced research options" else "Advanced research options", color = TextTertiary)
                        }
                        if (showResearch) {
                            Text("Detector research packs are managed separately from chat models. Importing a pack does not validate its accuracy.",
                                color = TextSecondary, style = MaterialTheme.typography.bodySmall)
                            state.installedPack?.let { Text("Installed pack: ${it.packId}", color = TextTertiary, style = MaterialTheme.typography.bodySmall) }
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text("Allow custom research publisher", color = TextSecondary, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                                Switch(state.allowCustomKey, onCheckedChange = onAllowCustomKey, enabled = !state.isImporting)
                            }
                            OutlinedButton(onClick = onResearchImport, enabled = !state.isImporting, modifier = Modifier.heightIn(min = 48.dp)) { Text("Import research pack") }
                        }
                    }
                }
            }
            Text("Your conversations and files stay on this device.", color = TextTertiary, style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(horizontal = 8.dp, vertical = 8.dp))
            Spacer(Modifier.height(16.dp))
        }
    }
}

@Composable
private fun ImportProgress(name: String?, progress: String, onCancel: () -> Unit) {
    Column(Modifier.fillMaxWidth().testTag("chat_model_progress"), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        name?.let { Text(it, color = TextPrimary, style = MaterialTheme.typography.bodySmall, maxLines = 2, overflow = TextOverflow.Ellipsis) }
        LinearProgressIndicator(modifier = Modifier.fillMaxWidth(), color = CyberCyan, trackColor = BorderSubtle)
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(progress, color = TextSecondary, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
            TextButton(onClick = onCancel, modifier = Modifier.heightIn(min = 48.dp).testTag("chat_model_cancel")) { Text("Cancel", color = TextSecondary) }
        }
    }
}

@Composable
private fun ModelImportNotice(name: String?, message: String, error: Boolean, onDismiss: () -> Unit, onRetry: (() -> Unit)? = null) {
    val tint = if (error) SyntheticRedLight else AuthenticGreenLight
    LiquidGlassCard(Modifier.widthIn(max = 600.dp).fillMaxWidth().testTag("chat_model_notice"), accentColor = tint, cornerRadius = 18.dp) {
        Column(Modifier.padding(start = 16.dp, end = 8.dp, top = 8.dp, bottom = 12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(if (error) "Model wasn't added" else "All set", color = tint, style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                IconButton(onClick = onDismiss, modifier = Modifier.size(48.dp).semantics { contentDescription = "Dismiss message" }) {
                    ModelIcon(ModelSymbol.CLOSE, TextSecondary, Modifier.size(18.dp))
                }
            }
            name?.let { Text(it, color = TextPrimary, style = MaterialTheme.typography.labelMedium, maxLines = 2, overflow = TextOverflow.Ellipsis) }
            Text(message, color = TextSecondary, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(end = 8.dp))
            if (onRetry != null) TextButton(onClick = onRetry, modifier = Modifier.heightIn(min = 48.dp)) { Text("Choose another file", color = CyberCyan) }
        }
    }
}

@Composable
private fun ModelPill(text: String, tint: Color) {
    Text(text, color = tint, style = MaterialTheme.typography.labelSmall,
        modifier = Modifier.background(tint.copy(alpha = .09f), RoundedCornerShape(8.dp)).padding(horizontal = 9.dp, vertical = 5.dp))
}

private enum class ModelSymbol { BACK, ADD, FILE, CLOSE, UP, DOWN }

@Composable
private fun ModelIcon(symbol: ModelSymbol, tint: Color, modifier: Modifier = Modifier.size(22.dp)) {
    Canvas(modifier) {
        val width = size.width
        val height = size.height
        val stroke = 1.8.dp.toPx()
        fun line(x1: Float, y1: Float, x2: Float, y2: Float) = drawLine(tint,
            Offset(x1 * width, y1 * height), Offset(x2 * width, y2 * height), strokeWidth = stroke)
        when (symbol) {
            ModelSymbol.BACK -> { line(.78f, .5f, .2f, .5f); line(.2f, .5f, .46f, .24f); line(.2f, .5f, .46f, .76f) }
            ModelSymbol.ADD -> { line(.2f, .5f, .8f, .5f); line(.5f, .2f, .5f, .8f) }
            ModelSymbol.CLOSE -> { line(.22f, .22f, .78f, .78f); line(.78f, .22f, .22f, .78f) }
            ModelSymbol.UP -> { line(.2f, .64f, .5f, .34f); line(.5f, .34f, .8f, .64f) }
            ModelSymbol.DOWN -> { line(.2f, .36f, .5f, .66f); line(.5f, .66f, .8f, .36f) }
            ModelSymbol.FILE -> {
                drawPath(Path().apply {
                    moveTo(.24f * width, .12f * height); lineTo(.6f * width, .12f * height)
                    lineTo(.8f * width, .32f * height); lineTo(.8f * width, .88f * height)
                    lineTo(.24f * width, .88f * height); close()
                }, tint, style = Stroke(stroke))
                line(.6f, .12f, .6f, .34f); line(.6f, .34f, .8f, .34f)
                line(.38f, .54f, .66f, .54f); line(.38f, .69f, .66f, .69f)
            }
        }
    }
}
