package com.onndetect.ui.screens.chat

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import com.onndetect.rag.ForensicRagEngine
import com.onndetect.ui.components.GlassBottomTab
import com.onndetect.ui.components.LiquidGlassCard
import com.onndetect.ui.components.LiquidGlassPill
import com.onndetect.ui.components.LiquidGlassScaffold
import com.onndetect.ui.components.RunningCircleIndicator
import com.onndetect.ui.navigation.Routes
import com.onndetect.ui.theme.*
import java.util.Locale

@Composable
fun ChatRagScreen(
    navController: NavController,
    viewModel: ChatRagViewModel = hiltViewModel(),
) {
    val context = LocalContext.current
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    var inputQuery by remember { mutableStateOf("") }
    var showAttachmentMenu by remember { mutableStateOf(false) }

    val listState = rememberLazyListState()

    // Auto-scroll to bottom on new messages or when streaming at bottom
    LaunchedEffect(uiState.messages.size) {
        if (uiState.messages.isNotEmpty()) {
            listState.animateScrollToItem(uiState.messages.size - 1)
        }
    }

    LaunchedEffect(uiState.messages.lastOrNull()?.text?.length) {
        if (uiState.messages.isNotEmpty()) {
            val lastVisible = listState.layoutInfo.visibleItemsInfo.lastOrNull()?.index ?: 0
            val total = listState.layoutInfo.totalItemsCount
            if (lastVisible >= total - 2) {
                listState.scrollToItem(uiState.messages.size - 1)
            }
        }
    }

    // Attachment Activity Launchers
    val imagePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            val (name, size) = queryFileInfo(context, uri, "Photo ROI")
            viewModel.stageAttachment(
                uri = uri,
                kind = ForensicRagEngine.RagMediaKind.IMAGE,
                fileName = name,
                mimeType = "image/*",
                fileSizeFormatted = size,
            )
            showAttachmentMenu = false
        }
    }

    val videoPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) {
            val (name, size) = queryFileInfo(context, uri, "Video Clip")
            viewModel.stageAttachment(
                uri = uri,
                kind = ForensicRagEngine.RagMediaKind.VIDEO,
                fileName = name,
                mimeType = "video/*",
                fileSizeFormatted = size,
            )
            showAttachmentMenu = false
        }
    }

    val audioPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            val (name, size) = queryFileInfo(context, uri, "Audio Note")
            viewModel.stageAttachment(
                uri = uri,
                kind = ForensicRagEngine.RagMediaKind.AUDIO,
                fileName = name,
                mimeType = "audio/*",
                fileSizeFormatted = size,
            )
            showAttachmentMenu = false
        }
    }

    val documentPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION
                )
            }
            val (name, size) = queryFileInfo(context, uri, "Document File")
            viewModel.stageAttachment(
                uri = uri,
                kind = ForensicRagEngine.RagMediaKind.DOCUMENT,
                fileName = name,
                mimeType = context.contentResolver.getType(uri),
                fileSizeFormatted = size,
            )
            showAttachmentMenu = false
        }
    }

    LiquidGlassScaffold(
        activeTab = GlassBottomTab.CHAT_RAG,
        onSelectTab = { tab ->
            if (tab != GlassBottomTab.CHAT_RAG) {
                navController.navigate(tab.route) {
                    launchSingleTop = true
                }
            }
        }
    ) { bottomBarPadding ->
        val density = androidx.compose.ui.platform.LocalDensity.current
        val isImeVisible = WindowInsets.ime.getBottom(density) > 0
        val effectiveBottomPadding = if (isImeVisible) 4.dp else bottomBarPadding.calculateBottomPadding()

        Column(
            modifier = Modifier
                .fillMaxSize()
                .safeDrawingPadding()
                .padding(bottom = effectiveBottomPadding)
                .padding(horizontal = 16.dp, vertical = 6.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // =================================================================
            // Top App Bar: Model Badges & Chat Action Controls
            // =================================================================
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 4.dp),
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
                                .background(NeonPurple)
                        )
                        Text(
                            text = "Forensic AI Chat",
                            style = MaterialTheme.typography.titleMedium,
                            color = TextPrimary,
                            fontWeight = FontWeight.Bold
                        )
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(8.dp))
                                .background(AuthenticGreen.copy(alpha = 0.16f))
                                .border(1.dp, AuthenticGreen.copy(alpha = 0.45f), RoundedCornerShape(8.dp))
                                .padding(horizontal = 6.dp, vertical = 2.dp)
                        ) {
                            Text(
                                text = "100% Offline",
                                style = MaterialTheme.typography.labelSmall,
                                color = AuthenticGreenLight,
                                fontSize = 9.sp,
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                    }
                    Text(
                        text = "Qwen 3.5 0.8B • 20-ch SRM • LCNN Voice Vault",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextTertiary,
                        fontSize = 11.sp
                    )
                }

                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (uiState.messages.isNotEmpty()) {
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(14.dp))
                                .background(SurfaceCard.copy(alpha = 0.70f))
                                .border(1.dp, BorderLight, RoundedCornerShape(14.dp))
                                .clickable { viewModel.clearConversation() }
                                .padding(horizontal = 9.dp, vertical = 6.dp)
                        ) {
                            Text(
                                text = "🗑️ Clear",
                                style = MaterialTheme.typography.labelSmall,
                                color = TextSecondary,
                                fontSize = 11.sp
                            )
                        }
                    }

                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(14.dp))
                            .background(NeonPurple.copy(alpha = 0.18f))
                            .border(1.dp, NeonPurple.copy(alpha = 0.5f), RoundedCornerShape(14.dp))
                            .clickable { navController.navigate(Routes.MODEL_MANAGER) }
                            .padding(horizontal = 9.dp, vertical = 6.dp)
                    ) {
                        Text(
                            text = "🧠 VLM Pack",
                            style = MaterialTheme.typography.labelSmall,
                            color = NeonPurpleLight,
                            fontWeight = FontWeight.Bold,
                            fontSize = 11.sp
                        )
                    }
                }
            }

            // =================================================================
            // Indexed Media Timeline Scrubber (When Media is active)
            // =================================================================
            uiState.knowledgeBase?.let { kb ->
                LiquidGlassCard(
                    accentColor = CyberCyan,
                    cornerRadius = 16.dp,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                val kindIcon = when (kb.mediaKind) {
                                    ForensicRagEngine.RagMediaKind.VIDEO -> "🎬"
                                    ForensicRagEngine.RagMediaKind.AUDIO -> "🎙️"
                                    ForensicRagEngine.RagMediaKind.DOCUMENT -> "📄"
                                    else -> "📷"
                                }
                                Text(text = kindIcon, fontSize = 14.sp)
                                Text(
                                    text = kb.mediaTitle.take(28),
                                    style = MaterialTheme.typography.labelMedium,
                                    color = CyberCyan,
                                    fontWeight = FontWeight.Bold
                                )
                            }

                            val isRisk = kb.overallAiProb >= 0.578f
                            val verdictColor = if (isRisk) SyntheticRed else AuthenticGreen
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(verdictColor.copy(alpha = 0.20f))
                                    .border(1.dp, verdictColor.copy(alpha = 0.45f), RoundedCornerShape(8.dp))
                                    .padding(horizontal = 7.dp, vertical = 2.dp)
                            ) {
                                Text(
                                    text = "${(kb.overallAiProb * 100f).toInt()}% AI Risk",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = verdictColor,
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 10.sp
                                )
                            }
                        }

                        if (kb.chunks.size > 1) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .horizontalScroll(rememberScrollState()),
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                kb.chunks.forEach { chunk ->
                                    val isSelected = chunk.index == uiState.selectedChunkIndex
                                    val isAnomaly = chunk.aiProbability >= 0.58f
                                    val pinColor = when {
                                        isAnomaly -> SyntheticRed
                                        chunk.hasPersonOrFace -> CyberCyan
                                        else -> TextTertiary
                                    }
                                    Box(
                                        modifier = Modifier
                                            .clip(RoundedCornerShape(10.dp))
                                            .background(
                                                if (isSelected) pinColor.copy(alpha = 0.28f)
                                                else SurfaceCard.copy(alpha = 0.70f)
                                            )
                                            .border(
                                                width = if (isSelected) 1.5.dp else 1.dp,
                                                color = if (isSelected) pinColor else pinColor.copy(alpha = 0.35f),
                                                shape = RoundedCornerShape(10.dp)
                                            )
                                            .clickable { viewModel.selectTimelineChunk(chunk.index) }
                                            .padding(horizontal = 8.dp, vertical = 5.dp)
                                    ) {
                                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                            Text(
                                                text = ForensicRagEngine.formatMs(chunk.startTimeMs),
                                                style = MaterialTheme.typography.labelSmall,
                                                color = TextPrimary,
                                                fontWeight = FontWeight.Bold,
                                                fontSize = 10.sp
                                            )
                                            Text(
                                                text = if (chunk.hasPersonOrFace) "👤 ${chunk.estimatedSubjectTag.take(7)}"
                                                else "🎞️ ${(chunk.aiProbability * 100f).toInt()}%",
                                                style = MaterialTheme.typography.labelSmall,
                                                color = pinColor,
                                                fontSize = 9.sp
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Error Notice Banner
            uiState.errorMessage?.let { err ->
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(12.dp))
                        .background(SyntheticRedContainer.copy(alpha = 0.45f))
                        .border(1.dp, SyntheticRed, RoundedCornerShape(12.dp))
                        .padding(10.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = err,
                            style = MaterialTheme.typography.bodySmall,
                            color = TextPrimary,
                            modifier = Modifier.weight(1f)
                        )
                        Text(
                            text = "Dismiss",
                            style = MaterialTheme.typography.labelSmall,
                            color = SyntheticRedLight,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier
                                .padding(start = 8.dp)
                                .clickable { viewModel.clearError() }
                        )
                    }
                }
            }

            // =================================================================
            // Chat Messages Stream (ChatGPT Style LazyColumn)
            // =================================================================
            LazyColumn(
                state = listState,
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                contentPadding = PaddingValues(vertical = 6.dp)
            ) {
                if (uiState.messages.isEmpty()) {
                    item {
                        LiquidGlassCard(
                            accentColor = NeonPurple,
                            cornerRadius = 20.dp,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Column(
                                modifier = Modifier.padding(18.dp),
                                verticalArrangement = Arrangement.spacedBy(10.dp),
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                RunningCircleIndicator(
                                    size = 40.dp,
                                    primaryColor = CyberCyan,
                                    secondaryColor = NeonPurple
                                )
                                Text(
                                    text = "Forensic Multimodal AI",
                                    style = MaterialTheme.typography.titleMedium,
                                    color = TextPrimary,
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    text = "Ask questions, or attach any Video, Audio note, Photo, or KYC/UPI Document below. " +
                                        "Analyzed 100% locally with zero cloud telemetry.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = TextSecondary,
                                    lineHeight = 18.sp
                                )

                                Spacer(modifier = Modifier.height(4.dp))

                                val samplePrompts = listOf(
                                    "When did a boy enter the video scene?",
                                    "Does this voice match my Family Voice Vault?",
                                    "Check UPI screenshot for spliced font tampering",
                                    "Pinpoint spatial residual noise hotspot",
                                )
                                Column(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalArrangement = Arrangement.spacedBy(6.dp)
                                ) {
                                    samplePrompts.forEach { prompt ->
                                        Box(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .clip(RoundedCornerShape(12.dp))
                                                .background(SurfaceElevated.copy(alpha = 0.70f))
                                                .border(1.dp, BorderLight, RoundedCornerShape(12.dp))
                                                .clickable { viewModel.sendMessage(prompt) }
                                                .padding(horizontal = 12.dp, vertical = 8.dp)
                                        ) {
                                            Text(
                                                text = "💬 $prompt",
                                                style = MaterialTheme.typography.labelSmall,
                                                color = CyberCyanLight,
                                                fontSize = 11.sp
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    items(uiState.messages, key = { it.id }) { msg ->
                        ChatBubbleRow(
                            msg = msg,
                            onSelectTimestamp = { ts ->
                                viewModel.selectTimelineChunkForTimestamp(ts)
                            }
                        )
                    }
                }

                // Running Circle Live Thinking Indicator
                if (uiState.isGenerating && uiState.streamingStatus != null) {
                    item {
                        LiquidGlassThinkingIndicatorRow(
                            status = uiState.streamingStatus ?: "Thinking...",
                            onStop = { viewModel.cancelGeneration() }
                        )
                    }
                }
            }

            // Quick Prompt Horizontal Chips Bar (when messages already present)
            if (uiState.messages.isNotEmpty()) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    val followUps = listOf(
                        "Explain forensic clues & ROI hotspot",
                        "When does the subject first appear?",
                        "Is this voice clone spoofed?",
                        "Check ELA compression artifacts",
                    )
                    followUps.forEach { prompt ->
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(14.dp))
                                .background(SurfaceCard.copy(alpha = 0.75f))
                                .border(1.dp, BorderLight.copy(alpha = 0.5f), RoundedCornerShape(14.dp))
                            .clickable { viewModel.sendMessage(prompt) }
                            .padding(horizontal = 10.dp, vertical = 5.dp)
                        ) {
                            Text(
                                text = prompt,
                                style = MaterialTheme.typography.labelSmall,
                                color = CyberCyanLight,
                                fontSize = 10.sp
                            )
                        }
                    }
                }
            }

            // =================================================================
            // Attachment Popover Tray (Image, Video, Audio, Document)
            // =================================================================
            AnimatedVisibility(
                visible = showAttachmentMenu,
                enter = expandVertically() + fadeIn(),
                exit = shrinkVertically() + fadeOut()
            ) {
                LiquidGlassCard(
                    accentColor = NeonPurple,
                    cornerRadius = 18.dp,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(10.dp),
                        horizontalArrangement = Arrangement.SpaceEvenly,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        AttachmentTypeButton(
                            icon = "📷",
                            label = "Image",
                            accent = AuthenticGreen,
                            onClick = {
                                imagePicker.launch(
                                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                )
                            }
                        )
                        AttachmentTypeButton(
                            icon = "🎬",
                            label = "Video",
                            accent = CyberCyan,
                            onClick = {
                                videoPicker.launch(
                                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)
                                )
                            }
                        )
                        AttachmentTypeButton(
                            icon = "🎙️",
                            label = "Audio",
                            accent = NeonPurple,
                            onClick = { audioPicker.launch(arrayOf("audio/*")) }
                        )
                        AttachmentTypeButton(
                            icon = "📄",
                            label = "Document",
                            accent = WarningAmber,
                            onClick = {
                                documentPicker.launch(
                                    arrayOf(
                                        "application/pdf",
                                        "text/*",
                                        "application/json",
                                        "image/*",
                                        "application/octet-stream"
                                    )
                                )
                            }
                        )
                    }
                }
            }

            // =================================================================
            // Composer Capsule: Staged Attachment Preview + Text Input + Send
            // =================================================================
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(26.dp))
                    .background(
                        Brush.verticalGradient(
                            listOf(
                                SurfaceElevated.copy(alpha = 0.88f),
                                SurfaceCard.copy(alpha = 0.80f),
                            )
                        )
                    )
                    .border(
                        width = 1.2.dp,
                        brush = Brush.horizontalGradient(
                            listOf(
                                Color.White.copy(alpha = 0.28f),
                                CyberCyan.copy(alpha = 0.38f),
                                NeonPurple.copy(alpha = 0.30f),
                            )
                        ),
                        shape = RoundedCornerShape(26.dp)
                    )
                    .padding(horizontal = 8.dp, vertical = 6.dp)
            ) {
                Column(
                    modifier = Modifier.fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    // Staged Attachment Preview Chip (inside composer)
                    uiState.stagedAttachment?.let { att ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(12.dp))
                                .background(DeepIndigo.copy(alpha = 0.22f))
                                .border(1.dp, CyberCyan.copy(alpha = 0.40f), RoundedCornerShape(12.dp))
                                .padding(horizontal = 10.dp, vertical = 6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Row(
                                modifier = Modifier.weight(1f),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                val icon = when (att.kind) {
                                    ForensicRagEngine.RagMediaKind.VIDEO -> "🎬"
                                    ForensicRagEngine.RagMediaKind.AUDIO -> "🎙️"
                                    ForensicRagEngine.RagMediaKind.DOCUMENT -> "📄"
                                    else -> "📷"
                                }
                                Text(text = icon, fontSize = 16.sp)
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        text = att.fileName,
                                        style = MaterialTheme.typography.labelSmall,
                                        color = TextPrimary,
                                        fontWeight = FontWeight.Bold,
                                        maxLines = 1,
                                        overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis
                                    )
                                    att.fileSizeFormatted?.let { size ->
                                        Text(
                                            text = size,
                                            style = MaterialTheme.typography.labelSmall,
                                            color = TextTertiary,
                                            fontSize = 9.sp
                                        )
                                    }
                                }
                            }

                            Box(
                                modifier = Modifier
                                    .size(24.dp)
                                    .clip(CircleShape)
                                    .background(SurfaceCard.copy(alpha = 0.8f))
                                    .clickable { viewModel.clearStagedAttachment() },
                                contentAlignment = Alignment.Center
                            ) {
                                Text(text = "✕", color = TextSecondary, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                            }
                        }
                    }

                    // Input Row: Attachment Button + TextField + Send/Stop Button
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        // Attachment Plus Button
                        Box(
                            modifier = Modifier
                                .size(40.dp)
                                .clip(CircleShape)
                                .background(
                                    if (showAttachmentMenu) NeonPurple.copy(alpha = 0.25f)
                                    else SurfaceHighlight.copy(alpha = 0.6f)
                                )
                                .border(
                                    1.dp,
                                    if (showAttachmentMenu) NeonPurple else BorderLight,
                                    CircleShape
                                )
                                .clickable { showAttachmentMenu = !showAttachmentMenu },
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = if (showAttachmentMenu) "✕" else "+",
                                color = if (showAttachmentMenu) NeonPurpleLight else CyberCyan,
                                fontSize = 18.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        // Text Input
                        OutlinedTextField(
                            value = inputQuery,
                            onValueChange = { inputQuery = it },
                            placeholder = {
                                Text(
                                    text = if (uiState.stagedAttachment != null) "Ask about this attachment..."
                                    else "Ask anything, or attach media...",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = TextTertiary,
                                    fontSize = 13.sp
                                )
                            },
                            maxLines = 4,
                            shape = RoundedCornerShape(20.dp),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Color.Transparent,
                                unfocusedBorderColor = Color.Transparent,
                                focusedContainerColor = Color.Transparent,
                                unfocusedContainerColor = Color.Transparent,
                                focusedTextColor = TextPrimary,
                                unfocusedTextColor = TextPrimary,
                                cursorColor = CyberCyan,
                            ),
                            modifier = Modifier.weight(1f)
                        )

                        // Circular Send / Stop Button
                        val hasPayload = inputQuery.isNotBlank() || uiState.stagedAttachment != null
                        if (uiState.isGenerating) {
                            // Stop Generation Button
                            Box(
                                modifier = Modifier
                                    .size(42.dp)
                                    .clip(CircleShape)
                                    .background(SyntheticRed.copy(alpha = 0.25f))
                                    .border(1.2.dp, SyntheticRed, CircleShape)
                                    .clickable { viewModel.cancelGeneration() },
                                contentAlignment = Alignment.Center
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(12.dp)
                                        .background(SyntheticRed, RoundedCornerShape(2.dp))
                                )
                            }
                        } else {
                            // Send Button
                            val sendBrush = if (hasPayload) {
                                Brush.horizontalGradient(listOf(CyberCyan, DeepIndigo))
                            } else {
                                Brush.horizontalGradient(listOf(SurfaceHighlight.copy(alpha = 0.4f), SurfaceCard.copy(alpha = 0.4f)))
                            }
                            Box(
                                modifier = Modifier
                                    .size(42.dp)
                                    .clip(CircleShape)
                                    .background(sendBrush)
                                    .border(
                                        1.dp,
                                        if (hasPayload) CyberCyanLight else BorderLight.copy(alpha = 0.4f),
                                        CircleShape
                                    )
                                    .clickable(enabled = hasPayload) {
                                        val q = inputQuery
                                        inputQuery = ""
                                        viewModel.sendMessage(q)
                                    },
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = "↑",
                                    color = if (hasPayload) BackgroundVoid else TextDisabled,
                                    fontSize = 20.sp,
                                    fontWeight = FontWeight.Black
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Chat Bubble Row (ChatGPT Style: User right, Bot left)
// =============================================================================
@Composable
private fun ChatBubbleRow(
    msg: ChatMessage,
    onSelectTimestamp: (String) -> Unit,
) {
    val clipboardManager = LocalClipboardManager.current
    var isCopied by remember { mutableStateOf(false) }

    LaunchedEffect(isCopied) {
        if (isCopied) {
            kotlinx.coroutines.delay(1800)
            isCopied = false
        }
    }

    val align = if (msg.isUser) Alignment.End else Alignment.Start
    val bubbleBrush = if (msg.isUser) {
        Brush.horizontalGradient(listOf(DeepIndigo.copy(alpha = 0.85f), CyberCyanDark.copy(alpha = 0.75f)))
    } else {
        Brush.verticalGradient(listOf(SurfaceElevated.copy(alpha = 0.94f), SurfaceCard.copy(alpha = 0.90f)))
    }
    val borderColor = if (msg.isUser) CyberCyan.copy(alpha = 0.55f) else NeonPurple.copy(alpha = 0.45f)

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = align
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 340.dp)
                .clip(
                    RoundedCornerShape(
                        topStart = 18.dp,
                        topEnd = 18.dp,
                        bottomStart = if (msg.isUser) 18.dp else 4.dp,
                        bottomEnd = if (msg.isUser) 4.dp else 18.dp
                    )
                )
                .background(bubbleBrush)
                .border(
                    width = 1.dp,
                    color = borderColor,
                    shape = RoundedCornerShape(
                        topStart = 18.dp,
                        topEnd = 18.dp,
                        bottomStart = if (msg.isUser) 18.dp else 4.dp,
                        bottomEnd = if (msg.isUser) 4.dp else 18.dp
                    )
                )
                .padding(14.dp)
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                // Attached Media Card inside User Bubble
                msg.attachment?.let { att ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(10.dp))
                            .background(BackgroundVoid.copy(alpha = 0.35f))
                            .border(1.dp, CyberCyanLight.copy(alpha = 0.35f), RoundedCornerShape(10.dp))
                            .padding(horizontal = 8.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        val icon = when (att.kind) {
                            ForensicRagEngine.RagMediaKind.VIDEO -> "🎬"
                            ForensicRagEngine.RagMediaKind.AUDIO -> "🎙️"
                            ForensicRagEngine.RagMediaKind.DOCUMENT -> "📄"
                            else -> "📷"
                        }
                        Text(text = icon, fontSize = 14.sp)
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = att.fileName,
                                style = MaterialTheme.typography.labelSmall,
                                color = TextPrimary,
                                fontWeight = FontWeight.Bold,
                                fontSize = 11.sp,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                            Text(
                                text = "Indexed On-Device",
                                style = MaterialTheme.typography.labelSmall,
                                color = CyberCyanLight,
                                fontSize = 9.sp
                            )
                        }
                    }
                }

                // Assistant Badge Header & Copy Button
                if (!msg.isUser) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(6.dp)
                                    .clip(CircleShape)
                                    .background(NeonPurpleLight)
                            )
                            Text(
                                text = msg.engineBadge ?: "Forensic AI",
                                style = MaterialTheme.typography.labelSmall,
                                color = NeonPurpleLight,
                                fontWeight = FontWeight.Bold,
                                fontSize = 10.sp
                            )
                        }

                        if (!msg.isStreaming && msg.text.isNotBlank()) {
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(SurfaceCard.copy(alpha = 0.70f))
                                    .border(0.8.dp, BorderLight.copy(alpha = 0.5f), RoundedCornerShape(8.dp))
                                    .clickable {
                                        clipboardManager.setText(AnnotatedString(msg.text))
                                        isCopied = true
                                    }
                                    .padding(horizontal = 6.dp, vertical = 2.dp)
                            ) {
                                Text(
                                    text = if (isCopied) "✓ Copied" else "📋 Copy",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = if (isCopied) AuthenticGreenLight else TextTertiary,
                                    fontSize = 9.sp,
                                    fontWeight = FontWeight.Medium
                                )
                            }
                        }
                    }
                }

                // Message Text with Rich Markdown Formatting and Inline Typewriter Cursor
                val annotatedText = formatMarkdownText(msg.text, msg.isStreaming)
                Text(
                    text = annotatedText,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextPrimary,
                    lineHeight = 19.sp,
                    fontSize = 13.sp
                )

                // Cited Timestamps Badges
                if (msg.citedTimestamps.isNotEmpty()) {
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        modifier = Modifier.horizontalScroll(rememberScrollState())
                    ) {
                        msg.citedTimestamps.forEach { ts ->
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(CyberCyan.copy(alpha = 0.16f))
                                    .border(1.dp, CyberCyan.copy(alpha = 0.45f), RoundedCornerShape(8.dp))
                                    .clickable { onSelectTimestamp(ts) }
                                    .padding(horizontal = 7.dp, vertical = 3.dp)
                            ) {
                                Text(
                                    text = "⏱ $ts",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = CyberCyan,
                                    fontWeight = FontWeight.SemiBold,
                                    fontSize = 10.sp
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun formatMarkdownText(
    text: String,
    isStreaming: Boolean,
): AnnotatedString {
    val infiniteTransition = rememberInfiniteTransition(label = "cursor")
    val cursorAlpha by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(500, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "cursorAlpha"
    )

    return buildAnnotatedString {
        var i = 0
        while (i < text.length) {
            when {
                text.startsWith("**", i) -> {
                    val endBold = text.indexOf("**", i + 2)
                    if (endBold != -1) {
                        withStyle(SpanStyle(fontWeight = FontWeight.Bold, color = TextPrimary)) {
                            append(text.substring(i + 2, endBold))
                        }
                        i = endBold + 2
                    } else {
                        append(text.substring(i))
                        break
                    }
                }
                text.startsWith("`", i) -> {
                    val endCode = text.indexOf("`", i + 1)
                    if (endCode != -1) {
                        withStyle(
                            SpanStyle(
                                fontFamily = FontFamily.Monospace,
                                color = CyberCyanLight,
                                background = DeepIndigo.copy(alpha = 0.35f)
                            )
                        ) {
                            append(" ${text.substring(i + 1, endCode)} ")
                        }
                        i = endCode + 1
                    } else {
                        append(text.substring(i))
                        break
                    }
                }
                else -> {
                    val nextBold = text.indexOf("**", i)
                    val nextCode = text.indexOf("`", i)
                    val nextSpecial = listOf(nextBold, nextCode).filter { it != -1 }.minOrNull() ?: text.length
                    append(text.substring(i, nextSpecial))
                    i = nextSpecial
                }
            }
        }

        if (isStreaming) {
            withStyle(
                SpanStyle(
                    color = CyberCyan.copy(alpha = if (cursorAlpha > 0.40f) 1f else 0.12f),
                    fontWeight = FontWeight.Black
                )
            ) {
                append(" ▌")
            }
        }
    }
}

// =============================================================================
// Live Thinking Indicator Row with Running Circle
// =============================================================================
@Composable
private fun LiquidGlassThinkingIndicatorRow(
    status: String,
    onStop: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.Start
    ) {
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(16.dp))
                .background(SurfaceElevated.copy(alpha = 0.85f))
                .border(1.dp, NeonPurple.copy(alpha = 0.40f), RoundedCornerShape(16.dp))
                .padding(horizontal = 12.dp, vertical = 8.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                RunningCircleIndicator(
                    size = 18.dp,
                    strokeWidth = 2.dp,
                    primaryColor = CyberCyan,
                    secondaryColor = NeonPurple
                )
                Text(
                    text = status,
                    style = MaterialTheme.typography.bodySmall,
                    color = CyberCyanLight,
                    fontSize = 11.sp
                )
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(SurfaceCard.copy(alpha = 0.6f))
                        .clickable { onStop() }
                        .padding(horizontal = 6.dp, vertical = 2.dp)
                ) {
                    Text(
                        text = "Stop",
                        style = MaterialTheme.typography.labelSmall,
                        color = SyntheticRedLight,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}

// =============================================================================
// Attachment Sheet Button
// =============================================================================
@Composable
private fun AttachmentTypeButton(
    icon: String,
    label: String,
    accent: Color,
    onClick: () -> Unit,
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(4.dp),
        modifier = Modifier
            .clip(RoundedCornerShape(12.dp))
            .clickable { onClick() }
            .padding(8.dp)
    ) {
        Box(
            modifier = Modifier
                .size(44.dp)
                .clip(CircleShape)
                .background(accent.copy(alpha = 0.18f))
                .border(1.2.dp, accent.copy(alpha = 0.50f), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Text(text = icon, fontSize = 20.sp)
        }
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = TextPrimary,
            fontSize = 11.sp,
            fontWeight = FontWeight.SemiBold
        )
    }
}

// Helper: Query display name and formatted file size from content URI
private fun queryFileInfo(context: Context, uri: Uri, fallbackName: String): Pair<String, String?> {
    var name = fallbackName
    var sizeStr: String? = null
    runCatching {
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val nameIdx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            val sizeIdx = cursor.getColumnIndex(OpenableColumns.SIZE)
            if (cursor.moveToFirst()) {
                if (nameIdx != -1) {
                    cursor.getString(nameIdx)?.takeIf { it.isNotBlank() }?.let { name = it }
                }
                if (sizeIdx != -1) {
                    val bytes = cursor.getLong(sizeIdx)
                    if (bytes > 0) {
                        sizeStr = when {
                            bytes >= 1024 * 1024 -> String.format(Locale.US, "%.1f MB", bytes / (1024.0 * 1024.0))
                            bytes >= 1024 -> String.format(Locale.US, "%.0f KB", bytes / 1024.0)
                            else -> "$bytes B"
                        }
                    }
                }
            }
        }
    }
    return Pair(name, sizeStr)
}