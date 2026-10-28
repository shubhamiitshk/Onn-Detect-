package com.onndetect.ui.screens.chat

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import com.onndetect.rag.ForensicRagEngine
import com.onndetect.ui.components.GlassBottomTab
import com.onndetect.ui.components.LiquidGlassCard
import com.onndetect.ui.components.LiquidGlassScaffold
import com.onndetect.ui.navigation.Routes
import com.onndetect.ui.theme.*

@Composable
fun ChatRagScreen(
    navController: NavController,
    viewModel: ChatRagViewModel = hiltViewModel(),
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    var queryInput by remember { mutableStateOf("") }

    val imagePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        if (uri != null) viewModel.indexMedia(uri, ForensicRagEngine.RagMediaKind.IMAGE)
    }
    val videoPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) viewModel.indexMedia(uri, ForensicRagEngine.RagMediaKind.VIDEO)
    }
    val audioPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) viewModel.indexMedia(uri, ForensicRagEngine.RagMediaKind.AUDIO)
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
        Column(
            modifier = Modifier
                .fillMaxSize()
                .safeDrawingPadding()
                .padding(bottomBarPadding)
                .padding(horizontal = 18.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
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
                                .background(NeonPurple)
                        )
                        Text(
                            text = "Forensic RAG Chat",
                            style = MaterialTheme.typography.headlineSmall,
                            color = TextPrimary,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    Text(
                        text = "SigLIP-B/16 Embedder • Qwen 3.5 0.8B / Specialist RAG",
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary,
                        fontSize = 11.sp
                    )
                }

                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(16.dp))
                        .background(NeonPurple.copy(alpha = 0.18f))
                        .border(1.dp, NeonPurple.copy(alpha = 0.5f), RoundedCornerShape(16.dp))
                        .clickable { navController.navigate(Routes.MODEL_MANAGER) }
                        .padding(horizontal = 10.dp, vertical = 6.dp)
                ) {
                    Text(
                        text = "🧠 VLM Pack",
                        style = MaterialTheme.typography.labelSmall,
                        color = NeonPurpleLight,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            LiquidGlassCard(
                accentColor = NeonPurple,
                cornerRadius = 18.dp,
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = uiState.knowledgeBase?.mediaTitle ?: "SELECT MEDIA TO INDEX & QUERY",
                            style = MaterialTheme.typography.labelMedium,
                            color = if (uiState.knowledgeBase != null) CyberCyan else TextSecondary,
                            fontWeight = FontWeight.Bold
                        )
                        uiState.knowledgeBase?.let { kb ->
                            val verdictColor = if (kb.overallAiProb >= 0.578f) SyntheticRed else AuthenticGreen
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(verdictColor.copy(alpha = 0.18f))
                                    .padding(horizontal = 8.dp, vertical = 3.dp)
                            ) {
                                Text(
                                    text = "${(kb.overallAiProb * 100f).toInt()}% AI Risk",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = verdictColor,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        MediaAttachChip(
                            icon = "🎬",
                            label = "Video RAG",
                            accent = CyberCyan,
                            modifier = Modifier.weight(1f),
                            onClick = { videoPicker.launch(arrayOf("video/*")) }
                        )
                        MediaAttachChip(
                            icon = "🎙️",
                            label = "Audio RAG",
                            accent = NeonPurple,
                            modifier = Modifier.weight(1f),
                            onClick = { audioPicker.launch(arrayOf("audio/*")) }
                        )
                        MediaAttachChip(
                            icon = "📷",
                            label = "Image ROI",
                            accent = AuthenticGreen,
                            modifier = Modifier.weight(1f),
                            onClick = {
                                imagePicker.launch(
                                    PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                )
                            }
                        )
                    }

                    if (uiState.isIndexing) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(10.dp),
                            modifier = Modifier.padding(top = 4.dp)
                        ) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(18.dp),
                                color = CyberCyan,
                                strokeWidth = 2.dp
                            )
                            Text(
                                text = uiState.indexingProgressText,
                                style = MaterialTheme.typography.bodySmall,
                                color = CyberCyan
                            )
                        }
                    }
                }
            }

            uiState.knowledgeBase?.let { kb ->
                if (kb.chunks.size > 1) {
                    TimelineScrubberCard(
                        kb = kb,
                        selectedIndex = uiState.selectedChunkIndex,
                        onSelectChunk = { idx ->
                            viewModel.selectTimelineChunk(idx)
                        }
                    )
                }
            }

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

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                val quickPrompts = listOf(
                    "When did a boy come into the scene?",
                    "Where is the highest fake anomaly timestamp?",
                    "Does the voice match my Family Voice Vault?",
                    "Explain the forensic clues & ROI hotspot",
                )
                quickPrompts.forEach { prompt ->
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(16.dp))
                            .background(SurfaceElevated.copy(alpha = 0.75f))
                            .border(1.dp, BorderLight, RoundedCornerShape(16.dp))
                            .clickable { viewModel.askQuestion(prompt) }
                            .padding(horizontal = 12.dp, vertical = 7.dp)
                    ) {
                        Text(
                            text = prompt,
                            style = MaterialTheme.typography.labelSmall,
                            color = CyberCyanLight,
                            fontSize = 11.sp
                        )
                    }
                }
            }

            LazyColumn(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                if (uiState.messages.isEmpty()) {
                    item {
                        LiquidGlassCard(
                            accentColor = CyberCyan,
                            cornerRadius = 18.dp,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Column(
                                modifier = Modifier.padding(18.dp),
                                verticalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                Text(
                                    text = "100% Offline Multimodal Forensic RAG",
                                    style = MaterialTheme.typography.titleSmall,
                                    color = TextPrimary,
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    text = "Attach a video, voice note, or image above. ONN-DETECT indexes scene cuts, person/boy appearances, LCNN voice-clone scores, and Family Voice Vault matches locally so you can ask questions with zero cloud uploads.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = TextSecondary,
                                    lineHeight = 18.sp
                                )
                            }
                        }
                    }
                } else {
                    items(uiState.messages, key = { it.id }) { msg ->
                        ChatBubble(
                            msg = msg,
                            onTimestampClick = { ts -> viewModel.selectTimelineTimestamp(ts) }
                        )
                    }
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedTextField(
                    value = queryInput,
                    onValueChange = { queryInput = it },
                    placeholder = {
                        Text(
                            text = "Ask about video scene, timestamps, or voice...",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextTertiary
                        )
                    },
                    singleLine = true,
                    shape = RoundedCornerShape(18.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = CyberCyan,
                        unfocusedBorderColor = BorderLight,
                        focusedContainerColor = SurfaceElevated.copy(alpha = 0.8f),
                        unfocusedContainerColor = SurfaceCard.copy(alpha = 0.7f),
                        focusedTextColor = TextPrimary,
                        unfocusedTextColor = TextPrimary,
                    ),
                    modifier = Modifier.weight(1f)
                )

                Button(
                    onClick = {
                        val q = queryInput
                        queryInput = ""
                        viewModel.askQuestion(q)
                    },
                    enabled = queryInput.isNotBlank() && !uiState.isGenerating,
                    shape = RoundedCornerShape(16.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = CyberCyan,
                        contentColor = BackgroundVoid
                    ),
                    modifier = Modifier.height(52.dp)
                ) {
                    Text(
                        text = if (uiState.isGenerating) "..." else "Ask →",
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}

@Composable
private fun MediaAttachChip(
    icon: String,
    label: String,
    accent: Color,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(12.dp))
            .background(accent.copy(alpha = 0.14f))
            .border(1.dp, accent.copy(alpha = 0.45f), RoundedCornerShape(12.dp))
            .clickable { onClick() }
            .padding(vertical = 9.dp, horizontal = 8.dp),
        contentAlignment = Alignment.Center
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(text = icon, fontSize = 14.sp)
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = TextPrimary,
                fontWeight = FontWeight.Bold,
                fontSize = 11.sp
            )
        }
    }
}

@Composable
private fun TimelineScrubberCard(
    kb: ForensicRagEngine.IndexedMediaKnowledgeBase,
    selectedIndex: Int?,
    onSelectChunk: (Int) -> Unit,
) {
    val activeIndex = selectedIndex ?: 0
    val activeChunk = kb.chunks.getOrNull(activeIndex)

    LiquidGlassCard(
        accentColor = CyberCyan,
        cornerRadius = 16.dp,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            // Header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "DUAL-TRACK TIMELINE SCRUBBER (${kb.chunks.size} SEGMENTS)",
                    style = MaterialTheme.typography.labelSmall,
                    color = CyberCyan,
                    fontWeight = FontWeight.Bold,
                    fontSize = 10.sp
                )
                Text(
                    text = "Red: AI Spikes • Cyan: Semantic Pins",
                    style = MaterialTheme.typography.labelSmall,
                    color = TextTertiary,
                    fontSize = 9.sp
                )
            }

            // Track 1: Red Forensic Spikes (Visual Waveform / Spike Graph)
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text = "FORENSIC MANIPULATION SPIKES",
                        style = MaterialTheme.typography.labelSmall,
                        color = SyntheticRedLight,
                        fontSize = 9.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "Anomaly Thr: 58%",
                        style = MaterialTheme.typography.labelSmall,
                        color = WarningAmber,
                        fontSize = 9.sp
                    )
                }

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .background(SurfaceCard.copy(alpha = 0.85f))
                        .border(1.dp, BorderSubtle, RoundedCornerShape(8.dp))
                        .padding(horizontal = 8.dp, vertical = 4.dp),
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalAlignment = Alignment.Bottom
                ) {
                    kb.chunks.forEach { chunk ->
                        val isSelected = chunk.index == activeIndex
                        val isAnomaly = chunk.aiProbability >= 0.58f
                        val barColor = when {
                            isAnomaly -> SyntheticRed
                            chunk.hasPersonOrFace -> CyberCyan
                            else -> TextTertiary
                        }
                        val barHeightFraction = chunk.aiProbability.coerceIn(0.12f, 1f)

                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxHeight(barHeightFraction)
                                .clip(RoundedCornerShape(topStart = 4.dp, topEnd = 4.dp))
                                .background(if (isSelected) barColor else barColor.copy(alpha = 0.55f))
                                .border(
                                    width = if (isSelected) 1.5.dp else 0.5.dp,
                                    color = if (isSelected) Color.White else Color.Transparent,
                                    shape = RoundedCornerShape(topStart = 4.dp, topEnd = 4.dp)
                                )
                                .clickable { onSelectChunk(chunk.index) }
                        )
                    }
                }
            }

            // Track 2: Cyan Semantic Scene & Event Pins (Horizontal Scroller)
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(
                    text = "SEMANTIC RAG SCENE PINS",
                    style = MaterialTheme.typography.labelSmall,
                    color = CyberCyan,
                    fontSize = 9.sp,
                    fontWeight = FontWeight.SemiBold
                )

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    kb.chunks.forEach { chunk ->
                        val isSelected = chunk.index == activeIndex
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
                                    else SurfaceCard.copy(alpha = 0.7f)
                                )
                                .border(
                                    width = if (isSelected) 1.5.dp else 1.dp,
                                    color = if (isSelected) pinColor else pinColor.copy(alpha = 0.4f),
                                    shape = RoundedCornerShape(10.dp)
                                )
                                .clickable { onSelectChunk(chunk.index) }
                                .padding(horizontal = 9.dp, vertical = 6.dp)
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
                                    text = if (chunk.hasPersonOrFace) "👤 ${chunk.estimatedSubjectTag.take(8)}"
                                    else if (chunk.isSceneCut) "🎬 Cut"
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

            // Selected Segment Telemetry Inspector Box
            if (activeChunk != null) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(10.dp))
                        .background(SurfaceElevated.copy(alpha = 0.85f))
                        .border(1.dp, BorderLight, RoundedCornerShape(10.dp))
                        .padding(10.dp)
                ) {
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "⏱ [${activeChunk.formattedTimeRange}]",
                                style = MaterialTheme.typography.labelSmall,
                                color = CyberCyan,
                                fontWeight = FontWeight.Bold
                            )
                            val riskColor = if (activeChunk.aiProbability >= 0.58f) SyntheticRed else AuthenticGreen
                            Text(
                                text = "${(activeChunk.aiProbability * 100f).toInt()}% AI Risk (SRM ${(activeChunk.srmScore * 100).toInt()}%)",
                                style = MaterialTheme.typography.labelSmall,
                                color = riskColor,
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                        Text(
                            text = activeChunk.visualDescription,
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary,
                            fontSize = 11.sp,
                            lineHeight = 15.sp
                        )
                        if (activeChunk.audioSummary.isNotBlank()) {
                            Text(
                                text = "Audio: ${activeChunk.audioSummary}",
                                style = MaterialTheme.typography.bodySmall,
                                color = TextTertiary,
                                fontSize = 10.sp
                            )
                        }
                        if (activeChunk.hotspotBoxNorm != null || activeChunk.hotspotZScore >= 1.5f) {
                            Text(
                                text = "Hotspot ROI: ${String.format(java.util.Locale.US, "%.2fσ", activeChunk.hotspotZScore)} localized anomaly cluster",
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
    }
}

@Composable
private fun ChatBubble(
    msg: ChatMessage,
    onTimestampClick: (String) -> Unit = {},
) {
    val align = if (msg.isUser) Alignment.End else Alignment.Start
    val bubbleBrush = if (msg.isUser) {
        Brush.horizontalGradient(listOf(DeepIndigo.copy(alpha = 0.75f), CyberCyanDark.copy(alpha = 0.65f)))
    } else {
        Brush.verticalGradient(listOf(SurfaceElevated.copy(alpha = 0.90f), SurfaceCard.copy(alpha = 0.85f)))
    }
    val borderColor = if (msg.isUser) CyberCyan.copy(alpha = 0.5f) else NeonPurple.copy(alpha = 0.45f)

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = align
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 330.dp)
                .clip(RoundedCornerShape(16.dp))
                .background(bubbleBrush)
                .border(1.dp, borderColor, RoundedCornerShape(16.dp))
                .padding(14.dp)
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                msg.engineBadge?.let { badge ->
                    Text(
                        text = "⚡ $badge",
                        style = MaterialTheme.typography.labelSmall,
                        color = NeonPurpleLight,
                        fontWeight = FontWeight.Bold,
                        fontSize = 10.sp
                    )
                }

                Text(
                    text = msg.text,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextPrimary,
                    lineHeight = 18.sp
                )

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
                                    .clickable { onTimestampClick(ts) }
                                    .padding(horizontal = 8.dp, vertical = 4.dp)
                            ) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Text(
                                        text = "⏱",
                                        fontSize = 10.sp
                                    )
                                    Text(
                                        text = ts,
                                        style = MaterialTheme.typography.labelSmall,
                                        color = CyberCyan,
                                        fontWeight = FontWeight.Bold,
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
}
