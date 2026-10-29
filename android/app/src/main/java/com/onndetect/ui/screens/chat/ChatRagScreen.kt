package com.onndetect.ui.screens.chat

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.OpenableColumns
import android.widget.MediaController
import android.widget.VideoView
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.interaction.collectIsDraggedAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.hilt.lifecycle.viewmodel.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavController
import com.onndetect.rag.ForensicRagEngine
import com.onndetect.ui.components.GlassBottomTab
import com.onndetect.ui.components.LiquidGlassScaffold
import com.onndetect.ui.navigation.Routes
import com.onndetect.ui.screens.scanner.cancellableProviderWork
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.util.Locale

private val ChatInk = Color(0xFFF0F5FA)
private val ChatMuted = Color(0xFF9AA9BA)
private val ChatFaint = Color(0xFF748396)
private val ChatAccent = Color(0xFFA7E8EC)
private val ChatPanel = Color(0xFF19232E)

/** Android picker integration around the independently renderable conversation surface. */
@Composable
fun ChatRagScreen(navController: NavController, viewModel: ChatRagViewModel = hiltViewModel()) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    var showAttachments by remember { mutableStateOf(false) }
    var requestedKind by remember { mutableStateOf<ForensicRagEngine.RagMediaKind?>(null) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) {
            val fallbackKind = requestedKind
            scope.launch {
                val attachments = withContext(Dispatchers.IO) {
                    // A fifth item lets staging report the four-file limit instead of silently truncating.
                    uris.take(5).map { uri -> queryAttachment(context, uri, fallbackKind) }
                }
                viewModel.stageAttachments(attachments)
            }
        }
    }
    fun pick(kind: ForensicRagEngine.RagMediaKind?) {
        requestedKind = kind
        showAttachments = false
        picker.launch(when (kind) {
            ForensicRagEngine.RagMediaKind.IMAGE -> arrayOf("image/*")
            ForensicRagEngine.RagMediaKind.VIDEO -> arrayOf("video/*")
            ForensicRagEngine.RagMediaKind.AUDIO -> arrayOf("audio/*")
            else -> arrayOf("application/pdf", "text/*", "application/json", "image/*", "video/*", "audio/*")
        })
    }
    LiquidGlassScaffold(
        activeTab = GlassBottomTab.CHAT_RAG,
        onSelectTab = { tab -> if (tab != GlassBottomTab.CHAT_RAG) navController.navigate(tab.route) { launchSingleTop = true } },
    ) { scaffoldPadding ->
        val imeVisible = WindowInsets.ime.getBottom(LocalDensity.current) > 0
        ChatRagContent(
            uiState = uiState,
            onSend = viewModel::sendMessage,
            onStop = viewModel::cancelGeneration,
            onAttach = { showAttachments = true },
            onRemoveAttachment = viewModel::removeStagedAttachment,
            onNewConversation = viewModel::newConversation,
            onRetry = viewModel::retryLastMessage,
            onDismissError = viewModel::clearError,
            onSelectChunk = viewModel::selectTimelineChunk,
            onSelectTimestamp = viewModel::selectTimelineChunkForTimestamp,
            onOpenModels = { navController.navigate(Routes.MODEL_MANAGER) },
            onCopy = { text -> (context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager)
                ?.setPrimaryClip(ClipData.newPlainText("ONN response", text)) },
            modifier = Modifier.fillMaxSize().padding(bottom = if (imeVisible) 0.dp else scaffoldPadding.calculateBottomPadding()),
        )
    }
    if (showAttachments) AttachmentSheet(onDismiss = { showAttachments = false }, onPick = ::pick)
}

/** State-driven content used by the app, Compose tests and visual previews. */
@Composable
fun ChatRagContent(
    uiState: ChatRagUiState,
    onSend: (String) -> Unit,
    onStop: () -> Unit,
    onAttach: () -> Unit,
    onRemoveAttachment: (Uri) -> Unit,
    onNewConversation: () -> Unit,
    onRetry: () -> Unit,
    onDismissError: () -> Unit,
    onSelectChunk: (Int) -> Unit,
    onSelectTimestamp: (String) -> Unit,
    onOpenModels: () -> Unit,
    onCopy: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    var input by rememberSaveable { mutableStateOf("") }
    var contextExpanded by rememberSaveable { mutableStateOf(false) }
    val listState = rememberLazyListState()
    val scope = rememberCoroutineScope()
    val dragging by listState.interactionSource.collectIsDraggedAsState()
    val nearBottom by remember { derivedStateOf {
        !listState.canScrollForward || (listState.layoutInfo.visibleItemsInfo.lastOrNull()?.index ?: 0) >= listState.layoutInfo.totalItemsCount - 2
    } }
    var followLatest by remember { mutableStateOf(true) }
    val busy = uiState.isGenerating || uiState.isIndexing
    val staged = uiState.stagedAttachments.ifEmpty { listOfNotNull(uiState.stagedAttachment) }
    val lastMessage = uiState.messages.lastOrNull()
    LaunchedEffect(dragging, nearBottom) {
        if (dragging) followLatest = nearBottom else if (nearBottom) followLatest = true
    }
    LaunchedEffect(lastMessage?.id, lastMessage?.text?.length, busy) {
        if (followLatest && uiState.messages.isNotEmpty()) {
            withFrameNanos { }
            listState.scrollToItem((listState.layoutInfo.totalItemsCount - 1).coerceAtLeast(0))
        }
    }
    Box(modifier = modifier.statusBarsPadding().imePadding()) {
        Column(Modifier.widthIn(max = 840.dp).fillMaxSize().align(Alignment.TopCenter)) {
            ChatHeader(
                hasMessages = uiState.messages.isNotEmpty() || staged.isNotEmpty() || uiState.knowledgeBase != null,
                onNew = { input = ""; contextExpanded = false; followLatest = true; onNewConversation() },
                onOpenModels = onOpenModels,
            )
            uiState.knowledgeBase?.let { kb ->
                SourceContext(kb, uiState.selectedChunkIndex, contextExpanded, { contextExpanded = !contextExpanded }, onSelectChunk)
            }
            Box(Modifier.weight(1f).fillMaxWidth()) {
                LazyColumn(
                    state = listState,
                    modifier = Modifier.fillMaxSize().testTag("chat_messages"),
                    contentPadding = PaddingValues(start = 20.dp, end = 20.dp, top = 16.dp, bottom = 28.dp),
                    verticalArrangement = Arrangement.spacedBy(26.dp),
                ) {
                    if (uiState.messages.isEmpty()) item(key = "welcome") { WelcomeState(onAttach) { input = it } }
                    items(uiState.messages, key = { it.id }) { message ->
                        MessageRow(message, !busy && uiState.canRetry && message.id == lastMessage?.id,
                            onRetry, onCopy, { contextExpanded = true; onSelectTimestamp(it) })
                    }
                    if (busy) item(key = "working") {
                        WorkingIndicator(uiState.streamingStatus ?: if (uiState.isIndexing) "Reading your files" else "Thinking")
                    }
                }
                if (!nearBottom && uiState.messages.isNotEmpty()) {
                    IconButton(
                        onClick = { followLatest = true; scope.launch { listState.animateScrollToItem((listState.layoutInfo.totalItemsCount - 1).coerceAtLeast(0)) } },
                        modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 8.dp).size(44.dp).glassSurface(22.dp)
                            .semantics { contentDescription = "Jump to latest message" },
                    ) { ChatIcon(Symbol.DOWN, ChatInk) }
                }
            }
            uiState.errorMessage?.let { error ->
                Row(
                    Modifier.padding(horizontal = 16.dp, vertical = 4.dp).fillMaxWidth().glassSurface(18.dp)
                        .padding(start = 14.dp, top = 4.dp, bottom = 4.dp), verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(error, color = Color(0xFFF4B9B5), style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.weight(1f).semantics { liveRegion = LiveRegionMode.Polite }, maxLines = 4, overflow = TextOverflow.Ellipsis)
                    if (uiState.canRetry && !busy) TextButton(onClick = onRetry) { Text("Retry", color = ChatAccent) }
                    IconButton(onClick = onDismissError, modifier = Modifier.semantics { contentDescription = "Dismiss error" }) { ChatIcon(Symbol.CLOSE, ChatMuted, 16.dp) }
                }
            }
            ChatComposer(input, { input = it }, staged, busy, onAttach, onRemoveAttachment, onStop) {
                if (!busy && (input.isNotBlank() || staged.isNotEmpty())) {
                    val message = input; input = ""; followLatest = true; onSend(message)
                }
            }
        }
    }
}

@Composable
private fun ChatHeader(hasMessages: Boolean, onNew: () -> Unit, onOpenModels: () -> Unit) {
    Row(Modifier.fillMaxWidth().padding(start = 22.dp, end = 14.dp, top = 6.dp, bottom = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(38.dp).glassSurface(13.dp), contentAlignment = Alignment.Center) { ChatIcon(Symbol.SPARK, ChatAccent, 21.dp) }
        Spacer(Modifier.width(11.dp))
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text("ONN Chat", color = ChatInk, fontSize = 18.sp, fontWeight = FontWeight.SemiBold)
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                Box(Modifier.size(5.dp).background(Color(0xFF92D5BD), CircleShape))
                Text("On your device", color = ChatMuted, fontSize = 11.sp)
            }
        }
        IconButton(onClick = onNew, modifier = Modifier.testTag("chat_new").semantics { contentDescription = "New conversation" }, enabled = hasMessages) {
            ChatIcon(Symbol.NEW, if (hasMessages) ChatInk else ChatFaint)
        }
        IconButton(onClick = onOpenModels, modifier = Modifier.semantics { contentDescription = "Local model settings" }) { ChatIcon(Symbol.SETTINGS, ChatMuted) }
    }
}

@Composable
private fun WelcomeState(onAttach: () -> Unit, onPrompt: (String) -> Unit) {
    Column(Modifier.fillMaxWidth().padding(top = 46.dp, bottom = 20.dp), verticalArrangement = Arrangement.spacedBy(20.dp)) {
        Box(Modifier.size(64.dp).glassSurface(22.dp), contentAlignment = Alignment.Center) { ChatIcon(Symbol.SPARK, ChatAccent, 34.dp) }
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("A clearer look\nat your media.", color = ChatInk, fontSize = 32.sp, lineHeight = 38.sp, fontWeight = FontWeight.SemiBold)
            Text("Ask a question or add a photo, video,\naudio clip or document.", color = ChatMuted, fontSize = 15.sp, lineHeight = 23.sp)
        }
        Column(Modifier.fillMaxWidth().padding(top = 8.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            StarterCard(Symbol.IMAGE, "Take a closer look", "Add an image or video", onAttach)
            StarterCard(Symbol.FILE, "Read something together", "Add a PDF or text file", onAttach)
            StarterCard(Symbol.SPARK, "Understand a result", "What can an AI detector tell me?") {
                onPrompt("What can an AI detector tell me, and what are its limitations?")
            }
        }
    }
}

@Composable
private fun StarterCard(symbol: Symbol, title: String, subtitle: String, onClick: () -> Unit) {
    Row(Modifier.fillMaxWidth().glassSurface(18.dp).clickable(onClick = onClick).padding(horizontal = 16.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
        ChatIcon(symbol, ChatAccent, 21.dp)
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text(title, color = ChatInk, fontSize = 14.sp, fontWeight = FontWeight.Medium)
            Text(subtitle, color = ChatMuted, fontSize = 12.sp)
        }
        ChatIcon(Symbol.ARROW, ChatFaint, 17.dp)
    }
}

@Composable
private fun MessageRow(message: ChatMessage, canRetry: Boolean, onRetry: () -> Unit, onCopy: (String) -> Unit, onTimestamp: (String) -> Unit) {
    var copied by remember(message.id) { mutableStateOf(false) }
    LaunchedEffect(copied) { if (copied) { kotlinx.coroutines.delay(1600); copied = false } }
    val attachments = message.attachments.ifEmpty { listOfNotNull(message.attachment) }
    Column(Modifier.fillMaxWidth(), horizontalAlignment = if (message.isUser) Alignment.End else Alignment.Start,
        verticalArrangement = Arrangement.spacedBy(10.dp)) {
        if (!message.isUser) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ChatIcon(Symbol.SPARK, ChatAccent, 17.dp)
                Text("ONN", color = ChatInk, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                if (message.isStreaming) Text("· Writing", color = ChatFaint, fontSize = 11.sp)
                else if (message.wasStopped) Text("· Stopped", color = ChatFaint, fontSize = 11.sp)
                else if (message.isError) Text("· Couldn't finish", color = Color(0xFFF4B9B5), fontSize = 11.sp)
            }
        }
        if (attachments.isNotEmpty()) {
            Column(Modifier.widthIn(max = 350.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                attachments.forEach { AttachmentCard(it, onRemove = null) }
            }
        }
        if (message.text.isNotEmpty() || message.isStreaming) {
            val bubble = if (message.isUser) Modifier.fillMaxWidth(0.88f).glassSurface(23.dp, user = true).padding(horizontal = 17.dp, vertical = 13.dp)
                else Modifier.fillMaxWidth().padding(start = 1.dp)
            Box(bubble) { SelectionContainer { MessageText(message.text, message.isStreaming) } }
        }
        if (message.citedTimestamps.isNotEmpty()) {
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                message.citedTimestamps.distinct().forEach { time ->
                    Text("↗ $time", color = ChatAccent, fontSize = 11.sp,
                        modifier = Modifier.glassSurface(10.dp).clickable { onTimestamp(time) }.padding(horizontal = 10.dp, vertical = 8.dp)
                            .semantics { contentDescription = "Open source at $time" })
                }
            }
        }
        if (!message.isUser && !message.isStreaming && message.text.isNotBlank()) {
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = { onCopy(message.text); copied = true }, modifier = Modifier.size(40.dp)
                    .semantics { contentDescription = if (copied) "Response copied" else "Copy response" }) {
                    ChatIcon(if (copied) Symbol.CHECK else Symbol.COPY, if (copied) ChatAccent else ChatFaint, 17.dp)
                }
                if (copied) Text("Copied", color = ChatMuted, fontSize = 11.sp)
                if (canRetry) IconButton(onClick = onRetry, modifier = Modifier.size(40.dp).testTag("chat_retry").semantics { contentDescription = "Retry response" }) {
                    ChatIcon(Symbol.RETRY, ChatFaint, 18.dp)
                }
                message.engineBadge?.let { badge ->
                    val replyMode = when {
                        badge.contains("built-in", ignoreCase = true) -> "Built-in evidence reply"
                        badge == "Local evidence" -> "Saved scan evidence"
                        badge.contains("on device", ignoreCase = true) -> "Local model reply"
                        else -> badge.take(42)
                    }
                    Text(replyMode, color = ChatFaint, fontSize = 10.sp,
                        modifier = Modifier.padding(start = 4.dp).weight(1f), maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            }
        }
    }
}

@Composable
private fun MessageText(text: String, streaming: Boolean) {
    val formatted = remember(text) { chatMarkdown(text) }
    var cursorAlpha = 1f
    if (streaming) {
        val animation = rememberInfiniteTransition(label = "typing-cursor")
        val alpha by animation.animateFloat(0.15f, 1f, infiniteRepeatable(tween(520), RepeatMode.Reverse), label = "cursor-alpha")
        cursorAlpha = alpha
    }
    val display = buildAnnotatedString {
        append(formatted)
        if (streaming) withStyle(SpanStyle(color = ChatAccent.copy(alpha = cursorAlpha))) { append(" ▍") }
    }
    Text(display, color = ChatInk, fontSize = 15.sp, lineHeight = 25.sp)
}

/** Preserve unmatched delimiters while the response is arriving. */
internal fun chatMarkdown(text: String): AnnotatedString = buildAnnotatedString {
    val lines = text.split('\n')
    var codeBlock = false
    lines.forEachIndexed { lineIndex, source ->
        if (source.trimStart().startsWith("```")) codeBlock = !codeBlock
        else if (codeBlock) withStyle(SpanStyle(fontFamily = FontFamily.Monospace, color = ChatAccent)) { append(source) }
        else {
            val heading = source.trimStart().startsWith("#")
            val line = if (heading) source.trimStart().trimStart('#').trimStart() else if (source.startsWith("* ")) "• ${source.drop(2)}" else source
            withStyle(SpanStyle(fontWeight = if (heading) FontWeight.SemiBold else FontWeight.Normal)) {
                var position = 0
                while (position < line.length) {
                    val marker = when { line.startsWith("**", position) -> "**"; line[position] == '`' -> "`"; else -> null }
                    if (marker != null) {
                        val end = line.indexOf(marker, position + marker.length)
                        if (end >= 0) {
                            val style = if (marker == "**") SpanStyle(fontWeight = FontWeight.SemiBold) else SpanStyle(fontFamily = FontFamily.Monospace, color = ChatAccent)
                            withStyle(style) { append(line.substring(position + marker.length, end)) }
                            position = end + marker.length
                        } else { append(line[position]); position++ }
                    } else { append(line[position]); position++ }
                }
            }
        }
        if (lineIndex < lines.lastIndex) append('\n')
    }
}

@Composable
private fun WorkingIndicator(status: String) {
    Row(Modifier.padding(vertical = 4.dp).semantics { liveRegion = LiveRegionMode.Polite }, horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
        val transition = rememberInfiniteTransition(label = "working-circle")
        val angle by transition.animateFloat(0f, 360f, infiniteRepeatable(tween(1000, easing = LinearEasing)), label = "orbit-angle")
        Canvas(Modifier.size(20.dp).semantics { contentDescription = "Response in progress" }.testTag("chat_working")) {
            drawCircle(ChatAccent.copy(alpha = 0.15f), style = Stroke(2.dp.toPx()))
            rotate(angle) { drawArc(ChatAccent, -90f, 90f, false, style = Stroke(2.dp.toPx(), cap = StrokeCap.Round)) }
        }
        Text(status.take(110), color = ChatMuted, fontSize = 12.sp, maxLines = 2, overflow = TextOverflow.Ellipsis)
    }
}

@Composable
private fun ChatComposer(input: String, onInput: (String) -> Unit, attachments: List<AttachedMedia>, busy: Boolean,
    onAttach: () -> Unit, onRemoveAttachment: (Uri) -> Unit, onStop: () -> Unit, onSend: () -> Unit) {
    val canSend = !busy && (input.isNotBlank() || attachments.isNotEmpty())
    Column(Modifier.padding(start = 14.dp, end = 14.dp, top = 6.dp, bottom = 8.dp)) {
        Column(Modifier.fillMaxWidth().glassSurface(28.dp).animateContentSize().padding(horizontal = 7.dp, vertical = 7.dp)) {
            AnimatedVisibility(attachments.isNotEmpty()) {
                Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(5.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    attachments.forEach { attachment -> Box(Modifier.width(230.dp)) { AttachmentCard(attachment) { onRemoveAttachment(attachment.uri) } } }
                }
            }
            Row(verticalAlignment = Alignment.Bottom, modifier = Modifier.fillMaxWidth()) {
                IconButton(onClick = onAttach, enabled = !busy && attachments.size < 4,
                    modifier = Modifier.size(44.dp).testTag("chat_attach").semantics { contentDescription = "Add photos, videos or files" }) {
                    ChatIcon(Symbol.PLUS, if (!busy && attachments.size < 4) ChatMuted else ChatFaint)
                }
                BasicTextField(
                    value = input, onValueChange = { onInput(it.take(ChatRagViewModel.MAX_MESSAGE_LENGTH)) },
                    modifier = Modifier.weight(1f).heightIn(min = 44.dp, max = 160.dp).padding(top = 11.dp, bottom = 9.dp, end = 7.dp)
                        .testTag("chat_input").semantics { contentDescription = "Message" },
                    textStyle = MaterialTheme.typography.bodyLarge.copy(color = ChatInk, fontSize = 15.sp, lineHeight = 22.sp),
                    cursorBrush = SolidColor(ChatAccent), maxLines = 6,
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send), keyboardActions = KeyboardActions(onSend = { if (canSend) onSend() }),
                    decorationBox = { inner -> Box { if (input.isEmpty()) Text("Message ONN…", color = ChatFaint, fontSize = 15.sp); inner() } },
                )
                IconButton(onClick = if (busy) onStop else onSend, enabled = busy || canSend,
                    modifier = Modifier.size(44.dp).padding(3.dp).background(if (busy || canSend) ChatAccent else Color.White.copy(alpha = 0.08f), CircleShape)
                        .testTag(if (busy) "chat_stop" else "chat_send").semantics { contentDescription = if (busy) "Stop response" else "Send message" }) {
                    ChatIcon(if (busy) Symbol.STOP else Symbol.UP, if (busy || canSend) Color(0xFF14232A) else ChatFaint, 20.dp)
                }
            }
        }
        Text(
            if (input.length >= ChatRagViewModel.MAX_MESSAGE_LENGTH - 200) "${input.length} / ${ChatRagViewModel.MAX_MESSAGE_LENGTH} characters"
            else "Images, video, audio, PDF and text · Up to 4 files",
            modifier = Modifier.align(Alignment.CenterHorizontally).padding(top = 8.dp), color = ChatFaint, fontSize = 10.sp,
        )
    }
}

@Composable
private fun AttachmentCard(attachment: AttachedMedia, onRemove: (() -> Unit)?) {
    Row(Modifier.fillMaxWidth().glassSurface(16.dp).padding(8.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        AttachmentThumbnail(attachment)
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(attachment.fileName, color = ChatInk, fontSize = 12.sp, fontWeight = FontWeight.Medium, maxLines = 1, overflow = TextOverflow.Ellipsis)
            val kind = when (attachment.kind) { ForensicRagEngine.RagMediaKind.IMAGE -> "Image"; ForensicRagEngine.RagMediaKind.VIDEO -> "Video"; ForensicRagEngine.RagMediaKind.AUDIO -> "Audio"; else -> "Document" }
            Text(listOfNotNull(kind, attachment.fileSizeFormatted).joinToString(" · "), color = ChatMuted, fontSize = 10.sp)
        }
        if (onRemove != null) IconButton(onClick = onRemove, modifier = Modifier.size(40.dp).semantics { contentDescription = "Remove ${attachment.fileName}" }) { ChatIcon(Symbol.CLOSE, ChatMuted, 15.dp) }
    }
}

@Composable
private fun AttachmentThumbnail(attachment: AttachedMedia) {
    val context = LocalContext.current
    val thumbnail by produceState<Bitmap?>(null, attachment.uri, attachment.kind) {
        if (attachment.kind == ForensicRagEngine.RagMediaKind.IMAGE) value = loadThumbnail(context, attachment.uri)
    }
    val image = thumbnail
    DisposableEffect(image) { onDispose { image?.recycle() } }
    Box(Modifier.size(44.dp).clip(RoundedCornerShape(10.dp)).background(Color.White.copy(alpha = 0.045f)), contentAlignment = Alignment.Center) {
        if (image != null && !image.isRecycled) Image(image.asImageBitmap(), "Preview of ${attachment.fileName}", contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
        else ChatIcon(when (attachment.kind) {
            ForensicRagEngine.RagMediaKind.IMAGE -> Symbol.IMAGE; ForensicRagEngine.RagMediaKind.VIDEO -> Symbol.VIDEO
            ForensicRagEngine.RagMediaKind.AUDIO -> Symbol.AUDIO; else -> Symbol.FILE
        }, ChatAccent, 22.dp)
    }
}

@Composable
private fun SourceContext(kb: ForensicRagEngine.IndexedMediaKnowledgeBase, selectedChunkIndex: Int?, expanded: Boolean, onToggle: () -> Unit, onSelectChunk: (Int) -> Unit) {
    val chunk = kb.chunks.firstOrNull { it.index == selectedChunkIndex } ?: kb.chunks.firstOrNull()
    Column(Modifier.padding(horizontal = 16.dp, vertical = 2.dp).fillMaxWidth().glassSurface(18.dp).animateContentSize()) {
        Row(Modifier.fillMaxWidth().clickable(onClick = onToggle).padding(horizontal = 13.dp, vertical = 11.dp).testTag("chat_context")
            .semantics { contentDescription = if (expanded) "Collapse source details" else "Expand source details" }, verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            ChatIcon(Symbol.FILE, ChatAccent, 17.dp)
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(kb.mediaTitle, color = ChatInk, fontSize = 12.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text("Source details · ${kb.chunks.size} ${if (kb.chunks.size == 1) "section" else "sections"}", color = ChatMuted, fontSize = 10.sp)
            }
            ChatIcon(if (expanded) Symbol.UP else Symbol.DOWN, ChatMuted, 16.dp)
        }
        AnimatedVisibility(expanded) {
            Column(Modifier.padding(start = 13.dp, end = 13.dp, bottom = 13.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                if (kb.mediaKind == ForensicRagEngine.RagMediaKind.VIDEO && kb.sourceUri != null) SourceVideo(kb.sourceUri, chunk?.startTimeMs ?: 0L)
                if (kb.chunks.size > 1) Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    kb.chunks.forEach { entry -> Text(entry.formattedTimeRange, color = if (entry.index == chunk?.index) ChatInk else ChatMuted, fontSize = 11.sp,
                        modifier = Modifier.clip(RoundedCornerShape(9.dp)).background(if (entry.index == chunk?.index) ChatAccent.copy(alpha = 0.13f) else Color.Transparent)
                            .clickable { onSelectChunk(entry.index) }.padding(horizontal = 10.dp, vertical = 9.dp)) }
                }
                Text(kb.headlineVerdict, color = ChatAccent, fontSize = 12.sp)
                chunk?.let { Text(it.visualDescription, color = ChatMuted, fontSize = 12.sp, lineHeight = 19.sp, maxLines = 5, overflow = TextOverflow.Ellipsis) }
            }
        }
    }
}

@Composable
private fun SourceVideo(uri: String, seekMs: Long) {
    val requestedPosition = seekMs.coerceIn(0, Int.MAX_VALUE.toLong()).toInt()
    AndroidView(
        modifier = Modifier.fillMaxWidth().heightIn(max = 170.dp).aspectRatio(16f / 9f).clip(RoundedCornerShape(12.dp)),
        factory = { context -> VideoView(context).apply {
            val controls = MediaController(context); controls.setAnchorView(this); setMediaController(controls)
            setOnPreparedListener { seekTo(requestedPosition); pause() }; setOnErrorListener { _, _, _ -> true }
            setVideoURI(Uri.parse(uri)); tag = Pair(uri, requestedPosition)
        } },
        update = { view ->
            val previous = view.tag as? Pair<*, *>
            if (previous?.first != uri) { view.setVideoURI(Uri.parse(uri)); view.tag = Pair(uri, requestedPosition) }
            else if (previous?.second != requestedPosition) { view.seekTo(requestedPosition); view.pause(); view.tag = Pair(uri, requestedPosition) }
        },
        onRelease = { it.stopPlayback() },
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AttachmentSheet(onDismiss: () -> Unit, onPick: (ForensicRagEngine.RagMediaKind?) -> Unit) {
    ModalBottomSheet(onDismissRequest = onDismiss, containerColor = Color(0xFF17212D), contentColor = ChatInk) {
        Column(Modifier.fillMaxWidth().padding(start = 22.dp, end = 22.dp, bottom = 30.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text("Add to your message", color = ChatInk, fontSize = 21.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.padding(bottom = 8.dp))
            Text("Choose up to 4 files. Your attachments stay on your device.", color = ChatMuted, fontSize = 13.sp, lineHeight = 20.sp, modifier = Modifier.padding(bottom = 12.dp))
            AttachmentOption(Symbol.IMAGE, "Photos & images", "Choose from your files") { onPick(ForensicRagEngine.RagMediaKind.IMAGE) }
            AttachmentOption(Symbol.VIDEO, "Video", "Clips and saved videos") { onPick(ForensicRagEngine.RagMediaKind.VIDEO) }
            AttachmentOption(Symbol.AUDIO, "Audio", "Voice notes and recordings") { onPick(ForensicRagEngine.RagMediaKind.AUDIO) }
            AttachmentOption(Symbol.FILE, "Files & documents", "PDF, TXT, Markdown, CSV and JSON") { onPick(null) }
        }
    }
}

@Composable
private fun AttachmentOption(symbol: Symbol, title: String, description: String, onClick: () -> Unit) {
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).clickable(onClick = onClick).padding(vertical = 13.dp, horizontal = 8.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        Box(Modifier.size(44.dp).glassSurface(14.dp), contentAlignment = Alignment.Center) { ChatIcon(symbol, ChatAccent, 22.dp) }
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(title, color = ChatInk, fontSize = 15.sp, fontWeight = FontWeight.Medium)
            Text(description, color = ChatMuted, fontSize = 12.sp)
        }
    }
}

private fun Modifier.glassSurface(radius: Dp, user: Boolean = false): Modifier {
    val shape = RoundedCornerShape(radius)
    return clip(shape).background(Brush.verticalGradient(listOf(
        if (user) Color(0xE52C3D4B) else ChatPanel.copy(alpha = 0.78f), if (user) Color(0xCC24343F) else Color(0xB3121A24),
    ))).border(1.dp, Brush.verticalGradient(listOf(Color.White.copy(alpha = if (user) 0.19f else 0.14f), ChatAccent.copy(alpha = 0.035f))), shape)
}

private enum class Symbol { SPARK, PLUS, UP, DOWN, CLOSE, NEW, SETTINGS, IMAGE, VIDEO, AUDIO, FILE, ARROW, STOP, COPY, CHECK, RETRY }

/** Consistent stroke icons without an image loader or additional dependency. */
@Composable
private fun ChatIcon(symbol: Symbol, color: Color, size: Dp = 22.dp) {
    Canvas(Modifier.size(size)) {
        val iconScale = this.size.minDimension / 24f
        val stroke = Stroke(width = 1.65f, cap = StrokeCap.Round)
        withTransform({ scale(iconScale, iconScale, Offset.Zero) }) {
            fun line(x1: Float, y1: Float, x2: Float, y2: Float) = drawLine(color, Offset(x1, y1), Offset(x2, y2), 1.65f, StrokeCap.Round)
            fun path(vararg points: Pair<Float, Float>) {
                val p = Path().apply { points.forEachIndexed { i, point -> if (i == 0) moveTo(point.first, point.second) else lineTo(point.first, point.second) } }
                drawPath(p, color, style = stroke)
            }
            when (symbol) {
                Symbol.SPARK -> {
                    val p = Path().apply { moveTo(12f, 2.5f); cubicTo(13.4f, 9f, 15f, 10.5f, 21.5f, 12f); cubicTo(15f, 13.4f, 13.4f, 15f, 12f, 21.5f); cubicTo(10.6f, 15f, 9f, 13.4f, 2.5f, 12f); cubicTo(9f, 10.6f, 10.6f, 9f, 12f, 2.5f); close() }
                    drawPath(p, color, style = stroke)
                }
                Symbol.PLUS -> { line(12f, 5f, 12f, 19f); line(5f, 12f, 19f, 12f) }
                Symbol.UP -> { path(6f to 11f, 12f to 5f, 18f to 11f); line(12f, 5f, 12f, 20f) }
                Symbol.DOWN -> { path(6f to 11f, 12f to 17f, 18f to 11f); line(12f, 4f, 12f, 17f) }
                Symbol.CLOSE -> { line(6f, 6f, 18f, 18f); line(18f, 6f, 6f, 18f) }
                Symbol.NEW -> { path(10f to 5f, 5f to 5f, 5f to 20f, 20f to 20f, 20f to 14f); path(10f to 14f, 11f to 9f, 18f to 2f, 22f to 6f, 15f to 13f, 10f to 14f) }
                Symbol.SETTINGS -> { line(4f, 7f, 20f, 7f); line(4f, 17f, 20f, 17f); drawCircle(color, 2.5f, Offset(9f, 7f), style = stroke); drawCircle(color, 2.5f, Offset(16f, 17f), style = stroke) }
                Symbol.IMAGE -> { path(4f to 4f, 20f to 4f, 20f to 20f, 4f to 20f, 4f to 4f); drawCircle(color, 1.6f, Offset(9f, 9f), style = stroke); path(4f to 17f, 10f to 12f, 14f to 15f, 17f to 12f, 20f to 15f) }
                Symbol.VIDEO -> { path(3f to 6f, 15f to 6f, 15f to 18f, 3f to 18f, 3f to 6f); path(15f to 10f, 21f to 7f, 21f to 17f, 15f to 14f) }
                Symbol.AUDIO -> { listOf(5f to 6f, 9f to 13f, 13f to 19f, 17f to 11f, 21f to 5f).forEach { (x, h) -> line(x, 12f - h / 2f, x, 12f + h / 2f) } }
                Symbol.FILE -> { path(6f to 3f, 15f to 3f, 20f to 8f, 20f to 21f, 6f to 21f, 6f to 3f); path(15f to 3f, 15f to 8f, 20f to 8f); line(10f, 13f, 16f, 13f); line(10f, 17f, 15f, 17f) }
                Symbol.ARROW -> { line(5f, 12f, 19f, 12f); path(14f to 7f, 19f to 12f, 14f to 17f) }
                Symbol.STOP -> drawRoundRect(color, Offset(6f, 6f), androidx.compose.ui.geometry.Size(12f, 12f), androidx.compose.ui.geometry.CornerRadius(2f))
                Symbol.COPY -> { path(8f to 8f, 20f to 8f, 20f to 21f, 8f to 21f, 8f to 8f); path(4f to 16f, 4f to 3f, 15f to 3f) }
                Symbol.CHECK -> path(5f to 12f, 10f to 17f, 20f to 7f)
                Symbol.RETRY -> { drawArc(color, 30f, 290f, false, Offset(4f, 4f), androidx.compose.ui.geometry.Size(16f, 16f), style = stroke); path(17f to 3f, 20f to 7f, 15f to 8f) }
            }
        }
    }
}

private suspend fun loadThumbnail(context: Context, uri: Uri): Bitmap? = try {
    withTimeoutOrNull(3_000) {
        cancellableProviderWork(dispose = { bitmap: Bitmap? -> bitmap?.recycle() }) {
            fun input() = own(context.contentResolver.openAssetFileDescriptor(uri, "r", signal))?.let {
                own(it.createInputStream())
            } ?: own(context.contentResolver.openInputStream(uri))
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            input()?.use { BitmapFactory.decodeStream(it, null, bounds) }
            checkActive()
            if (bounds.outWidth <= 0 || bounds.outHeight <= 0 || bounds.outWidth.toLong() * bounds.outHeight > 100_000_000L) {
                return@cancellableProviderWork null
            }
            val options = BitmapFactory.Options().apply {
                inSampleSize = 1
                while (bounds.outWidth / inSampleSize > 160 || bounds.outHeight / inSampleSize > 160) inSampleSize *= 2
                inPreferredConfig = Bitmap.Config.RGB_565
            }
            input()?.use { BitmapFactory.decodeStream(it, null, options) }
        }
    }
} catch (cancelled: CancellationException) {
    throw cancelled
} catch (_: Exception) {
    null
} catch (_: OutOfMemoryError) {
    null
}

private suspend fun queryAttachment(context: Context, uri: Uri, fallbackKind: ForensicRagEngine.RagMediaKind?): AttachedMedia {
    runCatching { context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) }
    val metadata = try {
        withTimeoutOrNull(3_000) {
            cancellableProviderWork {
                var displayName = "Attachment"
                var byteCount: Long? = null
                own(context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE), null, null, null, signal))?.use { cursor ->
                    if (cursor.moveToFirst()) {
                        val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                        if (nameIndex >= 0) cursor.getString(nameIndex)?.takeIf(String::isNotBlank)?.let { displayName = it }
                        val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
                        if (sizeIndex >= 0 && !cursor.isNull(sizeIndex)) cursor.getLong(sizeIndex).takeIf { it >= 0 }?.let { byteCount = it }
                    }
                }
                checkActive()
                val type = context.contentResolver.getType(uri)
                checkActive()
                Triple(displayName, byteCount, type)
            }
        }
    } catch (cancelled: CancellationException) {
        throw cancelled
    } catch (_: Exception) {
        null
    }
    val name = metadata?.first ?: "Attachment"
    val bytes = metadata?.second
    val mime = metadata?.third
    val ext = name.substringAfterLast('.', "").lowercase(Locale.ROOT)
    val kind = when {
        mime?.startsWith("image/") == true || ext in setOf("jpg", "jpeg", "png", "webp", "bmp", "heic", "heif", "avif", "gif") -> ForensicRagEngine.RagMediaKind.IMAGE
        mime?.startsWith("video/") == true || ext in setOf("mp4", "webm", "mov", "mkv", "3gp") -> ForensicRagEngine.RagMediaKind.VIDEO
        mime?.startsWith("audio/") == true || ext in setOf("wav", "mp3", "m4a", "ogg", "opus", "flac", "aac") -> ForensicRagEngine.RagMediaKind.AUDIO
        else -> fallbackKind ?: ForensicRagEngine.RagMediaKind.DOCUMENT
    }
    val size = bytes?.let { when {
        it >= 1024 * 1024 -> String.format(Locale.US, "%.1f MB", it / (1024.0 * 1024))
        it >= 1024 -> String.format(Locale.US, "%.0f KB", it / 1024.0)
        else -> "$it B"
    } }
    return AttachedMedia(uri, kind, name, mime, size)
}
