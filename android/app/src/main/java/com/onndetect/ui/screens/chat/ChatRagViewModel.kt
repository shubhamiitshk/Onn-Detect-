package com.onndetect.ui.screens.chat

import android.net.Uri
import android.content.Context
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.data.repository.ScanHistoryRepository
import com.onndetect.chat.ChatModelStore
import com.onndetect.rag.ForensicRagEngine
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.text.DateFormat
import java.util.Date
import java.util.UUID
import javax.inject.Inject

data class AttachedMedia(
    val uri: Uri,
    val kind: ForensicRagEngine.RagMediaKind,
    val fileName: String,
    val mimeType: String? = null,
    val fileSizeFormatted: String? = null,
    /** Saved scan identity; report attachments never enter the provider importer. */
    val scanId: String? = null,
)

enum class ChatIntent { NORMAL, MEDIA, REPORT }

data class ScanReportChoice(val id: String, val title: String, val assessment: String, val dateLabel: String)

data class ChatMessage(
    val id: String,
    val isUser: Boolean,
    val text: String,
    val attachment: AttachedMedia? = null,
    val citedTimestamps: List<String> = emptyList(),
    val engineBadge: String? = null,
    val isStreaming: Boolean = false,
    val attachments: List<AttachedMedia> = emptyList(),
    val isError: Boolean = false,
    val wasStopped: Boolean = false,
)

data class ChatRagUiState(
    val isIndexing: Boolean = false,
    val indexingProgressText: String = "",
    val isGenerating: Boolean = false,
    val streamingStatus: String? = null,
    val knowledgeBase: ForensicRagEngine.IndexedMediaKnowledgeBase? = null,
    val messages: List<ChatMessage> = emptyList(),
    val stagedAttachment: AttachedMedia? = null,
    val selectedChunkIndex: Int? = null,
    val errorMessage: String? = null,
    val stagedAttachments: List<AttachedMedia> = emptyList(),
    val canRetry: Boolean = false,
    val intent: ChatIntent = ChatIntent.NORMAL,
    val recentReports: List<ScanReportChoice> = emptyList(),
    val reportsLoading: Boolean = true,
    val modelReady: Boolean = false,
    val modelName: String = "No chat model selected",
    val modelStatusMessage: String = "Set up a local model for generated conversation.",
    val modelSupportsVision: Boolean = false,
)

@HiltViewModel
class ChatRagViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val ragEngine: ForensicRagEngine,
    private val historyRepository: ScanHistoryRepository,
    @param:ApplicationContext private val appContext: Context,
) : ViewModel() {
    private val _uiState = MutableStateFlow(ChatRagUiState())
    val uiState: StateFlow<ChatRagUiState> = _uiState.asStateFlow()

    private var generationJob: Job? = null
    private var historyJob: Job? = null
    // A cancelled native/provider operation may finish later. It must never update a newer chat.
    @Volatile private var requestVersion = 0L
    private var failedRequest: Request? = null
    private val sourceKnowledge = linkedMapOf<String, ForensicRagEngine.IndexedMediaKnowledgeBase>()

    private data class Request(
        val assistantId: String,
        val query: String,
        val attachments: List<AttachedMedia>,
        val priorKnowledge: ForensicRagEngine.IndexedMediaKnowledgeBase?,
        val priorSources: List<Pair<String, ForensicRagEngine.IndexedMediaKnowledgeBase>>,
        val conversationContext: String,
    )

    init {
        viewModelScope.launch {
            ChatModelStore.status.collect { status ->
                _uiState.update { it.copy(modelReady = status.isReady, modelName = status.modelName,
                    modelStatusMessage = status.message, modelSupportsVision = status.supportsVision) }
            }
        }
        refreshChatModel()
        viewModelScope.launch {
            try {
                historyRepository.observeAll().collect { records ->
                    val formatter = DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT)
                    val choices = records.take(100).map { record ->
                        val kind = when {
                            record.scanMode.contains("VIDEO") -> "Video"
                            record.scanMode.contains("AUDIO") -> "Audio"
                            record.scanMode.contains("IMAGE") -> "Image"
                            else -> "Media"
                        }
                        val assessment = if (record.status == "SUCCESS" && record.modelValidated &&
                            record.verdict in setOf("AUTHENTIC", "SYNTHETIC")) record.verdict else "INCONCLUSIVE"
                        ScanReportChoice(record.id, "$kind scan · ${record.id.take(8)}", assessment,
                            formatter.format(Date(record.timestamp)))
                    }
                    _uiState.update { it.copy(recentReports = choices, reportsLoading = false) }
                }
            } catch (cancelled: CancellationException) { throw cancelled }
            catch (_: Exception) { _uiState.update { it.copy(reportsLoading = false) } }
        }
        // A new chat is empty. Previously scanned private media is loaded only by an explicit route.
        savedStateHandle.get<String>("scanId")?.takeIf(String::isNotBlank)?.let(::loadFromScanHistory)
    }

    fun refreshChatModel() {
        viewModelScope.launch(Dispatchers.IO) { ChatModelStore.refresh(appContext) }
    }

    fun setChatIntent(intent: ChatIntent) {
        if (_uiState.value.isGenerating || _uiState.value.isIndexing) return
        if (intent == ChatIntent.NORMAL) {
            sourceKnowledge.clear()
            _uiState.update { it.copy(intent = intent, knowledgeBase = null, selectedChunkIndex = null,
                stagedAttachments = emptyList(), stagedAttachment = null) }
        } else _uiState.update { it.copy(intent = intent) }
    }

    fun stageScanReport(scanId: String) {
        val choice = _uiState.value.recentReports.firstOrNull { it.id == scanId }
        if (choice == null) {
            _uiState.update { it.copy(errorMessage = "This scan report is no longer available.") }
            return
        }
        stageAttachments(listOf(AttachedMedia(
            uri = Uri.parse("onn-report://scan/${Uri.encode(scanId)}"),
            kind = ForensicRagEngine.RagMediaKind.SCAN_HISTORY_RECORD,
            fileName = choice.title, fileSizeFormatted = choice.assessment, scanId = scanId,
        )))
    }

    fun loadFromScanHistory(scanId: String) {
        resetWork()
        sourceKnowledge.clear()
        val version = requestVersion
        _uiState.update { it.copy(isIndexing = true, indexingProgressText = "Opening scan…", errorMessage = null, intent = ChatIntent.REPORT) }
        historyJob = viewModelScope.launch {
            try {
                val entity = withContext(Dispatchers.IO) { historyRepository.findById(scanId) }
                    ?: error("This scan is no longer available.")
                val kb = withContext(Dispatchers.IO) { ragEngine.indexFromScanHistory(entity) }
                if (version != requestVersion) return@launch
                sourceKnowledge["scan:$scanId"] = kb
                updateFor(version) {
                    it.copy(
                        isIndexing = false,
                        indexingProgressText = "",
                        knowledgeBase = kb,
                        selectedChunkIndex = kb.chunks.firstOrNull()?.index,
                        messages = listOf(ChatMessage(
                            id = UUID.randomUUID().toString(),
                            isUser = false,
                            text = "I've opened **${kb.mediaTitle}**. Its recorded assessment is **${kb.headlineVerdict}**. Ask about the findings, or attach another file.",
                            engineBadge = kb.activeEngineLabel,
                        )),
                    )
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                updateFor(version) { it.copy(isIndexing = false, indexingProgressText = "", errorMessage = error.message ?: "Couldn't open this scan.") }
            }
        }
    }

    fun stageAttachment(uri: Uri, kind: ForensicRagEngine.RagMediaKind, fileName: String, mimeType: String? = null, fileSizeFormatted: String? = null) =
        stageAttachments(listOf(AttachedMedia(uri, kind, fileName, mimeType, fileSizeFormatted)))

    fun stageAttachments(attachments: List<AttachedMedia>) {
        if (attachments.isEmpty()) return
        val supported = attachments.filter {
            if (it.kind == ForensicRagEngine.RagMediaKind.SCAN_HISTORY_RECORD) !it.scanId.isNullOrBlank() && it.scanId.length <= 200
            else it.uri.scheme == "content" && it.scanId == null
        }
        _uiState.update { state ->
            val combined = (state.stagedAttachments + supported).distinctBy { it.uri.toString() }
            val kept = combined.take(MAX_ATTACHMENTS)
            state.copy(
                stagedAttachments = kept,
                stagedAttachment = kept.firstOrNull(),
                intent = when {
                    kept.isEmpty() -> state.intent
                    kept.all { it.kind == ForensicRagEngine.RagMediaKind.SCAN_HISTORY_RECORD } -> ChatIntent.REPORT
                    else -> ChatIntent.MEDIA
                },
                errorMessage = when {
                    supported.size != attachments.size -> "Choose images, videos, audio, PDFs or text files from your device."
                    combined.size > MAX_ATTACHMENTS -> "You can attach up to $MAX_ATTACHMENTS files per message."
                    else -> null
                },
            )
        }
    }

    fun removeStagedAttachment(uri: Uri) {
        _uiState.update { state ->
            val kept = state.stagedAttachments.filterNot { it.uri == uri }
            state.copy(stagedAttachments = kept, stagedAttachment = kept.firstOrNull())
        }
    }

    fun clearStagedAttachment() {
        _uiState.update { it.copy(stagedAttachment = null, stagedAttachments = emptyList()) }
    }

    fun newConversation() {
        resetWork()
        sourceKnowledge.clear()
        val current = _uiState.value
        _uiState.value = ChatRagUiState(recentReports = current.recentReports, reportsLoading = current.reportsLoading,
            modelReady = current.modelReady, modelName = current.modelName, modelStatusMessage = current.modelStatusMessage,
            modelSupportsVision = current.modelSupportsVision)
    }

    fun clearConversation() = newConversation()

    private fun resetWork() {
        requestVersion++
        generationJob?.cancel()
        historyJob?.cancel()
        generationJob = null
        historyJob = null
        failedRequest = null
    }

    fun cancelGeneration() {
        resetWork()
        _uiState.update { state ->
            state.copy(
                isGenerating = false,
                isIndexing = false,
                indexingProgressText = "",
                streamingStatus = null,
                canRetry = false,
                messages = state.messages.map { message ->
                    if (message.isStreaming) message.copy(
                        text = message.text.ifBlank { "Response stopped." },
                        isStreaming = false,
                        wasStopped = true,
                    ) else message
                },
            )
        }
    }

    fun indexMedia(uri: Uri, mediaKind: ForensicRagEngine.RagMediaKind) {
        stageAttachment(uri, mediaKind, "Attached file")
        sendMessage("")
    }

    fun askQuestion(question: String) = sendMessage(question)

    fun sendMessage(text: String) {
        val state = _uiState.value
        if (state.isGenerating || state.isIndexing) return
        val staged = state.stagedAttachments.ifEmpty { listOfNotNull(state.stagedAttachment) }
        val trimmed = text.trim()
        if (trimmed.isEmpty() && staged.isEmpty()) return
        if (trimmed.length > MAX_MESSAGE_LENGTH) {
            _uiState.update { it.copy(errorMessage = "Please keep your message under $MAX_MESSAGE_LENGTH characters.") }
            return
        }
        val query = trimmed.ifBlank {
            if (staged.size > 1) "Review these files and explain the available evidence and limitations."
            else when (staged.firstOrNull()?.kind) {
                ForensicRagEngine.RagMediaKind.VIDEO -> "Review this video and explain what the sampled frames and audio can tell us."
                ForensicRagEngine.RagMediaKind.AUDIO -> "Review this audio for available speech-detector evidence and explain its limitations."
                ForensicRagEngine.RagMediaKind.DOCUMENT -> "Summarize this document and explain any available forensic observations."
                ForensicRagEngine.RagMediaKind.SCAN_HISTORY_RECORD -> "Explain this saved forensic report, its measured evidence, coverage, and limitations."
                else -> "Review this image for AI-generation evidence and explain any uncertainty."
            }
        }
        val request = Request(
            assistantId = UUID.randomUUID().toString(),
            query = query,
            attachments = staged,
            priorKnowledge = if (state.intent == ChatIntent.NORMAL) null else state.knowledgeBase,
            priorSources = if (state.intent == ChatIntent.NORMAL) emptyList() else sourceKnowledge.entries.map { it.key to it.value },
            conversationContext = state.messages.filter { !it.isError && !it.isStreaming }.takeLast(8)
                .joinToString("\n") { "${if (it.isUser) "User" else "Assistant"}: ${it.text.take(1600)}" }.takeLast(8000),
        )
        _uiState.update {
            it.copy(
                messages = it.messages + ChatMessage(
                    id = UUID.randomUUID().toString(), isUser = true, text = query,
                    attachment = staged.firstOrNull(), attachments = staged,
                ),
                stagedAttachment = null,
                stagedAttachments = emptyList(),
            )
        }
        runRequest(request)
    }

    fun retryLastMessage() {
        if (_uiState.value.isGenerating || _uiState.value.isIndexing) return
        val request = failedRequest ?: return
        _uiState.update { state -> state.copy(messages = state.messages.filterNot { it.id == request.assistantId }) }
        runRequest(request)
    }

    private fun runRequest(request: Request) {
        val version = ++requestVersion
        failedRequest = null
        _uiState.update { state ->
            state.copy(
                isGenerating = true,
                isIndexing = request.attachments.isNotEmpty(),
                indexingProgressText = if (request.attachments.isEmpty()) "" else "Opening attachments…",
                streamingStatus = if (request.attachments.isEmpty()) "Thinking…" else "Opening attachments…",
                errorMessage = null,
                canRetry = false,
                messages = state.messages + ChatMessage(id = request.assistantId, isUser = false, text = "", isStreaming = true),
            )
        }
        generationJob = viewModelScope.launch {
            try {
                withTimeout(REQUEST_TIMEOUT_MS) {
                    var knowledge = request.priorKnowledge
                    if (request.attachments.isNotEmpty()) {
                        val indexed = request.attachments.mapIndexed { index, attachment ->
                            val onProgress: (String) -> Unit = { detail ->
                                val label = if (request.attachments.size > 1) "${index + 1}/${request.attachments.size} · $detail" else detail
                                updateFor(version) { it.copy(indexingProgressText = label, streamingStatus = label) }
                            }
                            withContext(Dispatchers.IO) {
                                val base = when (attachment.kind) {
                                    ForensicRagEngine.RagMediaKind.SCAN_HISTORY_RECORD -> {
                                        onProgress("Opening saved scan report…")
                                        val record = historyRepository.findById(attachment.scanId ?: error("Missing report identity"))
                                            ?: error("This scan report is no longer available.")
                                        ragEngine.indexFromScanHistory(record)
                                    }
                                    ForensicRagEngine.RagMediaKind.VIDEO -> ragEngine.indexVideoUri(attachment.uri, onProgress = onProgress)
                                    ForensicRagEngine.RagMediaKind.AUDIO -> ragEngine.indexAudioUri(attachment.uri, onProgress)
                                    ForensicRagEngine.RagMediaKind.DOCUMENT -> ragEngine.indexDocumentUri(
                                        attachment.uri, attachment.fileName, attachment.mimeType, onProgress,
                                    )
                                    else -> ragEngine.indexImageUri(attachment.uri, onProgress)
                                }
                                base.copy(mediaTitle = attachment.fileName)
                            }
                        }
                        // Keep recent source evidence for follow-up comparisons, bounded to four files.
                        val sources = linkedMapOf<String, ForensicRagEngine.IndexedMediaKnowledgeBase>()
                        sources.putAll(request.priorSources)
                        request.attachments.zip(indexed).forEach { (attachment, kb) ->
                            sources.remove(attachment.uri.toString())
                            sources[attachment.uri.toString()] = kb
                        }
                        val retained = sources.entries.toList().takeLast(MAX_ATTACHMENTS)
                        val combinedKnowledge = ForensicRagEngine.combineKnowledgeBases(retained.map { it.value })
                        knowledge = combinedKnowledge
                        if (version == requestVersion) {
                            sourceKnowledge.clear()
                            retained.forEach { sourceKnowledge[it.key] = it.value }
                        }
                        updateFor(version) { it.copy(
                            isIndexing = false, indexingProgressText = "", knowledgeBase = combinedKnowledge,
                            selectedChunkIndex = combinedKnowledge.chunks.firstOrNull()?.index,
                        ) }
                    }
                    updateFor(version) { it.copy(streamingStatus = "Thinking…") }
                    val accumulated = StringBuilder()
                    withContext(Dispatchers.IO) {
                        ragEngine.askQuestionStream(knowledge, request.query, request.conversationContext).collect { event ->
                            when (event) {
                                is ForensicRagEngine.RagStreamEvent.Status -> updateFor(version) { it.copy(streamingStatus = event.statusText) }
                                is ForensicRagEngine.RagStreamEvent.TokenChunk -> {
                                    accumulated.append(event.token)
                                    updateFor(version) { state -> state.copy(messages = state.messages.map {
                                        if (it.id == request.assistantId) it.copy(text = accumulated.toString()) else it
                                    }) }
                                }
                                is ForensicRagEngine.RagStreamEvent.Complete -> updateFor(version) { state ->
                                    state.copy(
                                        messages = state.messages.map {
                                            if (it.id == request.assistantId) it.copy(
                                                text = event.answer.answerText,
                                                citedTimestamps = event.answer.citedTimestamps,
                                                engineBadge = event.answer.engineBadge,
                                                isStreaming = false,
                                            ) else it
                                        },
                                        selectedChunkIndex = event.answer.retrievedChunks.firstOrNull()?.index ?: state.selectedChunkIndex,
                                    )
                                }
                            }
                        }
                    }
                }
            } catch (timeout: TimeoutCancellationException) {
                failRequest(version, request, "This request took too long. Try a smaller file or retry.")
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                failRequest(version, request, error.localizedMessage?.take(240) ?: "Couldn't finish this response. Please retry.")
            } finally {
                updateFor(version) { state -> state.copy(
                    isGenerating = false, isIndexing = false, streamingStatus = null, indexingProgressText = "",
                    messages = state.messages.map {
                        if (it.id == request.assistantId) it.copy(isStreaming = false) else it
                    },
                ) }
            }
        }
    }

    private fun failRequest(version: Long, request: Request, reason: String) {
        if (version != requestVersion) return
        failedRequest = request
        updateFor(version) { state -> state.copy(
            canRetry = true,
            messages = state.messages.map { message ->
                if (message.id == request.assistantId) message.copy(
                    text = message.text.ifBlank { reason }, isStreaming = false, isError = true,
                ) else message
            },
            errorMessage = reason,
        ) }
    }

    private inline fun updateFor(version: Long, transform: (ChatRagUiState) -> ChatRagUiState) {
        if (version == requestVersion) _uiState.update { if (version == requestVersion) transform(it) else it }
    }

    fun selectTimelineChunk(index: Int) {
        if (_uiState.value.knowledgeBase?.chunks?.none { it.index == index } != false) return
        _uiState.update { it.copy(selectedChunkIndex = index) }
    }

    fun selectTimelineChunkForTimestamp(timestamp: String) {
        val kb = _uiState.value.knowledgeBase ?: return
        kb.chunks.firstOrNull { it.formattedTimeRange == timestamp }?.let {
            selectTimelineChunk(it.index)
            return
        }
        val match = Regex("(\\d{1,2}):(\\d{2})").find(timestamp) ?: return
        val targetMs = ((match.groupValues[1].toLongOrNull() ?: return) * 60 + (match.groupValues[2].toLongOrNull() ?: return)) * 1000
        val chunk = kb.chunks.firstOrNull { targetMs in it.startTimeMs..it.endTimeMs }
            ?: kb.chunks.minByOrNull { kotlin.math.abs(it.startTimeMs - targetMs) }
        chunk?.let { selectTimelineChunk(it.index) }
    }

    fun clearError() { _uiState.update { it.copy(errorMessage = null) } }

    companion object {
        const val MAX_ATTACHMENTS = 4
        const val MAX_MESSAGE_LENGTH = 8_000
        private const val REQUEST_TIMEOUT_MS = 180_000L
    }
}
