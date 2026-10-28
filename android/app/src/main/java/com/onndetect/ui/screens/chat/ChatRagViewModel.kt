package com.onndetect.ui.screens.chat

import android.net.Uri
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.data.repository.ScanHistoryRepository
import com.onndetect.rag.ForensicRagEngine
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import javax.inject.Inject

data class AttachedMedia(
    val uri: Uri,
    val kind: ForensicRagEngine.RagMediaKind,
    val fileName: String,
    val mimeType: String? = null,
    val fileSizeFormatted: String? = null,
)

data class ChatMessage(
    val id: String,
    val isUser: Boolean,
    val text: String,
    val attachment: AttachedMedia? = null,
    val citedTimestamps: List<String> = emptyList(),
    val engineBadge: String? = null,
    val isStreaming: Boolean = false,
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
)

@HiltViewModel
class ChatRagViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val ragEngine: ForensicRagEngine,
    private val historyRepository: ScanHistoryRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(ChatRagUiState())
    val uiState: StateFlow<ChatRagUiState> = _uiState.asStateFlow()

    private var generationJob: Job? = null

    init {
        val initialScanId = savedStateHandle.get<String>("scanId")?.takeIf { it.isNotBlank() }
        if (initialScanId != null) {
            loadFromScanHistory(initialScanId)
        } else {
            loadMostRecentHistoryIfAvailable()
        }
    }

    private fun loadMostRecentHistoryIfAvailable() {
        viewModelScope.launch {
            val latest = withContext(Dispatchers.IO) {
                runCatching {
                    var found: ScanHistoryEntity? = null
                    historyRepository.observeAll().firstOrNull()?.let { list ->
                        found = list.firstOrNull()
                    }
                    found
                }.getOrNull()
            }
            if (latest != null && _uiState.value.knowledgeBase == null) {
                loadFromScanHistory(latest.id)
            }
        }
    }

    fun loadFromScanHistory(scanId: String) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isIndexing = true,
                    indexingProgressText = "Loading forensic telemetry for Scan #${scanId.take(8)}...",
                    errorMessage = null,
                )
            }
            try {
                val entity = withContext(Dispatchers.IO) {
                    historyRepository.findById(scanId)
                }
                if (entity == null) {
                    _uiState.update {
                        it.copy(
                            isIndexing = false,
                            errorMessage = "Scan record not found in local database.",
                        )
                    }
                    return@launch
                }
                val kb = withContext(Dispatchers.IO) {
                    ragEngine.indexFromScanHistory(entity)
                }
                val welcomeMsg = ChatMessage(
                    id = "sys_${System.currentTimeMillis()}",
                    isUser = false,
                    text = "Indexed **${kb.mediaTitle}** — Verdict: **${kb.headlineVerdict}**.\n" +
                        "Ask me anything about this scan, or attach an Image / Video / Audio / Document file below for frame-by-frame multimodal RAG.",
                    citedTimestamps = kb.chunks.map { it.formattedTimeRange },
                    engineBadge = kb.activeEngineLabel,
                )
                _uiState.update {
                    it.copy(
                        isIndexing = false,
                        knowledgeBase = kb,
                        messages = listOf(welcomeMsg),
                        selectedChunkIndex = 0,
                    )
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isIndexing = false,
                        errorMessage = e.message ?: "Failed to load scan telemetry",
                    )
                }
            }
        }
    }

    fun stageAttachment(
        uri: Uri,
        kind: ForensicRagEngine.RagMediaKind,
        fileName: String,
        mimeType: String? = null,
        fileSizeFormatted: String? = null,
    ) {
        _uiState.update {
            it.copy(
                stagedAttachment = AttachedMedia(
                    uri = uri,
                    kind = kind,
                    fileName = fileName,
                    mimeType = mimeType,
                    fileSizeFormatted = fileSizeFormatted,
                )
            )
        }
    }

    fun clearStagedAttachment() {
        _uiState.update { it.copy(stagedAttachment = null) }
    }

    fun clearConversation() {
        generationJob?.cancel()
        generationJob = null
        _uiState.update {
            it.copy(
                messages = emptyList(),
                isGenerating = false,
                streamingStatus = null,
                selectedChunkIndex = null,
                errorMessage = null,
            )
        }
    }

    fun cancelGeneration() {
        generationJob?.cancel()
        generationJob = null
        _uiState.update { state ->
            val updated = state.messages.mapNotNull { msg ->
                if (msg.isStreaming) {
                    if (msg.text.isBlank()) null
                    else msg.copy(isStreaming = false, text = "${msg.text}\n\n*[Generation stopped]*")
                } else msg
            }
            state.copy(
                isGenerating = false,
                isIndexing = false,
                streamingStatus = null,
                messages = updated,
            )
        }
    }

    fun indexMedia(uri: Uri, mediaKind: ForensicRagEngine.RagMediaKind) {
        val defaultName = when (mediaKind) {
            ForensicRagEngine.RagMediaKind.VIDEO -> "Video Clip"
            ForensicRagEngine.RagMediaKind.AUDIO -> "Voice Recording"
            ForensicRagEngine.RagMediaKind.DOCUMENT -> "Document File"
            else -> "Image ROI"
        }
        stageAttachment(uri, mediaKind, defaultName)
        sendMessage("")
    }

    fun askQuestion(question: String) {
        sendMessage(question)
    }

    fun sendMessage(text: String) {
        val query = text.trim()
        val staged = _uiState.value.stagedAttachment
        if (query.isEmpty() && staged == null) return

        val effectiveText = if (query.isNotBlank()) {
            query
        } else {
            when (staged?.kind) {
                ForensicRagEngine.RagMediaKind.VIDEO -> "Analyze this video for AI manipulation, scene cuts, and deepfake anomalies."
                ForensicRagEngine.RagMediaKind.AUDIO -> "Inspect this voice recording for synthetic cloning and Family Voice Vault match."
                ForensicRagEngine.RagMediaKind.DOCUMENT -> "Verify this document for layout tampering, ELA compression anomalies, and forged edits."
                else -> "Examine this image for generative artifacts, high-pass SRM noise, and localized tampering."
            }
        }

        val userMsgId = "usr_${System.currentTimeMillis()}"
        val userMsg = ChatMessage(
            id = userMsgId,
            isUser = true,
            text = effectiveText,
            attachment = staged,
        )

        _uiState.update {
            it.copy(
                messages = it.messages + userMsg,
                stagedAttachment = null,
                errorMessage = null,
            )
        }

        generationJob?.cancel()
        generationJob = viewModelScope.launch {
            try {
                var currentKb = _uiState.value.knowledgeBase

                if (staged != null) {
                    _uiState.update {
                        it.copy(
                            isIndexing = true,
                            isGenerating = true,
                            streamingStatus = "Indexing ${staged.fileName}...",
                            indexingProgressText = "Analyzing multimodal embeddings...",
                        )
                    }
                    val newKb = withContext(Dispatchers.IO) {
                        when (staged.kind) {
                            ForensicRagEngine.RagMediaKind.VIDEO ->
                                ragEngine.indexVideoUri(staged.uri) { progress ->
                                    _uiState.update { s -> s.copy(indexingProgressText = progress, streamingStatus = progress) }
                                }
                            ForensicRagEngine.RagMediaKind.AUDIO ->
                                ragEngine.indexAudioUri(staged.uri) { progress ->
                                    _uiState.update { s -> s.copy(indexingProgressText = progress, streamingStatus = progress) }
                                }
                            ForensicRagEngine.RagMediaKind.DOCUMENT ->
                                ragEngine.indexDocumentUri(staged.uri, fileName = staged.fileName, mimeType = staged.mimeType) { progress ->
                                    _uiState.update { s -> s.copy(indexingProgressText = progress, streamingStatus = progress) }
                                }
                            else ->
                                ragEngine.indexImageUri(staged.uri) { progress ->
                                    _uiState.update { s -> s.copy(indexingProgressText = progress, streamingStatus = progress) }
                                }
                        }
                    }
                    currentKb = newKb
                    _uiState.update {
                        it.copy(
                            isIndexing = false,
                            knowledgeBase = newKb,
                            selectedChunkIndex = 0,
                            indexingProgressText = "",
                        )
                    }
                }

                _uiState.update {
                    it.copy(
                        isGenerating = true,
                        streamingStatus = if (currentKb != null) "Searching timeline embeddings..." else "Accessing forensic AI knowledge...",
                    )
                }

                val botMsgId = "bot_${System.currentTimeMillis()}"
                val initialBotMsg = ChatMessage(
                    id = botMsgId,
                    isUser = false,
                    text = "",
                    isStreaming = true,
                )
                _uiState.update { it.copy(messages = it.messages + initialBotMsg) }

                var streamedContent = ""
                withContext(Dispatchers.IO) {
                    ragEngine.askQuestionStream(currentKb, effectiveText).collect { event ->
                        when (event) {
                            is ForensicRagEngine.RagStreamEvent.Status -> {
                                _uiState.update { it.copy(streamingStatus = event.statusText) }
                            }
                            is ForensicRagEngine.RagStreamEvent.TokenChunk -> {
                                streamedContent += event.token
                                _uiState.update { state ->
                                    val updated = state.messages.map { msg ->
                                        if (msg.id == botMsgId) msg.copy(text = streamedContent) else msg
                                    }
                                    state.copy(messages = updated)
                                }
                            }
                            is ForensicRagEngine.RagStreamEvent.Complete -> {
                                val finalAns = event.answer
                                val firstIdx = finalAns.retrievedChunks.firstOrNull()?.index ?: _uiState.value.selectedChunkIndex
                                _uiState.update { state ->
                                    val updated = state.messages.map { msg ->
                                        if (msg.id == botMsgId) {
                                            msg.copy(
                                                text = finalAns.answerText,
                                                citedTimestamps = finalAns.citedTimestamps,
                                                engineBadge = finalAns.engineBadge,
                                                isStreaming = false,
                                            )
                                        } else msg
                                    }
                                    state.copy(
                                        messages = updated,
                                        isGenerating = false,
                                        streamingStatus = null,
                                        selectedChunkIndex = firstIdx,
                                    )
                                }
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                if (e is kotlinx.coroutines.CancellationException) throw e
                _uiState.update { state ->
                    val updated = state.messages.mapNotNull { msg ->
                        if (msg.isStreaming) {
                            if (msg.text.isBlank()) null
                            else msg.copy(
                                text = "${msg.text}\n\n*[Interrupted: ${e.message ?: "Analysis failed"}]*",
                                isStreaming = false,
                                engineBadge = "Error"
                            )
                        } else msg
                    }
                    state.copy(
                        isGenerating = false,
                        isIndexing = false,
                        streamingStatus = null,
                        messages = updated,
                        errorMessage = e.message ?: "Analysis failed",
                    )
                }
            }
        }
    }

    fun selectTimelineChunk(index: Int) {
        _uiState.update { it.copy(selectedChunkIndex = index) }
    }

    fun selectTimelineChunkForTimestamp(timestamp: String) {
        val kb = _uiState.value.knowledgeBase ?: return
        if (kb.chunks.isEmpty()) return

        val match = Regex("(\\d{1,2}):(\\d{2})").find(timestamp)
        val targetMs = if (match != null) {
            val minutes = match.groupValues[1].toLongOrNull() ?: 0L
            val seconds = match.groupValues[2].toLongOrNull() ?: 0L
            (minutes * 60L + seconds) * 1000L
        } else null

        val chunk = if (targetMs != null) {
            kb.chunks.firstOrNull { targetMs in it.startTimeMs..it.endTimeMs } ?: kb.chunks.minByOrNull {
                kotlin.math.abs(it.startTimeMs - targetMs)
            }
        } else {
            kb.chunks.firstOrNull { it.formattedTimeRange.contains(timestamp) }
        }

        chunk?.let { selectTimelineChunk(it.index) }
    }

    fun clearError() {
        _uiState.update { it.copy(errorMessage = null) }
    }
}
