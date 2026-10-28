package com.onndetect.ui.screens.chat

import android.net.Uri
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.data.repository.ScanHistoryRepository
import com.onndetect.rag.ForensicRagEngine
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import javax.inject.Inject

data class ChatMessage(
    val id: String,
    val isUser: Boolean,
    val text: String,
    val citedTimestamps: List<String> = emptyList(),
    val engineBadge: String? = null,
)

data class ChatRagUiState(
    val isIndexing: Boolean = false,
    val indexingProgressText: String = "",
    val isGenerating: Boolean = false,
    val knowledgeBase: ForensicRagEngine.IndexedMediaKnowledgeBase? = null,
    val messages: List<ChatMessage> = emptyList(),
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
                runCatching { historyRepository.observePage(1, 0).first().firstOrNull() }.getOrNull()
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
                        "Ask me anything about this scan, or attach a Video / Audio / Image above for frame-by-frame timeline RAG.",
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

    fun indexMedia(uri: Uri, mediaKind: ForensicRagEngine.RagMediaKind) {
        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isIndexing = true,
                    indexingProgressText = "Starting offline multimodal RAG indexing...",
                    errorMessage = null,
                )
            }
            try {
                val kb = withContext(Dispatchers.IO) {
                    when (mediaKind) {
                        ForensicRagEngine.RagMediaKind.VIDEO ->
                            ragEngine.indexVideoUri(uri) { msg ->
                                _uiState.update { s -> s.copy(indexingProgressText = msg) }
                            }
                        ForensicRagEngine.RagMediaKind.AUDIO ->
                            ragEngine.indexAudioUri(uri) { msg ->
                                _uiState.update { s -> s.copy(indexingProgressText = msg) }
                            }
                        else ->
                            ragEngine.indexImageUri(uri) { msg ->
                                _uiState.update { s -> s.copy(indexingProgressText = msg) }
                            }
                    }
                }
                val introText = buildString {
                    append("Indexed **${kb.mediaTitle}** (${kb.chunks.size} timeline segments).\n")
                    append("• **Forensic Verdict**: **${kb.headlineVerdict}** (${(kb.overallAiProb * 100f).toInt()}% AI risk)\n")
                    append("• **Voice Vault**: ${kb.voiceVaultMatchSummary}\n")
                    append("Ask a question below (e.g., *\"When did a boy come into the scene?\"* or *\"Where is the fake anomaly?\"*).")
                }
                val welcomeMsg = ChatMessage(
                    id = "sys_${System.currentTimeMillis()}",
                    isUser = false,
                    text = introText,
                    citedTimestamps = kb.chunks.take(3).map { it.formattedTimeRange },
                    engineBadge = kb.activeEngineLabel,
                )
                _uiState.update {
                    it.copy(
                        isIndexing = false,
                        knowledgeBase = kb,
                        messages = listOf(welcomeMsg),
                        selectedChunkIndex = 0,
                        errorMessage = null,
                    )
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isIndexing = false,
                        errorMessage = e.message ?: "Failed to index media file",
                    )
                }
            }
        }
    }

    fun askQuestion(question: String) {
        val trimmed = question.trim()
        if (trimmed.isEmpty()) return
        val currentKb = _uiState.value.knowledgeBase
        val userMsg = ChatMessage(
            id = "usr_${System.currentTimeMillis()}",
            isUser = true,
            text = trimmed,
        )

        if (currentKb == null) {
            val promptAttach = ChatMessage(
                id = "bot_${System.currentTimeMillis()}",
                isUser = false,
                text = "Please select a **Video**, **Audio**, or **Image** above (or tap **Ask Forensic AI** from any scan result) so I can index its timeline and forensic signals offline.",
                engineBadge = "Offline RAG Router",
            )
            _uiState.update { it.copy(messages = it.messages + userMsg + promptAttach) }
            return
        }

        viewModelScope.launch {
            _uiState.update {
                it.copy(
                    isGenerating = true,
                    messages = it.messages + userMsg,
                    errorMessage = null,
                )
            }
            try {
                val ragAnswer = withContext(Dispatchers.IO) {
                    ragEngine.askQuestion(currentKb, trimmed)
                }
                val botMsg = ChatMessage(
                    id = "bot_${System.currentTimeMillis()}",
                    isUser = false,
                    text = ragAnswer.answerText,
                    citedTimestamps = ragAnswer.citedTimestamps,
                    engineBadge = ragAnswer.engineBadge,
                )
                val firstIdx = ragAnswer.retrievedChunks.firstOrNull()?.index ?: _uiState.value.selectedChunkIndex
                _uiState.update {
                    it.copy(
                        isGenerating = false,
                        messages = it.messages + botMsg,
                        selectedChunkIndex = firstIdx,
                    )
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isGenerating = false,
                        errorMessage = e.message ?: "RAG query failed",
                    )
                }
            }
        }
    }

    fun selectTimelineChunk(index: Int) {
        _uiState.update { it.copy(selectedChunkIndex = index) }
    }

    fun clearError() {
        _uiState.update { it.copy(errorMessage = null) }
    }
}
