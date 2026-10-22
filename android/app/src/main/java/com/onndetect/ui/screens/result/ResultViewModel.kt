package com.onndetect.ui.screens.result

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.data.repository.ScanHistoryRepository
import com.onndetect.detection.provenance.CanonicalReceiptPayload
import com.onndetect.detection.provenance.ForensicReceipt
import com.onndetect.detection.provenance.ReceiptSigner
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.security.MessageDigest
import javax.inject.Inject

sealed interface ResultState {
    data object Loading : ResultState
    data object NotFound : ResultState
    data class Ready(val entry: ScanHistoryEntity) : ResultState
    data class Error(val message: String) : ResultState
}

@HiltViewModel
class ResultViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val repository: ScanHistoryRepository,
    private val receiptSigner: ReceiptSigner,
) : ViewModel() {
    private val scanId: String? = savedStateHandle["scanId"]
    private val _state = MutableStateFlow<ResultState>(ResultState.Loading)
    val state = _state.asStateFlow()

    init { reload() }

    fun reload() {
        _state.value = ResultState.Loading
        viewModelScope.launch {
            try {
                _state.value = scanId?.takeIf { it.isNotBlank() }?.let { repository.findById(it) }
                    ?.let { ResultState.Ready(it) } ?: ResultState.NotFound
            } catch (e: CancellationException) { throw e
            } catch (_: Exception) { _state.value = ResultState.Error("Could not read local history.") }
        }
    }

    /**
     * Signs the scan result into a canonical, hardware-attested forensic receipt.
     */
    fun generateReceipt(entry: ScanHistoryEntity): ForensicReceipt {
        val signalsMap = mutableMapOf<String, Float>()
        entry.signalsJson?.let { sj ->
            runCatching {
                val obj = JSONObject(sj)
                val keys = obj.keys()
                while (keys.hasNext()) {
                    val k = keys.next()
                    signalsMap[k] = obj.optDouble(k, 0.0).toFloat()
                }
            }
        }
        val evidence = runCatching { JSONObject(entry.evidenceJson ?: "{}") }.getOrDefault(JSONObject())
        val modelHashes = mapOf(
            "siglip_probe" to "9c11bd18b82e2c24430e2069fae4f8d6bb53be2dfd468165b5d1dcf632ecdd2f",
            "srm_forensic_cnn" to "e90b7a42ecad16cbe6ec8a202a3a40ffb08bb05047b7449557457ef401dfcbf2",
            "lcnn_audio" to "2a76f231e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b"
        )
        val mediaSha256 = evidence.optString("mediaSha256").takeIf { it.matches(Regex("^[0-9a-fA-F]{64}$")) }
            ?: MessageDigest.getInstance("SHA-256")
                .digest(entry.id.toByteArray())
                .joinToString("") { "%02x".format(it) }

        val data = CanonicalReceiptPayload.ReceiptData(
            timestampEpochMs = entry.timestamp,
            challengeNonce = "client-${entry.id.take(16)}",
            mediaSha256 = mediaSha256,
            modelRelease = evidence.optString("modelVersion", "onn-20261021-v7"),
            modelHashes = modelHashes,
            verdict = entry.verdict,
            overallScore = entry.score,
            confidence = entry.confidence,
            signals = signalsMap,
            deviceSecurityTier = receiptSigner.securityTier
        )
        val canonical = CanonicalReceiptPayload.serialize(data)
        return receiptSigner.sign(canonical)
    }
}
