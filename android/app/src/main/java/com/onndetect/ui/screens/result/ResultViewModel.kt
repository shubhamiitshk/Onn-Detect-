package com.onndetect.ui.screens.result

import android.content.Context
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.data.repository.ScanHistoryRepository
import com.onndetect.detection.provenance.CanonicalReceiptPayload
import com.onndetect.detection.provenance.ForensicReceipt
import com.onndetect.detection.provenance.ReceiptSigner
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
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
    @param:ApplicationContext private val appContext: Context,
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
     * The SHA-256 of every model in the shipped release manifest.
     *
     * These are the hashes the app's own integrity checker verifies the
     * packaged bytes against, so the receipt names the exact models that
     * produced the verdict.
     */
    private fun releaseModelHashes(): Map<String, String> {
        val manifest = runCatching {
            JSONObject(
                appContext.assets.open("constants/release_manifest.json")
                    .bufferedReader().use { it.readText() }
            )
        }.getOrNull() ?: return emptyMap()
        val models = manifest.optJSONObject("models") ?: return emptyMap()
        val hashes = sortedMapOf<String, String>()
        models.keys().forEach { name -> hashes[name] = models.optJSONObject(name)?.optString("sha256").orEmpty() }
        return hashes
    }

    /** 128 random bits, hex encoded. A fresh value for every export. */
    private fun randomNonce(): String {
        val bytes = ByteArray(16)
        java.security.SecureRandom().nextBytes(bytes)
        return bytes.joinToString("") { "%02x".format(it) }
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
        // The model hashes are READ from the release manifest - the same
        // manifest `ModelIntegrityChecker` verifies the packaged bytes
        // against. A hard-coded hash would make the receipt's model binding
        // a decoration: an enterprise verifier comparing the payload against
        // its own release manifest would reject every receipt.
        val modelHashes = releaseModelHashes()
        // The media digest comes from the assessment's own evidence. There
        // is NO fallback: hashing the scan id would bind the receipt to
        // nothing, and the specification's replay guarantee rests on this
        // binding. A receipt without a real digest is not produced.
        val mediaSha256 = evidence.optString("mediaSha256")
            .takeIf { it.matches(Regex("^[0-9a-fA-F]{64}$")) && !it.all { c -> c == '0' } }
            ?: return throw IllegalStateException(
                "This assessment has no valid media digest, so it cannot be attested"
            )
        // A fresh random nonce per export. The previous value was derived
        // from the scan id, which is a DETERMINISTIC function of the
        // assessment and therefore provides no replay protection at all.
        val challengeNonce = randomNonce()
        // The distance from the decision threshold, normalised: the
        // specification defines `confidence` as that distance, and
        // `ScanResult.confidence` is always zero.
        val threshold = evidence.optDouble("threshold", 0.5).toFloat()
        val confidence = if (threshold in 0f..1f) {
            (kotlin.math.abs(entry.score - threshold) / maxOf(threshold, 1f - threshold)).coerceIn(0f, 1f)
        } else {
            0f
        }

        val data = CanonicalReceiptPayload.ReceiptData(
            timestampEpochMs = entry.timestamp,
            challengeNonce = challengeNonce,
            mediaSha256 = mediaSha256.lowercase(),
            modelRelease = evidence.optString("modelVersion", "unknown"),
            modelHashes = modelHashes,
            verdict = entry.verdict,
            overallScore = entry.score,
            confidence = confidence,
            signals = signalsMap,
            deviceSecurityTier = receiptSigner.securityTier
        )
        val canonical = CanonicalReceiptPayload.serialize(data)
        return receiptSigner.sign(canonical)
    }
}
