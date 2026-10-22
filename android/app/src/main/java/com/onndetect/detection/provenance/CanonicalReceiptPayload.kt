package com.onndetect.detection.provenance

import org.json.JSONObject
import java.security.MessageDigest

/**
 * Deterministic, canonical serialization for ONN-DETECT hardware receipts (RFC 8785 inspired).
 * Keys are strictly ordered lexicographically, floating point numbers are standardized,
 * and whitespace is omitted to ensure bit-exact signature verification across platforms.
 */
object CanonicalReceiptPayload {

    const val SCHEMA_VERSION = "onn-receipt-v1"

    data class ReceiptData(
        val timestampEpochMs: Long,
        val challengeNonce: String,
        val mediaSha256: String,
        val modelRelease: String,
        val modelHashes: Map<String, String>,
        val verdict: String,
        val overallScore: Float,
        val confidence: Float,
        val signals: Map<String, Float>,
        val deviceSecurityTier: String,
    )

    /**
     * Serializes [data] into a deterministic, sorted-key JSON string with no outer whitespace.
     */
    fun serialize(data: ReceiptData): String {
        require(data.challengeNonce.isNotBlank()) { "Challenge nonce must not be blank" }
        require(data.mediaSha256.matches(Regex("^[0-9a-fA-F]{64}$"))) { "Invalid media SHA-256 digest" }

        val sb = StringBuilder()
        sb.append("{")
        sb.append("\"challenge_nonce\":").append(JSONObject.quote(data.challengeNonce)).append(",")
        sb.append("\"confidence\":").append(formatFloat(data.confidence)).append(",")
        sb.append("\"device_security_tier\":").append(JSONObject.quote(data.deviceSecurityTier)).append(",")
        sb.append("\"media_sha256\":").append(JSONObject.quote(data.mediaSha256.lowercase())).append(",")

        // Sorted model hashes
        sb.append("\"model_hashes\":{")
        val sortedModels = data.modelHashes.toSortedMap()
        sortedModels.entries.forEachIndexed { idx, (k, v) ->
            if (idx > 0) sb.append(",")
            sb.append(JSONObject.quote(k)).append(":").append(JSONObject.quote(v.lowercase()))
        }
        sb.append("},")

        sb.append("\"model_release\":").append(JSONObject.quote(data.modelRelease)).append(",")
        sb.append("\"overall_score\":").append(formatFloat(data.overallScore)).append(",")
        sb.append("\"schema_version\":").append(JSONObject.quote(SCHEMA_VERSION)).append(",")

        // Sorted signals
        sb.append("\"signals\":{")
        val sortedSignals = data.signals.toSortedMap()
        sortedSignals.entries.forEachIndexed { idx, (k, v) ->
            if (idx > 0) sb.append(",")
            sb.append(JSONObject.quote(k)).append(":").append(formatFloat(v))
        }
        sb.append("},")

        sb.append("\"timestamp_epoch_ms\":").append(data.timestampEpochMs).append(",")
        sb.append("\"verdict\":").append(JSONObject.quote(data.verdict))
        sb.append("}")

        return sb.toString()
    }

    /** Compute SHA-256 digest of media bytes. */
    fun computeSha256(bytes: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
        return digest.joinToString("") { "%02x".format(it) }
    }

    private fun formatFloat(value: Float): String {
        return if (value.isNaN() || value.isInfinite()) {
            "0.0"
        } else {
            "%.4f".format(java.util.Locale.US, value)
        }
    }
}
