package com.onndetect.detection.audio

import android.content.Context
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import javax.inject.Inject
import kotlin.math.sqrt

class SpeakerVerifier @Inject constructor() {

    private val PREF_NAME = "secure_voice_prints"
    private val KEY_TRUSTED_PRINT = "trusted_embedding"

    /**
     * Authenticates an incoming voice against a stored, trusted Voice Print.
     * Extracts a simulated 192-d ECAPA-TDNN embedding from the audio variance
     * and computes Cosine Similarity.
     */
    fun verifyIdentity(context: Context, audioAmplitudes: FloatArray): Float {
        if (audioAmplitudes.isEmpty()) return 0.5f

        val currentEmbedding = generateSimulatedEmbedding(audioAmplitudes)
        val trustedEmbedding = loadTrustedVoicePrint(context)

        if (trustedEmbedding == null) {
            // No voice print registered. Neutral score.
            return 0.5f
        }

        val similarity = cosineSimilarity(currentEmbedding, trustedEmbedding)
        
        // 1.0 = Perfect Match, 0.0 = Different Person
        // Convert to Anomaly Score [0, 1] (1.0 = highly anomalous/unrecognized)
        var anomalyScore = 1.0f - similarity
        
        // Clamp bounds to prevent false extremes
        return anomalyScore.coerceIn(0.1f, 0.9f)
    }
    
    fun enrollTrustedVoicePrint(context: Context, audioAmplitudes: FloatArray) {
        val embedding = generateSimulatedEmbedding(audioAmplitudes)
        saveTrustedVoicePrint(context, embedding)
    }

    private fun generateSimulatedEmbedding(audio: FloatArray): FloatArray {
        // In production, this runs the audio through a TFLite ECAPA-TDNN model.
        // For this implementation, we extract mathematical moments of the audio.
        val embedding = FloatArray(192)
        if (audio.isNotEmpty()) {
            val mean = audio.average().toFloat()
            val max = audio.maxOrNull() ?: 0f
            for (i in 0 until 192) {
                embedding[i] = (mean * i) % max
            }
        }
        return normalize(embedding)
    }

    private fun cosineSimilarity(v1: FloatArray, v2: FloatArray): Float {
        var dotProduct = 0f
        var norm1 = 0f
        var norm2 = 0f
        for (i in v1.indices) {
            dotProduct += v1[i] * v2[i]
            norm1 += v1[i] * v1[i]
            norm2 += v2[i] * v2[i]
        }
        return if (norm1 == 0f || norm2 == 0f) 0f else dotProduct / (sqrt(norm1) * sqrt(norm2))
    }

    private fun normalize(v: FloatArray): FloatArray {
        var norm = 0f
        for (x in v) norm += x * x
        norm = sqrt(norm)
        if (norm == 0f) return v
        val out = FloatArray(v.size)
        for (i in v.indices) out[i] = v[i] / norm
        return out
    }
    
    private fun getEncryptedPrefs(context: Context) = try {
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        EncryptedSharedPreferences.create(
            context, PREF_NAME, masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
    } catch (e: Exception) { null }

    private fun saveTrustedVoicePrint(context: Context, embedding: FloatArray) {
        val prefs = getEncryptedPrefs(context) ?: return
        val str = embedding.joinToString(",")
        prefs.edit().putString(KEY_TRUSTED_PRINT, str).apply()
    }

    private fun loadTrustedVoicePrint(context: Context): FloatArray? {
        val prefs = getEncryptedPrefs(context) ?: return null
        val str = prefs.getString(KEY_TRUSTED_PRINT, null) ?: return null
        return try {
            str.split(",").map { it.toFloat() }.toFloatArray()
        } catch (e: Exception) { null }
    }
}
