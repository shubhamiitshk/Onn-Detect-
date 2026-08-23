package com.onndetect.inference

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.security.MessageDigest

/**
 * Verifies shipped .tflite files against SHA-256 digests recorded at export
 * time (see ml/scripts/generate_android_constants.py -> model_hashes.json).
 */
object ModelIntegrityChecker {

    private const val MANIFEST = "constants/model_hashes.json"
    private const val TAG = "ModelIntegrity"

    fun verifyAllModels(context: Context): Boolean = verifyAllModelsDetailed(context).allOk

    data class Report(val allOk: Boolean, val failures: List<String>)

    fun verifyAllModelsDetailed(context: Context): Report {
        val expected = try {
            val root = JSONObject(
                context.assets.open(MANIFEST).bufferedReader().use { it.readText() }
            )
            val models = root.getJSONObject("models")
            val names = models.names()
            (0 until models.length()).associate { i ->
                val key = names.getString(i)
                key to models.getString(key)
            }
        } catch (e: Exception) {
            Log.w(TAG, "No hash manifest present; skipping integrity check", e)
            return Report(allOk = true, failures = emptyList())
        }

        if (expected.isEmpty()) return Report(true, emptyList())

        val failures = mutableListOf<String>()
        for ((fileName, expectedHash) in expected) {
            try {
                val actual = context.assets.open("models/$fileName").use { computeSHA256(it) }
                if (!actual.equals(expectedHash, ignoreCase = true)) {
                    failures += "$fileName: hash mismatch"
                }
            } catch (e: Exception) {
                failures += "$fileName: missing/unreadable (${e.message})"
            }
        }
        if (failures.isNotEmpty()) Log.e(TAG, "Integrity failures: $failures")
        return Report(failures.isEmpty(), failures)
    }

    private fun computeSHA256(inputStream: java.io.InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(8192)
        var read: Int
        while (inputStream.read(buffer).also { read = it } != -1) {
            digest.update(buffer, 0, read)
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
