package com.onndetect.inference

import android.content.Context
import org.json.JSONObject

/** Immutable in-APK assets are checked once per required modality set. */
object ModelIntegrityGate {
    private val cache = HashMap<Set<String>, ModelIntegrityChecker.Report>()
    @Synchronized fun report(context: Context, required: Set<String> = ModelIntegrityChecker.REQUIRED_MODELS): ModelIntegrityChecker.Report {
        val key = required.toSet()
        return cache.getOrPut(key) { ModelIntegrityChecker.verifyModelsDetailed(context, key) }
    }
    fun isOk(context: Context): Boolean = report(context).allOk
    /**
     * The shipped release id, read from the release manifest.
     *
     * The home screen used to hard-code it, so it named a release that had
     * been superseded. A release id is a fact about the APK's contents, not
     * a constant to be maintained by hand.
     */
    fun releaseId(context: Context): String = runCatching {
        val manifest = JSONObject(
            context.assets.open("constants/release_manifest.json").bufferedReader().use { it.readText() }
        )
        manifest.optString("release_id", "unknown")
    }.getOrDefault("unknown")
    fun failures(context: Context): List<String> = report(context).failures
    @Synchronized fun resetForTest() { cache.clear() }
}
