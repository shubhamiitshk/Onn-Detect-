package com.onndetect.inference

import android.content.Context

/** Immutable in-APK assets are checked once per required modality set. */
object ModelIntegrityGate {
    private val cache = HashMap<Set<String>, ModelIntegrityChecker.Report>()
    @Synchronized fun report(context: Context, required: Set<String> = ModelIntegrityChecker.REQUIRED_MODELS): ModelIntegrityChecker.Report {
        val key = required.toSet()
        return cache.getOrPut(key) { ModelIntegrityChecker.verifyModelsDetailed(context, key) }
    }
    fun isOk(context: Context): Boolean = report(context).allOk
    fun failures(context: Context): List<String> = report(context).failures
    @Synchronized fun resetForTest() { cache.clear() }
}
