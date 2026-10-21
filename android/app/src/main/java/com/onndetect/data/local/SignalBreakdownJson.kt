package com.onndetect.data.local

import com.onndetect.domain.models.SignalBreakdown
import org.json.JSONObject

/**
 * Compact JSON (de)serialization for persisting [SignalBreakdown] alongside
 * each history row. Uses the platform org.json classes - no extra dependency,
 * no kotlinx-serialization plugin.
 */
object SignalBreakdownJson {

    fun encode(signals: SignalBreakdown): String = JSONObject().apply {
        put("schemaVersion", 1)
        fun safe(v: Float) = v.takeIf { it.isFinite() && it in 0f..1f }?.toDouble() ?: JSONObject.NULL
        put("mobileClipScore", safe(signals.mobileClipScore))
        put("srmForensicScore", safe(signals.srmForensicScore))
        put("fftSpectralScore", safe(signals.fftSpectralScore))
        put("klDivergenceScore", safe(signals.klDivergenceScore))
        put("perturbationScore", safe(signals.perturbationScore))
        put("audioDspScore", safe(signals.audioDspScore))
        put("audioModelScore", safe(signals.audioModelScore))
        put("temporalScore", safe(signals.temporalScore))
        put("provenanceScore", safe(signals.provenanceScore))
        put("prnuScore", safe(signals.prnuScore))
        put("fusionScore", safe(signals.fusionScore))
    }.toString()

    /** @return null when json is null/corrupt; missing fields default to -1. */
    fun decode(json: String?): SignalBreakdown? {
        if (json.isNullOrBlank()) return null
        return try {
            val o = JSONObject(json)
            if (o.optInt("schemaVersion", 1) != 1) return null
            fun f(name: String) = o.optDouble(name, -1.0).toFloat().takeIf { it.isFinite() && it in 0f..1f } ?: -1f
            SignalBreakdown(
                mobileClipScore = f("mobileClipScore"),
                srmForensicScore = f("srmForensicScore"),
                fftSpectralScore = f("fftSpectralScore"),
                klDivergenceScore = f("klDivergenceScore"),
                perturbationScore = f("perturbationScore"),
                audioDspScore = f("audioDspScore"),
                audioModelScore = f("audioModelScore"),
                temporalScore = f("temporalScore"),
                provenanceScore = f("provenanceScore"),
                prnuScore = f("prnuScore"),
                // Absent in rows written before the fusion score was
                // persisted -> -1f (unavailable), never 0f.
                fusionScore = f("fusionScore"),
            )
        } catch (_: Exception) {
            null
        }
    }
}
