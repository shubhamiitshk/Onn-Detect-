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
        put("mobileClipScore", signals.mobileClipScore.toDouble())
        put("srmForensicScore", signals.srmForensicScore.toDouble())
        put("fftSpectralScore", signals.fftSpectralScore.toDouble())
        put("klDivergenceScore", signals.klDivergenceScore.toDouble())
        put("perturbationScore", signals.perturbationScore.toDouble())
        put("audioDspScore", signals.audioDspScore.toDouble())
        put("audioModelScore", signals.audioModelScore.toDouble())
        put("temporalScore", signals.temporalScore.toDouble())
        put("provenanceScore", signals.provenanceScore.toDouble())
        put("prnuScore", signals.prnuScore.toDouble())
    }.toString()

    /** @return null when json is null/corrupt; missing fields default to -1. */
    fun decode(json: String?): SignalBreakdown? {
        if (json.isNullOrBlank()) return null
        return try {
            val o = JSONObject(json)
            fun f(name: String) = o.optDouble(name, -1.0).toFloat()
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
            )
        } catch (_: Exception) {
            null
        }
    }
}
