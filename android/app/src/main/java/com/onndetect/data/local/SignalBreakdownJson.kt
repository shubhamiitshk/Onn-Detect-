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
        put("vfmProbeScore", safe(signals.vfmProbeScore))
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
        put("landmarkJitter", safe(signals.landmarkJitter))
    }.toString()

    /** @return null when json is null/corrupt; missing or null fields default to Float.NaN. */
    fun decode(json: String?): SignalBreakdown? {
        if (json.isNullOrBlank()) return null
        return try {
            val o = JSONObject(json)
            if (o.optInt("schemaVersion", 1) != 1) return null
            fun f(name: String): Float = if (o.isNull(name)) {
                Float.NaN
            } else {
                o.optDouble(name, Double.NaN).toFloat().takeIf { it.isFinite() && it in 0f..1f } ?: Float.NaN
            }
            // Slot 6's key was renamed from "mobileClipScore" to
            // "vfmProbeScore". The shape of the record did not change, so
            // the schema version stays 1 and the OLD key is still read:
            // history rows written by earlier releases must decode.
            fun fOr(name: String, legacy: String) = if (o.has(name)) f(name) else f(legacy)
            SignalBreakdown(
                vfmProbeScore = fOr("vfmProbeScore", "mobileClipScore"),
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
                // persisted -> Float.NaN (unavailable), never 0f.
                fusionScore = f("fusionScore"),
                landmarkJitter = f("landmarkJitter"),
            )
        } catch (_: Exception) {
            null
        }
    }
}
