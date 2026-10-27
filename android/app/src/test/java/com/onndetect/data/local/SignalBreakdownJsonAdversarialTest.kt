package com.onndetect.data.local

import com.onndetect.domain.models.SignalBreakdown
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

/**
 * Adversarial stress harness for [SignalBreakdownJson].
 * Tests IEEE 754 NaN preservation, null serialization, negative sentinel rejection,
 * schema evolution, and fuzzing resilience.
 */
class SignalBreakdownJsonAdversarialTest {

    @Test
    fun allNansEncodeToAllNullsAndDecodeToAllNans() {
        val allNans = SignalBreakdown() // All defaults are Float.NaN
        val jsonStr = SignalBreakdownJson.encode(allNans)
        val jsonObj = JSONObject(jsonStr)

        val signalKeys = listOf(
            "vfmProbeScore", "srmForensicScore", "fftSpectralScore", "klDivergenceScore",
            "perturbationScore", "audioDspScore", "audioModelScore", "temporalScore",
            "provenanceScore", "prnuScore", "fusionScore", "landmarkJitter"
        )

        for (key in signalKeys) {
            assertTrue("Key $key must be serialized as null in JSON", jsonObj.isNull(key))
        }

        val decoded = SignalBreakdownJson.decode(jsonStr)
        assertNotNull("Decoded SignalBreakdown must not be null", decoded)
        assertTrue("vfmProbeScore must be NaN", decoded!!.vfmProbeScore.isNaN())
        assertTrue("srmForensicScore must be NaN", decoded.srmForensicScore.isNaN())
        assertTrue("fftSpectralScore must be NaN", decoded.fftSpectralScore.isNaN())
        assertTrue("klDivergenceScore must be NaN", decoded.klDivergenceScore.isNaN())
        assertTrue("perturbationScore must be NaN", decoded.perturbationScore.isNaN())
        assertTrue("audioDspScore must be NaN", decoded.audioDspScore.isNaN())
        assertTrue("audioModelScore must be NaN", decoded.audioModelScore.isNaN())
        assertTrue("temporalScore must be NaN", decoded.temporalScore.isNaN())
        assertTrue("provenanceScore must be NaN", decoded.provenanceScore.isNaN())
        assertTrue("prnuScore must be NaN", decoded.prnuScore.isNaN())
        assertTrue("fusionScore must be NaN", decoded.fusionScore.isNaN())
        assertTrue("landmarkJitter must be NaN", decoded.landmarkJitter.isNaN())
    }

    @Test
    fun allValidScoresRoundtripFaithfully() {
        val original = SignalBreakdown(
            vfmProbeScore = 0.12f,
            srmForensicScore = 0.23f,
            fftSpectralScore = 0.34f,
            klDivergenceScore = 0.45f,
            perturbationScore = 0.56f,
            audioDspScore = 0.67f,
            audioModelScore = 0.78f,
            temporalScore = 0.89f,
            provenanceScore = 0.91f,
            prnuScore = 0.92f,
            fusionScore = 0.93f,
            landmarkJitter = 0.94f,
        )

        val jsonStr = SignalBreakdownJson.encode(original)
        val decoded = SignalBreakdownJson.decode(jsonStr)
        assertNotNull(decoded)

        assertEquals(0.12f, decoded!!.vfmProbeScore, 1e-4f)
        assertEquals(0.23f, decoded.srmForensicScore, 1e-4f)
        assertEquals(0.34f, decoded.fftSpectralScore, 1e-4f)
        assertEquals(0.45f, decoded.klDivergenceScore, 1e-4f)
        assertEquals(0.56f, decoded.perturbationScore, 1e-4f)
        assertEquals(0.67f, decoded.audioDspScore, 1e-4f)
        assertEquals(0.78f, decoded.audioModelScore, 1e-4f)
        assertEquals(0.89f, decoded.temporalScore, 1e-4f)
        assertEquals(0.91f, decoded.provenanceScore, 1e-4f)
        assertEquals(0.92f, decoded.prnuScore, 1e-4f)
        assertEquals(0.93f, decoded.fusionScore, 1e-4f)
        assertEquals(0.94f, decoded.landmarkJitter, 1e-4f)
    }

    @Test
    fun legacyMobileClipScoreBackwardCompatibility() {
        val legacyJson = """{"schemaVersion": 1, "mobileClipScore": 0.8123}"""
        val decodedLegacy = SignalBreakdownJson.decode(legacyJson)
        assertNotNull(decodedLegacy)
        assertEquals(0.8123f, decodedLegacy!!.vfmProbeScore, 1e-4f)

        val updatedJson = """{"schemaVersion": 1, "vfmProbeScore": 0.75, "mobileClipScore": 0.50}"""
        val decodedUpdated = SignalBreakdownJson.decode(updatedJson)
        assertNotNull(decodedUpdated)
        assertEquals(0.75f, decodedUpdated!!.vfmProbeScore, 1e-4f)
    }

    @Test
    fun negativeSentinelsAndInfinitiesAreSanitizedToNullAndNan() {
        val dirty = SignalBreakdown(
            vfmProbeScore = -1.0f,
            srmForensicScore = Float.NEGATIVE_INFINITY,
            fftSpectralScore = Float.POSITIVE_INFINITY,
            klDivergenceScore = 2.0f,
            perturbationScore = -0.001f,
            audioDspScore = 1.001f,
            audioModelScore = Float.NaN,
            temporalScore = -100.0f,
            provenanceScore = Float.NaN,
            prnuScore = Float.POSITIVE_INFINITY,
            fusionScore = -1.0f,
            landmarkJitter = Float.NaN,
        )

        val jsonStr = SignalBreakdownJson.encode(dirty)
        val jsonObj = JSONObject(jsonStr)

        val signalKeys = listOf(
            "vfmProbeScore", "srmForensicScore", "fftSpectralScore", "klDivergenceScore",
            "perturbationScore", "audioDspScore", "audioModelScore", "temporalScore",
            "provenanceScore", "prnuScore", "fusionScore", "landmarkJitter"
        )
        for (key in signalKeys) {
            assertTrue("Key $key must be serialized as null when invalid", jsonObj.isNull(key))
        }

        val decoded = SignalBreakdownJson.decode(jsonStr)
        assertNotNull(decoded)
        assertTrue(decoded!!.vfmProbeScore.isNaN())
        assertTrue(decoded.srmForensicScore.isNaN())
        assertTrue(decoded.fftSpectralScore.isNaN())
        assertTrue(decoded.klDivergenceScore.isNaN())
        assertTrue(decoded.perturbationScore.isNaN())
        assertTrue(decoded.audioDspScore.isNaN())
        assertTrue(decoded.audioModelScore.isNaN())
        assertTrue(decoded.temporalScore.isNaN())
        assertTrue(decoded.provenanceScore.isNaN())
        assertTrue(decoded.prnuScore.isNaN())
        assertTrue(decoded.fusionScore.isNaN())
        assertTrue(decoded.landmarkJitter.isNaN())
    }

    @Test
    fun rawNegativeValuesInJsonDecodeToNan() {
        val rawJson = """
            {
                "schemaVersion": 1,
                "srmForensicScore": -1.0,
                "audioModelScore": -0.5,
                "temporalScore": 99.0,
                "prnuScore": -1
            }
        """.trimIndent()
        val decoded = SignalBreakdownJson.decode(rawJson)
        assertNotNull(decoded)
        assertTrue(decoded!!.srmForensicScore.isNaN())
        assertTrue(decoded.audioModelScore.isNaN())
        assertTrue(decoded.temporalScore.isNaN())
        assertTrue(decoded.prnuScore.isNaN())
    }

    @Test
    fun partialJsonAndMissingKeysDefaultToNan() {
        val json = """{"schemaVersion": 1, "srmForensicScore": 0.42}"""
        val decoded = SignalBreakdownJson.decode(json)
        assertNotNull(decoded)
        assertEquals(0.42f, decoded!!.srmForensicScore, 1e-4f)
        assertTrue(decoded.vfmProbeScore.isNaN())
        assertTrue(decoded.fftSpectralScore.isNaN())
        assertTrue(decoded.temporalScore.isNaN())
        assertTrue(decoded.audioModelScore.isNaN())
    }

    @Test
    fun malformedOrUnsupportedSchemaReturnsNull() {
        assertNull(SignalBreakdownJson.decode(null))
        assertNull(SignalBreakdownJson.decode(""))
        assertNull(SignalBreakdownJson.decode("   "))
        assertNull(SignalBreakdownJson.decode("not valid json"))
        assertNull(SignalBreakdownJson.decode("{\"schemaVersion\": 2}"))
        assertNull(SignalBreakdownJson.decode("{\"schemaVersion\": -1}"))
    }

    @Test
    fun corruptFieldTypesDefaultToNanWithoutCrashing() {
        val json = """
            {
                "schemaVersion": 1,
                "srmForensicScore": "not a number",
                "audioDspScore": true,
                "temporalScore": [1, 2, 3],
                "fusionScore": {"nested": "obj"}
            }
        """.trimIndent()
        val decoded = SignalBreakdownJson.decode(json)
        assertNotNull(decoded)
        assertTrue(decoded!!.srmForensicScore.isNaN())
        assertTrue(decoded.audioDspScore.isNaN())
        assertTrue(decoded.temporalScore.isNaN())
        assertTrue(decoded.fusionScore.isNaN())
    }

    @Test
    fun randomFuzzingRoundtripInvariance() {
        val rand = Random(12345)
        val edgeValues = listOf(
            Float.NaN,
            Float.NEGATIVE_INFINITY,
            Float.POSITIVE_INFINITY,
            -1.0f,
            -0.001f,
            0.0f,
            0.0001f,
            0.5f,
            0.9999f,
            1.0f,
            1.001f,
            10.0f
        )

        fun genFloat(): Float = if (rand.nextBoolean()) edgeValues.random(rand) else rand.nextFloat()

        for (step in 0..500) {
            val original = SignalBreakdown(
                vfmProbeScore = genFloat(),
                srmForensicScore = genFloat(),
                fftSpectralScore = genFloat(),
                klDivergenceScore = genFloat(),
                perturbationScore = genFloat(),
                audioDspScore = genFloat(),
                audioModelScore = genFloat(),
                temporalScore = genFloat(),
                provenanceScore = genFloat(),
                prnuScore = genFloat(),
                fusionScore = genFloat(),
                landmarkJitter = genFloat(),
            )

            val jsonStr = SignalBreakdownJson.encode(original)
            val decoded = SignalBreakdownJson.decode(jsonStr)
            assertNotNull(decoded)

            fun checkField(orig: Float, dec: Float, name: String) {
                if (orig.isFinite() && orig in 0f..1f) {
                    assertEquals("Field $name roundtrip mismatch", orig, dec, 1e-4f)
                } else {
                    assertTrue("Field $name ($orig) must decode to NaN, got $dec", dec.isNaN())
                }
            }

            checkField(original.vfmProbeScore, decoded!!.vfmProbeScore, "vfmProbeScore")
            checkField(original.srmForensicScore, decoded.srmForensicScore, "srmForensicScore")
            checkField(original.fftSpectralScore, decoded.fftSpectralScore, "fftSpectralScore")
            checkField(original.klDivergenceScore, decoded.klDivergenceScore, "klDivergenceScore")
            checkField(original.perturbationScore, decoded.perturbationScore, "perturbationScore")
            checkField(original.audioDspScore, decoded.audioDspScore, "audioDspScore")
            checkField(original.audioModelScore, decoded.audioModelScore, "audioModelScore")
            checkField(original.temporalScore, decoded.temporalScore, "temporalScore")
            checkField(original.provenanceScore, decoded.provenanceScore, "provenanceScore")
            checkField(original.prnuScore, decoded.prnuScore, "prnuScore")
            checkField(original.fusionScore, decoded.fusionScore, "fusionScore")
            checkField(original.landmarkJitter, decoded.landmarkJitter, "landmarkJitter")
        }
    }
}
