package com.onndetect.ui.screens.result

import com.onndetect.data.local.ScanHistoryEntity
import com.onndetect.data.local.SignalBreakdownJson
import com.onndetect.detection.provenance.CanonicalReceiptPayload
import com.onndetect.detection.provenance.ForensicReceipt
import com.onndetect.domain.models.SignalBreakdown
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Locale

/**
 * Comprehensive Unit Test Suite for [ForensicCertificateExporter].
 *
 * Verifies:
 *  1. Complete modality telemetry formatting (all 6 engine rows with exact percentages).
 *  2. Missing modality formatting (Float.NaN signals serialized/formatted as "N/A" without crashing).
 *  3. Edge-case boundary score formatting (0.0f, 1.0f, Float.NaN, +/- Infinity, out-of-range).
 *  4. Share intent text formatting resilience (preventing fatal crash on Float.NaN scores).
 *  5. Cryptographic SHA-256 media digest splitting (64 hex characters into two 32-char lines).
 *  6. Unified JSON certificate export structure and RFC 8785 canonical receipt payload compliance.
 *  7. Multi-engine composite row selection across image, audio, and video modalities.
 */
class ForensicCertificateExporterTest {

    private fun createSampleEntry(
        id: String = "test-assessment-uuid-12345678",
        score: Float = 0.8942f,
        verdict: String = "SYNTHETIC",
        status: String = "SUCCESS",
        modelValidated: Boolean = true,
        signalsJson: String? = null,
        durationMs: Long = 245L,
        scanMode: String = "IMAGE_ANALYSIS",
    ) = ScanHistoryEntity(
        id = id,
        timestamp = 1729612800000L,
        scanMode = scanMode,
        score = score,
        isAiGenerated = score >= 0.58f,
        confidence = 0.92f,
        durationMs = durationMs,
        signalsJson = signalsJson,
        status = status,
        verdict = verdict,
        reason = null,
        modelValidated = modelValidated,
        evidenceJson = """{"modelVersion":"onn-prod-v10","threshold":0.5782,"mediaSha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}"""
    )

    // =========================================================================
    // 1. Boundary Score Formatting Tests
    // =========================================================================

    @Test
    fun `score boundary 0f formats to 0 percent`() {
        assertEquals("0%", ForensicCertificateExporter.formatScorePercent(0.0f))
        assertEquals("0.0%", ForensicCertificateExporter.formatTelemetryValue(0.0f))
    }

    @Test
    fun `score boundary 1f formats to 100 percent`() {
        assertEquals("100%", ForensicCertificateExporter.formatScorePercent(1.0f))
        assertEquals("100.0%", ForensicCertificateExporter.formatTelemetryValue(1.0f))
    }

    @Test
    fun `score boundary Float_NaN formats to NA without throwing`() {
        assertEquals("N/A", ForensicCertificateExporter.formatScorePercent(Float.NaN))
        assertEquals("N/A", ForensicCertificateExporter.formatTelemetryValue(Float.NaN))
    }

    @Test
    fun `score boundary positive and negative infinity format to NA`() {
        assertEquals("N/A", ForensicCertificateExporter.formatScorePercent(Float.POSITIVE_INFINITY))
        assertEquals("N/A", ForensicCertificateExporter.formatScorePercent(Float.NEGATIVE_INFINITY))
        assertEquals("N/A", ForensicCertificateExporter.formatTelemetryValue(Float.POSITIVE_INFINITY))
        assertEquals("N/A", ForensicCertificateExporter.formatTelemetryValue(Float.NEGATIVE_INFINITY))
    }

    @Test
    fun `out-of-bounds scores format to NA`() {
        assertEquals("N/A", ForensicCertificateExporter.formatScorePercent(-0.001f))
        assertEquals("N/A", ForensicCertificateExporter.formatScorePercent(1.001f))
        assertEquals("N/A", ForensicCertificateExporter.formatScorePercent(-100f))
        assertEquals("N/A", ForensicCertificateExporter.formatTelemetryValue(-0.01f))
        assertEquals("N/A", ForensicCertificateExporter.formatTelemetryValue(1.5f))
    }

    @Test
    fun `threshold percentage formatting handles standard and boundary values`() {
        assertEquals("58%", ForensicCertificateExporter.formatThresholdPercent(0.5782f))
        assertEquals("50%", ForensicCertificateExporter.formatThresholdPercent(0.50f))
        assertEquals("0%", ForensicCertificateExporter.formatThresholdPercent(0.0f))
        assertEquals("100%", ForensicCertificateExporter.formatThresholdPercent(1.0f))
        assertEquals("N/A", ForensicCertificateExporter.formatThresholdPercent(Float.NaN))
        assertEquals("N/A", ForensicCertificateExporter.formatThresholdPercent(Float.POSITIVE_INFINITY))
    }

    // =========================================================================
    // 2. Multi-Engine Telemetry Breakdown Tests
    // =========================================================================

    @Test
    fun `telemetry extraction with complete modality signals produces exact percentages`() {
        val signals = SignalBreakdown(
            vfmProbeScore = 0.8523f,
            fusionScore = 0.9145f,
            srmForensicScore = 0.7301f,
            fftSpectralScore = 0.6254f,
            klDivergenceScore = 0.4500f,
            perturbationScore = 0.8800f,
            audioModelScore = Float.NaN,
            landmarkJitter = Float.NaN,
        )

        val rows = ForensicCertificateExporter.buildTelemetryRows(signals)
        assertEquals(6, rows.size)

        assertEquals("Engine C (SigLIP-B/16 TAP Probe)", rows[0].first)
        assertEquals(0.8523f, rows[0].second, 0.0001f)
        assertEquals("85.2%", ForensicCertificateExporter.formatTelemetryValue(rows[0].second))

        assertEquals("Engine A (XGBoost 7-Tree Fusion)", rows[1].first)
        assertEquals(0.9145f, rows[1].second, 0.0001f)
        assertEquals("91.4%", ForensicCertificateExporter.formatTelemetryValue(rows[1].second))

        assertEquals("20-Ch SRM + PatchCraft Mosaic", rows[2].first)
        assertEquals(0.7301f, rows[2].second, 0.0001f)
        assertEquals("73.0%", ForensicCertificateExporter.formatTelemetryValue(rows[2].second))

        assertEquals("2D FFT Radial Power Spectrum", rows[3].first)
        assertEquals(0.6254f, rows[3].second, 0.0001f)
        assertEquals("62.5%", ForensicCertificateExporter.formatTelemetryValue(rows[3].second))

        assertEquals("Pixel Co-Occurrence KL Divergence", rows[4].first)
        assertEquals(0.4500f, rows[4].second, 0.0001f)
        assertEquals("45.0%", ForensicCertificateExporter.formatTelemetryValue(rows[4].second))

        assertEquals("Localized ROI / Audio / BioLip", rows[5].first)
        assertEquals(0.8800f, rows[5].second, 0.0001f)
        assertEquals("88.0%", ForensicCertificateExporter.formatTelemetryValue(rows[5].second))
    }

    @Test
    fun `telemetry extraction with all-NaN signals formats all rows as NA`() {
        val allNans = SignalBreakdown() // All default to Float.NaN
        val rows = ForensicCertificateExporter.buildTelemetryRows(allNans)
        assertEquals(6, rows.size)

        for ((label, score) in rows) {
            assertTrue("Score for $label must be NaN", score.isNaN())
            assertEquals("Formatted value for $label must be N/A", "N/A", ForensicCertificateExporter.formatTelemetryValue(score))
        }
    }

    @Test
    fun `telemetry extraction with null signals formats all rows as NA`() {
        val rows = ForensicCertificateExporter.buildTelemetryRows(null)
        assertEquals(6, rows.size)
        for ((label, score) in rows) {
            assertTrue("Score for $label must be NaN", score.isNaN())
            assertEquals("N/A", ForensicCertificateExporter.formatTelemetryValue(score))
        }
    }

    @Test
    fun `audio-only scan formats visual engines as NA and extracts audioModelScore in composite row`() {
        val audioSignals = SignalBreakdown(
            vfmProbeScore = Float.NaN,
            fusionScore = Float.NaN,
            srmForensicScore = Float.NaN,
            fftSpectralScore = Float.NaN,
            klDivergenceScore = Float.NaN,
            perturbationScore = Float.NaN,
            audioModelScore = 0.7654f,
            landmarkJitter = Float.NaN,
        )

        val rows = ForensicCertificateExporter.buildTelemetryRows(audioSignals)
        assertEquals(6, rows.size)

        // Visual rows must be NaN / N/A
        for (i in 0..4) {
            assertTrue(rows[i].second.isNaN())
            assertEquals("N/A", ForensicCertificateExporter.formatTelemetryValue(rows[i].second))
        }

        // Composite row must pick audioModelScore
        assertEquals("Localized ROI / Audio / BioLip", rows[5].first)
        assertEquals(0.7654f, rows[5].second, 0.0001f)
        assertEquals("76.5%", ForensicCertificateExporter.formatTelemetryValue(rows[5].second))
    }

    @Test
    fun `video scan selects maximum of perturbation, audio, and landmark jitter in composite row`() {
        val videoSignals = SignalBreakdown(
            perturbationScore = 0.35f,
            audioModelScore = 0.62f,
            landmarkJitter = 0.88f,
        )
        val rows = ForensicCertificateExporter.buildTelemetryRows(videoSignals)
        assertEquals(0.88f, rows[5].second, 0.0001f)
        assertEquals("88.0%", ForensicCertificateExporter.formatTelemetryValue(rows[5].second))
    }

    // =========================================================================
    // 3. Share Intent Text Formatting & Regression Defense
    // =========================================================================

    @Test
    fun `share text generation formats score cleanly for standard scores`() {
        val entry = createSampleEntry(score = 0.8942f, verdict = "SYNTHETIC")
        val shareText = ForensicCertificateExporter.formatShareText(entry)
        assertEquals("ONN-DETECT Forensic Verification Certificate (SYNTHETIC • Score 89%)", shareText)
    }

    @Test
    fun `share text generation handles Float_NaN score without crashing`() {
        // REGRESSION TEST: Line 243 in ForensicCertificateExporter previously called
        // (entry.score * 100).roundToInt() which crashed with IllegalArgumentException on NaN.
        val nanEntry = createSampleEntry(score = Float.NaN, verdict = "INCONCLUSIVE", status = "MODEL_FAILURE", modelValidated = false)
        val shareText = ForensicCertificateExporter.formatShareText(nanEntry)
        assertEquals("ONN-DETECT Forensic Verification Certificate (INCONCLUSIVE • Score N/A)", shareText)
    }

    @Test
    fun `share text generation handles boundary scores 0f and 1f`() {
        val zeroEntry = createSampleEntry(score = 0.0f, verdict = "AUTHENTIC")
        assertEquals("ONN-DETECT Forensic Verification Certificate (AUTHENTIC • Score 0%)", ForensicCertificateExporter.formatShareText(zeroEntry))

        val oneEntry = createSampleEntry(score = 1.0f, verdict = "SYNTHETIC")
        assertEquals("ONN-DETECT Forensic Verification Certificate (SYNTHETIC • Score 100%)", ForensicCertificateExporter.formatShareText(oneEntry))
    }

    // =========================================================================
    // 4. Media SHA-256 Digest Formatting Tests
    // =========================================================================

    @Test
    fun `media SHA-256 splitting splits 64-character digest into two 32-character halves`() {
        val sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        val (p1, p2) = ForensicCertificateExporter.formatSha256(sha)
        assertEquals(32, p1.length)
        assertEquals(32, p2.length)
        assertEquals("e3b0c44298fc1c149afbf4c8996fb924", p1)
        assertEquals("27ae41e4649b934ca495991b7852b855", p2)
    }

    @Test
    fun `media SHA-256 splitting handles short or unavailable digests safely`() {
        val (p1, p2) = ForensicCertificateExporter.formatSha256("UNAVAILABLE")
        assertEquals("UNAVAILABLE", p1)
        assertEquals("", p2)

        val (empty1, empty2) = ForensicCertificateExporter.formatSha256("")
        assertEquals("", empty1)
        assertEquals("", empty2)
    }

    // =========================================================================
    // 5. Unified JSON Certificate & Canonical Receipt Format Tests
    // =========================================================================

    @Test
    fun `generateCertificateJson produces structured audit JSON with valid scores`() {
        val signals = SignalBreakdown(vfmProbeScore = 0.85f, fusionScore = 0.90f)
        val entry = createSampleEntry(
            score = 0.89f,
            signalsJson = SignalBreakdownJson.encode(signals)
        )
        val evidence = JSONObject("""{"modelVersion":"onn-prod-v10","mediaSha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}""")

        val receipt = ForensicReceipt(
            canonicalPayloadJson = """{"verdict":"SYNTHETIC"}""",
            signatureBase64 = "MEQCIBx...==",
            publicKeyPem = "-----BEGIN PUBLIC KEY-----\nMFkw...\n-----END PUBLIC KEY-----",
            securityTier = "STRONGBOX"
        )

        val jsonStr = ForensicCertificateExporter.generateCertificateJson(entry, evidence, receipt)
        val json = JSONObject(jsonStr)

        assertEquals("ONN-DETECT-FORENSIC-CERTIFICATE-V1", json.getString("certificate_type"))
        assertEquals(entry.id, json.getString("assessment_id"))
        assertEquals("SYNTHETIC", json.getString("verdict"))
        assertEquals(0.89, json.getDouble("score"), 0.001)
        assertTrue(json.getBoolean("model_validated"))
        assertEquals("IMAGE_ANALYSIS", json.getString("modality"))
        assertTrue(json.has("evidence"))
        assertTrue(json.has("signals"))
        assertTrue(json.has("cryptographic_receipt"))

        val cryptoJson = json.getJSONObject("cryptographic_receipt")
        assertEquals("STRONGBOX", cryptoJson.getString("security_tier"))
        assertEquals("SHA256withECDSA", cryptoJson.getString("signature_algorithm"))
    }

    @Test
    fun `generateCertificateJson serializes NaN score as JSON null`() {
        val entry = createSampleEntry(score = Float.NaN, verdict = "INCONCLUSIVE", status = "MODEL_FAILURE", modelValidated = false)
        val evidence = JSONObject()

        val jsonStr = ForensicCertificateExporter.generateCertificateJson(entry, evidence, null)
        val json = JSONObject(jsonStr)

        assertTrue("score must be serialized as JSON null when Float.NaN", json.isNull("score"))
        assertFalse(json.has("cryptographic_receipt"))
    }

    @Test
    fun `canonical receipt payload enforces strictly lexicographically sorted keys and 4dp floats`() {
        val data = CanonicalReceiptPayload.ReceiptData(
            timestampEpochMs = 1729612800000L,
            challengeNonce = "nonce-cert-test-4321",
            mediaSha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            modelRelease = "onn-prod-v10",
            modelHashes = mapOf("srm_forensic_cnn" to "e90b7a42", "siglip_probe" to "9c11bd18"),
            verdict = "SYNTHETIC",
            overallScore = 0.89416f,
            confidence = 0.92001f,
            signals = mapOf("srm_score" to 0.68712f, "vfm_score" to 0.89100f),
            deviceSecurityTier = "STRONGBOX"
        )

        val serialized = CanonicalReceiptPayload.serialize(data)

        // Verify key ordering
        val expectedPrefix = "{\"challenge_nonce\":\"nonce-cert-test-4321\",\"confidence\":0.9200,\"device_security_tier\":\"STRONGBOX\""
        assertTrue(serialized.startsWith(expectedPrefix))

        // Verify 4 decimal places formatting
        assertTrue(serialized.contains("\"overall_score\":0.8942"))
        assertTrue(serialized.contains("\"confidence\":0.9200"))
        assertTrue(serialized.contains("\"srm_score\":0.6871"))
        assertTrue(serialized.contains("\"vfm_score\":0.8910"))
    }

    @Test(expected = IllegalArgumentException::class)
    fun `canonical receipt payload strictly rejects non-finite float scores`() {
        val data = CanonicalReceiptPayload.ReceiptData(
            timestampEpochMs = 1729612800000L,
            challengeNonce = "nonce-nan",
            mediaSha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            modelRelease = "onn-prod-v10",
            modelHashes = mapOf("model" to "hash"),
            verdict = "INCONCLUSIVE",
            overallScore = Float.NaN,
            confidence = 0.0f,
            signals = emptyMap(),
            deviceSecurityTier = "TEE"
        )
        CanonicalReceiptPayload.serialize(data)
    }
}
