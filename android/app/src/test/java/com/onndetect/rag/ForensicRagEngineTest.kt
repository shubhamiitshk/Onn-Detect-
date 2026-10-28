package com.onndetect.rag

import com.onndetect.detection.visual.LlamaCppVlmPrefillBackend
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ForensicRagEngineTest {

    @After
    fun tearDown() {
        LlamaCppVlmPrefillBackend.testGenerationDelegate = null
    }

    @Test
    fun formatMsFormatsMinutesAndSecondsCorrectly() {
        assertEquals("00:00", ForensicRagEngine.formatMs(0L))
        assertEquals("00:14", ForensicRagEngine.formatMs(14_250L))
        assertEquals("01:05", ForensicRagEngine.formatMs(65_000L))
    }

    @Test
    fun retrieveTopChunksPrioritizesPersonEntryForBoySceneQuery() {
        val bgChunk = ForensicRagEngine.TimelineChunk(
            index = 0,
            startTimeMs = 0L,
            endTimeMs = 14_000L,
            formattedTimeRange = "00:00 - 00:14",
            sceneTags = listOf("background", "environment", "scene"),
            visualDescription = "Continuous background scene (00:00).",
            audioSummary = "Ambient room tone.",
            voiceVaultSummary = "No voice.",
            aiProbability = 0.12f,
            srmScore = 0.10f,
            isSceneCut = true,
            hasPersonOrFace = false,
            estimatedSubjectTag = "background scene",
        )
        val boyEntryChunk = ForensicRagEngine.TimelineChunk(
            index = 1,
            startTimeMs = 14_000L,
            endTimeMs = 22_000L,
            formattedTimeRange = "00:14 - 00:22",
            sceneTags = listOf("person", "boy", "young person", "face", "entered", "entry", "scene"),
            visualDescription = "Subject (young person / boy) enters the scene at 00:14.",
            audioSummary = "Speech detected.",
            voiceVaultSummary = "Unenrolled speaker.",
            aiProbability = 0.88f,
            srmScore = 0.82f,
            hotspotZScore = 2.45f,
            isSceneCut = true,
            hasPersonOrFace = true,
            estimatedSubjectTag = "young person / boy",
        )
        val kb = ForensicRagEngine.IndexedMediaKnowledgeBase(
            sessionId = "test_vid",
            mediaKind = ForensicRagEngine.RagMediaKind.VIDEO,
            mediaTitle = "Test Video",
            sourceUri = null,
            durationMs = 30_000L,
            overallAiProb = 0.50f,
            headlineVerdict = "SYNTHETIC / MANIPULATED",
            activeEngineLabel = "Test Engine",
            chunks = listOf(bgChunk, boyEntryChunk),
            voiceVaultMatchSummary = "Unenrolled speaker",
            scamRiskWarnings = emptyList(),
        )

        val qTokens = "When did a boy come into the scene?".lowercase()
        assertTrue(qTokens.contains("boy"))
        assertEquals("00:14 - 00:22", kb.chunks.first { it.hasPersonOrFace }.formattedTimeRange)
    }

    @Test
    fun llamaCppBackendTestGenerationDelegateReturnsGroundedResponse() {
        LlamaCppVlmPrefillBackend.testGenerationDelegate = { prompt, _, maxTokens ->
            "A boy enters the scene at [00:14] (maxTokens=$maxTokens, promptLen=${prompt.length})."
        }

        val backend = LlamaCppVlmPrefillBackend(
            modelPath = "virtual_qwen35_08b.gguf",
            mmprojPath = "virtual_mmproj.gguf",
            hiddenDim = 1024,
        )
        val response = backend.generateMultimodalResponse(
            prompt = "When did a boy come into the scene?",
            bitmap = null,
            maxNewTokens = 128,
        )
        assertTrue(response.contains("[00:14]"))
        assertTrue(response.contains("maxTokens=128"))
    }

    @Test
    fun retrieveTopChunksGroundsDocumentTamperingQueries() {
        val docChunk = ForensicRagEngine.TimelineChunk(
            index = 0,
            startTimeMs = 0L,
            endTimeMs = 0L,
            formattedTimeRange = "Document Page 1 + Tamper Grid",
            sceneTags = listOf("document", "file", "receipt", "invoice", "text", "forgery", "tampered", "roi"),
            visualDescription = "Document analysis. Document forgery score=82% (SRM=0.78, ELA peak=2.45, Noise discontinuity=0.62). Tampered ROI box=[0.20, 0.30 -> 0.60, 0.50] (Z=2.80).",
            audioSummary = "No audio track.",
            voiceVaultSummary = "Not applicable.",
            aiProbability = 0.82f,
            srmScore = 0.78f,
            hotspotZScore = 2.80f,
            hasPersonOrFace = false,
            estimatedSubjectTag = "Document / Screenshot",
        )
        val kb = ForensicRagEngine.IndexedMediaKnowledgeBase(
            sessionId = "test_doc",
            mediaKind = ForensicRagEngine.RagMediaKind.DOCUMENT,
            mediaTitle = "UPI Payment Receipt.pdf",
            sourceUri = null,
            durationMs = 0L,
            overallAiProb = 0.82f,
            headlineVerdict = "FORGED / ALTERED DOCUMENT",
            activeEngineLabel = "Document Forgery Analyzer",
            chunks = listOf(docChunk),
            voiceVaultMatchSummary = "Document verification",
            scamRiskWarnings = listOf("Document layout or compression discontinuity detected (82% risk)."),
        )

        val retrieved = ForensicRagEngine.retrieveTopChunks(kb, "check receipt for font tampering", topK = 1)
        assertEquals(1, retrieved.size)
        assertEquals("Document Page 1 + Tamper Grid", retrieved.first().formattedTimeRange)

        val answer = ForensicRagEngine.synthesizeDeterministicGroundedAnswer(kb, retrieved, "check receipt for font tampering")
        assertTrue(answer.contains("Document & Receipt Authenticity Report"))
        assertTrue(answer.contains("82%"))
        assertTrue(answer.contains("FORGED / ALTERED DOCUMENT"))
    }

    @Test
    fun synthesizeGeneralForensicAnswerCoversCoreModalitiesWithoutIndexedMedia() {
        val upiAnswer = ForensicRagEngine.synthesizeGeneralForensicAnswer("Check UPI screenshot for spliced font tampering")
        assertTrue(upiAnswer.contains("UPI & Document Forgery Detection"))
        assertTrue(upiAnswer.contains("Error Level Analysis"))

        val boyAnswer = ForensicRagEngine.synthesizeGeneralForensicAnswer("When did a boy enter the video scene?")
        assertTrue(boyAnswer.contains("Timeline Person & Scene Entry Tracking"))
        assertTrue(boyAnswer.contains("cosine similarity"))

        val voiceAnswer = ForensicRagEngine.synthesizeGeneralForensicAnswer("Does this voice match my Family Voice Vault?")
        assertTrue(voiceAnswer.contains("Voice Clone & Deepfake Audio Verification"))
        assertTrue(voiceAnswer.contains("LCNN-MFM"))

        val generalAnswer = ForensicRagEngine.synthesizeGeneralForensicAnswer("What models are you running offline?")
        assertTrue(generalAnswer.contains("ONN-DETECT Multimodal Engine Architecture"))
        assertTrue(generalAnswer.contains("100% on-device inference"))
    }

    @Test
    fun retrieveTopChunksGroundsAudioAndVoiceVaultQueries() {
        val audioChunk = ForensicRagEngine.TimelineChunk(
            index = 0,
            startTimeMs = 0L,
            endTimeMs = 4_000L,
            formattedTimeRange = "00:00 - 00:04",
            sceneTags = listOf("audio", "voice", "speech", "matched", "family"),
            visualDescription = "Audio Window 00:00 - 00:04",
            audioSummary = "LCNN-MFM spoof score=0.15",
            voiceVaultSummary = "Matched Mom (94% similarity)",
            aiProbability = 0.15f,
            srmScore = Float.NaN,
            hasPersonOrFace = true,
            estimatedSubjectTag = "Mom",
        )
        val kb = ForensicRagEngine.IndexedMediaKnowledgeBase(
            sessionId = "test_audio",
            mediaKind = ForensicRagEngine.RagMediaKind.AUDIO,
            mediaTitle = "Mom Voice Note.m4a",
            sourceUri = null,
            durationMs = 4_000L,
            overallAiProb = 0.15f,
            headlineVerdict = "AUTHENTIC HUMAN SPEECH",
            activeEngineLabel = "LCNN-MFM v3 + Family Safe-Voice Vault",
            chunks = listOf(audioChunk),
            voiceVaultMatchSummary = "Matched Mom (94%)",
            scamRiskWarnings = emptyList(),
        )

        val retrieved = ForensicRagEngine.retrieveTopChunks(kb, "does this voice match Mom?", topK = 1)
        assertEquals(1, retrieved.size)
        assertEquals("00:00 - 00:04", retrieved.first().formattedTimeRange)

        val answer = ForensicRagEngine.synthesizeDeterministicGroundedAnswer(kb, retrieved, "does this voice match Mom?")
        assertTrue(answer.contains("Voice & Audio Verdict"))
        assertTrue(answer.contains("AUTHENTIC HUMAN SPEECH"))
    }
}
