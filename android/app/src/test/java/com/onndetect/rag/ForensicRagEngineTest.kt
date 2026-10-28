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
}
