package com.onndetect.rag

import com.onndetect.detection.visual.LlamaCppVlmPrefillBackend
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
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
        assertEquals("60:00", ForensicRagEngine.formatMs(3_600_000L))
    }

    private fun createTestVideoKnowledgeBase(): ForensicRagEngine.IndexedMediaKnowledgeBase {
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
            audioSummary = "Speech detected: 'Can you hear me?'.",
            voiceVaultSummary = "Unenrolled speaker.",
            aiProbability = 0.88f,
            srmScore = 0.82f,
            hotspotBoxNorm = floatArrayOf(0.15f, 0.20f, 0.45f, 0.70f),
            hotspotZScore = 2.45f,
            isSceneCut = true,
            hasPersonOrFace = true,
            estimatedSubjectTag = "young person / boy",
        )
        val outroChunk = ForensicRagEngine.TimelineChunk(
            index = 2,
            startTimeMs = 22_000L,
            endTimeMs = 30_000L,
            formattedTimeRange = "00:22 - 00:30",
            sceneTags = listOf("background", "room", "scene"),
            visualDescription = "Empty room.",
            audioSummary = "Silence.",
            voiceVaultSummary = "No voice.",
            aiProbability = 0.18f,
            srmScore = 0.15f,
            isSceneCut = false,
            hasPersonOrFace = false,
            estimatedSubjectTag = "background scene",
        )
        return ForensicRagEngine.IndexedMediaKnowledgeBase(
            sessionId = "test_vid_123",
            mediaKind = ForensicRagEngine.RagMediaKind.VIDEO,
            mediaTitle = "Test Video (3 keyframes, 00:30)",
            sourceUri = null,
            durationMs = 30_000L,
            overallAiProb = 0.52f,
            headlineVerdict = "SYNTHETIC / MANIPULATED",
            activeEngineLabel = "Specialist Forensic RAG",
            chunks = listOf(bgChunk, boyEntryChunk, outroChunk),
            voiceVaultMatchSummary = "Unenrolled speaker",
            scamRiskWarnings = listOf("Peak visual manipulation detected at 00:14 - 00:22 (88% AI probability)."),
        )
    }

    @Test
    fun retrieveTopChunksPrioritizesPersonEntryForBoySceneQuery() {
        val kb = createTestVideoKnowledgeBase()
        val retrieved = ForensicRagEngine.retrieveTopChunks(kb, "When did a boy come into the scene?", topK = 3)

        assertNotNull(retrieved)
        assertTrue("Expected at least 1 retrieved chunk", retrieved.isNotEmpty())
        assertEquals("Top retrieved chunk must be the boy scene entry chunk", 1, retrieved.first().index)
        assertEquals("00:14 - 00:22", retrieved.first().formattedTimeRange)
        assertTrue(retrieved.first().hasPersonOrFace)
    }

    @Test
    fun retrieveTopChunksPrioritizesHighestAnomalyForFakeQuery() {
        val kb = createTestVideoKnowledgeBase()
        val retrieved = ForensicRagEngine.retrieveTopChunks(kb, "Where is the highest fake anomaly timestamp?", topK = 1)

        assertEquals(1, retrieved.size)
        assertEquals(1, retrieved.first().index)
        assertEquals(0.88f, retrieved.first().aiProbability, 0.001f)
    }

    @Test
    fun retrieveTopChunksPrioritizesAudioTrackForVoiceQuery() {
        val kb = createTestVideoKnowledgeBase()
        val retrieved = ForensicRagEngine.retrieveTopChunks(kb, "Does the voice match my Family Voice Vault?", topK = 1)

        assertEquals(1, retrieved.size)
        assertEquals(1, retrieved.first().index)
        assertTrue(retrieved.first().audioSummary.contains("Speech detected"))
    }

    @Test
    fun synthesizeDeterministicGroundedAnswerGeneratesBoySceneEntryCitation() {
        val kb = createTestVideoKnowledgeBase()
        val retrieved = ForensicRagEngine.retrieveTopChunks(kb, "When did a boy come into the scene?", topK = 2)
        val answer = ForensicRagEngine.synthesizeDeterministicGroundedAnswer(
            kb = kb,
            retrieved = retrieved,
            qLower = "when did a boy come into the scene?"
        )

        assertTrue("Answer must mention scene entry", answer.contains("Scene Entry Detected"))
        assertTrue("Answer must cite exact entry timestamp [00:14]", answer.contains("[00:14]"))
        assertTrue("Answer must cite window `00:14 - 00:22`", answer.contains("00:14 - 00:22"))
        assertTrue("Answer must mention young person / boy", answer.contains("young person / boy"))
        assertTrue("Answer must include AI probability 88%", answer.contains("88%"))
    }

    @Test
    fun synthesizeDeterministicGroundedAnswerGeneratesRoiHotspotExplanation() {
        val kb = createTestVideoKnowledgeBase()
        val retrieved = ForensicRagEngine.retrieveTopChunks(kb, "Explain the forensic clues & ROI hotspot", topK = 2)
        val answer = ForensicRagEngine.synthesizeDeterministicGroundedAnswer(
            kb = kb,
            retrieved = retrieved,
            qLower = "explain the forensic clues & roi hotspot"
        )

        assertTrue("Answer must mention Spatial Residual Hotspot Analysis", answer.contains("Spatial Residual Hotspot Analysis"))
        assertTrue("Answer must explain PRNU noise uniformity", answer.contains("PRNU"))
        assertTrue("Answer must include Z-score significance", answer.contains("2.45σ"))
    }

    @Test
    fun synthesizeDeterministicGroundedAnswerHandlesNoPersonDetectedGracefully() {
        val emptyPersonKb = ForensicRagEngine.IndexedMediaKnowledgeBase(
            sessionId = "empty_bg",
            mediaKind = ForensicRagEngine.RagMediaKind.VIDEO,
            mediaTitle = "Nature Video",
            sourceUri = null,
            durationMs = 10_000L,
            overallAiProb = 0.05f,
            headlineVerdict = "LIKELY AUTHENTIC",
            activeEngineLabel = "Specialist Forensic RAG",
            chunks = listOf(
                ForensicRagEngine.TimelineChunk(
                    index = 0,
                    startTimeMs = 0L,
                    endTimeMs = 10_000L,
                    formattedTimeRange = "00:00 - 00:10",
                    sceneTags = listOf("forest", "trees", "nature"),
                    visualDescription = "Forest landscape.",
                    audioSummary = "Wind.",
                    voiceVaultSummary = "No voice.",
                    aiProbability = 0.05f,
                    srmScore = 0.04f,
                    hasPersonOrFace = false,
                )
            ),
            voiceVaultMatchSummary = "No audio",
            scamRiskWarnings = emptyList(),
        )
        val answer = ForensicRagEngine.synthesizeDeterministicGroundedAnswer(
            kb = emptyPersonKb,
            retrieved = emptyPersonKb.chunks,
            qLower = "when did a boy enter?"
        )
        assertTrue(answer.contains("No prominent human face or boy/person track was detected"))
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
