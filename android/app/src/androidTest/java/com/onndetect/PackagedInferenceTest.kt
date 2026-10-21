package com.onndetect

import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.onndetect.detection.audio.AudioClassifier
import com.onndetect.detection.fusion.XGBoostMetaClassifier
import com.onndetect.detection.provenance.C2PAManifestChecker
import com.onndetect.detection.provenance.EXIFAnalyzer
import com.onndetect.detection.visual.*
import com.onndetect.domain.models.*
import com.onndetect.domain.pipeline.DetectionPipeline
import com.onndetect.domain.pipeline.SignalFusion
import com.onndetect.inference.*
import kotlinx.coroutines.flow.last
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.File
import kotlin.math.PI
import kotlin.math.sin

/** Device smoke/parity of packaged interfaces, never a forensic accuracy benchmark. */
@RunWith(AndroidJUnit4::class)
class PackagedInferenceTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext
    @Test fun actualPackagedModelsHaveFiniteScalarOutputAndRemainConsistentWithManifest() {
        ModelIntegrityGate.resetForTest()
        val report = ModelIntegrityGate.report(context)
        assertTrue(report.failures.joinToString(), report.allOk)
        LiteRTInferenceEngine(MemoryMappedModelLoader(context)).use { engine ->
            for (name in ModelIntegrityChecker.REQUIRED_MODELS) {
                val contract = engine.contract(name)
                assertArrayEquals(intArrayOf(1, 1), contract.outputShape)
                val input = ByteBuffer.allocateDirect(contract.inputBytes).order(ByteOrder.nativeOrder())
                while (input.remaining() >= 4) input.putFloat(.1f)
                input.flip()
                val score = engine.predict(name, input)
                assertTrue("Invalid scalar for $name: $score", score.isFinite() && score in 0f..1f)
            }
        }
    }
    @Test fun imagePipelineRespectsPackagedValidationStatus() = runBlocking {
        ModelIntegrityGate.resetForTest()
        val report = ModelIntegrityGate.report(context, setOf(ModelIntegrityChecker.IMAGE_MODEL))
        LiteRTInferenceEngine(MemoryMappedModelLoader(context)).use { engine ->
            val bitmap = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
            try {
                for (y in 0 until 128) for (x in 0 until 128) bitmap.setPixel(x, y,
                    Color.rgb((x * 2) % 256, (y * 2) % 256, (x * 13 + y * 7) % 256))
                val pipeline = DetectionPipeline(DeviceTier.HIGH_END, SignalFusion(), SRMForensicAnalyzer(context, engine),
                    FFTSpectralAnalyzer(), KLCooccurrenceAnalyzer(context), XGBoostMetaClassifier(),
                    EXIFAnalyzer(), C2PAManifestChecker(), PRNUAnalyzer(),
                    // semanticProbe was added to the DetectionPipeline
                    // constructor in 34bd2f2. Omitting it left this
                    // instrumentation test with one argument too few and
                    // the androidTest source set uncompilable.
                    SemanticProbeAnalyzer(context, engine), context)
                val result = pipeline.analyzeMedia(bitmap, ScanMode.IMAGE_ONLY).last().result!!
                if (!report.validated) {
                    assertEquals(DetectionVerdict.INCONCLUSIVE, result.verdict)
                    assertEquals(AnalysisStatus.INCONCLUSIVE, result.status)
                    assertFalse(result.isAiGenerated)
                    assertFalse(result.modelValidated)
                    assertEquals(-1f, result.overallScore, 0f)
                } else {
                    // A validated release may still legitimately ABSTAIN:
                    // the cascade returns INCONCLUSIVE when Engine A and
                    // Engine C disagree inside the ambiguity band. That is
                    // correct behaviour, not a failure, so this test must
                    // accept both outcomes. What it must still guarantee
                    // is that a classified verdict is well formed.
                    // JUnit's Assert takes the MESSAGE FIRST: assertTrue(message, condition).
                    assertTrue(
                        "Unexpected status ${result.status}: ${result.reason}",
                        result.status == AnalysisStatus.SUCCESS ||
                            result.status == AnalysisStatus.INCONCLUSIVE
                    )
                    assertTrue(result.modelValidated)
                    if (result.verdict != DetectionVerdict.INCONCLUSIVE) {
                        assertEquals(AnalysisStatus.SUCCESS, result.status)
                        assertTrue(result.overallScore.isFinite() && result.overallScore in 0f..1f)
                        assertTrue(result.decisionThreshold != null)
                    }
                }
            } finally { bitmap.recycle() }
        }
    }
    private fun wav(name: String, mode: Int): File {
        val samples = 64000
        val buffer = ByteBuffer.allocate(44 + samples * 2).order(ByteOrder.LITTLE_ENDIAN)
        buffer.put("RIFF".toByteArray(Charsets.US_ASCII)); buffer.putInt(36 + samples * 2)
        buffer.put("WAVEfmt ".toByteArray(Charsets.US_ASCII)); buffer.putInt(16); buffer.putShort(1); buffer.putShort(1)
        buffer.putInt(16000); buffer.putInt(32000); buffer.putShort(2); buffer.putShort(16)
        buffer.put("data".toByteArray(Charsets.US_ASCII)); buffer.putInt(samples * 2)
        for (i in 0 until samples) {
            val sample = when (mode) {
                0 -> 0
                1 -> (sin(2 * PI * 440 * i / 16000) * 8000).toInt()
                else -> {
                    // Syllable-modulatedvoiced/unvoiced synthetic speech-like fixture
                    val t = i.toDouble() / 16000.0
                    val syllableEnv = (0.5 + 0.5 * sin(2 * PI * 3.5 * t)).let { if (it < 0.25) 0.02 else it }
                    val voiced = (sin(2 * PI * 6.0 * t) >= -0.2)
                    val carrier = if (voiced) (0.6 * sin(2 * PI * 140 * t) + 0.3 * sin(2 * PI * 720 * t) + 0.1 * sin(2 * PI * 2400 * t))
                    else (sin(2 * PI * 3700 * t + sin(2 * PI * 890 * t) * 4.0))
                    (syllableEnv * carrier * 12000).toInt().coerceIn(-32000, 32000)
                }
            }
            buffer.putShort(sample.toShort())
        }
        return File(context.cacheDir, name).apply { writeBytes(buffer.array()) }
    }
    @Test fun decoderRecordsSampleCoverageAndSilenceOrPureToneCannotBecomeAuthentic() {
        ModelIntegrityGate.resetForTest()
        val report = ModelIntegrityGate.report(context, setOf(ModelIntegrityChecker.AUDIO_MODEL))
        val silence = wav("onn-smoke-silence.wav", 0)
        val tone = wav("onn-smoke-tone.wav", 1)
        val speechLike = wav("onn-smoke-speechlike.wav", 2)
        try {
            LiteRTInferenceEngine(MemoryMappedModelLoader(context)).use { engine ->
                val classifier = AudioClassifier(context, engine)
                val empty = classifier.analyze(Uri.fromFile(silence))
                assertEquals(AnalysisStatus.INCONCLUSIVE, empty.status)
                assertFalse(empty.isAiGenerated); assertEquals(-1f, empty.score, 0f)
                val diagnostic = classifier.analyze(Uri.fromFile(tone))
                assertEquals(AnalysisStatus.INCONCLUSIVE, diagnostic.status)
                assertFalse(diagnostic.isAiGenerated)
                assertEquals(report.validated, diagnostic.modelValidated)
                assertTrue(diagnostic.reason, diagnostic.score.isFinite() && diagnostic.score in 0f..1f)
                assertTrue(diagnostic.analyzedDurationMs in 3500..4000)
                assertTrue(diagnostic.totalDurationMs >= diagnostic.analyzedDurationMs)
                val speechResult = classifier.analyze(Uri.fromFile(speechLike))
                assertTrue(speechResult.score.isFinite() && speechResult.score in 0f..1f)
                if (report.validated) {
                    assertEquals(AnalysisStatus.SUCCESS, speechResult.status)
                    assertTrue(speechResult.modelValidated)
                } else {
                    assertEquals(AnalysisStatus.INCONCLUSIVE, speechResult.status)
                }
            }
        } finally { silence.delete(); tone.delete(); speechLike.delete() }
    }
}

