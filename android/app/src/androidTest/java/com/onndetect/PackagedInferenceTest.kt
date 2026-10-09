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
    @Test fun actualPackagedModelsHaveFiniteScalarOutputAndRemainUnvalidated() {
        ModelIntegrityGate.resetForTest()
        val report = ModelIntegrityGate.report(context)
        assertTrue(report.failures.joinToString(), report.allOk)
        assertFalse("Legacy byte validity must not claim calibrated accuracy", report.validated)
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
    @Test fun imagePipelineCannotClassifyUnvalidatedPackagedAssets() = runBlocking {
        LiteRTInferenceEngine(MemoryMappedModelLoader(context)).use { engine ->
            val bitmap = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888)
            try {
                for (y in 0 until 64) for (x in 0 until 64) bitmap.setPixel(x, y,
                    Color.rgb(x * 4, y * 4, (x * 13 + y * 7) % 256))
                val pipeline = DetectionPipeline(DeviceTier.HIGH_END, SignalFusion(), SRMForensicAnalyzer(context, engine),
                    FFTSpectralAnalyzer(), KLCooccurrenceAnalyzer(context), XGBoostMetaClassifier(),
                    EXIFAnalyzer(), C2PAManifestChecker(), PRNUAnalyzer(), context)
                val result = pipeline.analyzeMedia(bitmap, ScanMode.IMAGE_ONLY).last().result!!
                assertEquals(DetectionVerdict.INCONCLUSIVE, result.verdict)
                assertEquals(AnalysisStatus.INCONCLUSIVE, result.status)
                assertFalse(result.isAiGenerated)
                assertFalse(result.modelValidated)
                assertEquals(-1f, result.overallScore, 0f)
            } finally { bitmap.recycle() }
        }
    }
    private fun wav(name: String, silence: Boolean): File {
        val samples = 64000
        val buffer = ByteBuffer.allocate(44 + samples * 2).order(ByteOrder.LITTLE_ENDIAN)
        buffer.put("RIFF".toByteArray(Charsets.US_ASCII)); buffer.putInt(36 + samples * 2)
        buffer.put("WAVEfmt ".toByteArray(Charsets.US_ASCII)); buffer.putInt(16); buffer.putShort(1); buffer.putShort(1)
        buffer.putInt(16000); buffer.putInt(32000); buffer.putShort(2); buffer.putShort(16)
        buffer.put("data".toByteArray(Charsets.US_ASCII)); buffer.putInt(samples * 2)
        for (i in 0 until samples) buffer.putShort(if (silence) 0 else (sin(2 * PI * 440 * i / 16000) * 8000).toInt().toShort())
        return File(context.cacheDir, name).apply { writeBytes(buffer.array()) }
    }
    @Test fun decoderRecordsSampleCoverageAndSilenceCannotBecomeAuthentic() {
        val silence = wav("onn-smoke-silence.wav", true)
        val tone = wav("onn-smoke-tone.wav", false)
        try {
            LiteRTInferenceEngine(MemoryMappedModelLoader(context)).use { engine ->
                val classifier = AudioClassifier(context, engine)
                val empty = classifier.analyze(Uri.fromFile(silence))
                assertEquals(AnalysisStatus.INCONCLUSIVE, empty.status)
                assertFalse(empty.isAiGenerated); assertEquals(-1f, empty.score, 0f)
                val diagnostic = classifier.analyze(Uri.fromFile(tone))
                assertEquals(AnalysisStatus.INCONCLUSIVE, diagnostic.status)
                assertFalse(diagnostic.isAiGenerated); assertFalse(diagnostic.modelValidated)
                assertTrue(diagnostic.reason, diagnostic.score.isFinite() && diagnostic.score in 0f..1f)
                assertTrue(diagnostic.analyzedDurationMs in 3500..4000)
                assertTrue(diagnostic.totalDurationMs >= diagnostic.analyzedDurationMs)
            }
        } finally { silence.delete(); tone.delete() }
    }
}
