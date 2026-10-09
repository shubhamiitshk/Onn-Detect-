package com.onndetect

import com.onndetect.inference.ModelIntegrityChecker
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.FileNotFoundException

class ModelIntegrityCheckerTest {
    private val image = ModelIntegrityChecker.IMAGE_MODEL
    private val audio = ModelIntegrityChecker.AUDIO_MODEL
    private fun hash(bytes: ByteArray) = ModelIntegrityChecker.computeSHA256(ByteArrayInputStream(bytes))
    private fun fixture(): MutableMap<String, ByteArray> {
        val models = mapOf("models/$image" to byteArrayOf(1, 2, 3), "models/$audio" to byteArrayOf(4, 5, 6))
        val assets = models.toMutableMap()
        val hashes = JSONObject().put("algorithm", "sha256").put("models", JSONObject().put(image, hash(models.getValue("models/$image"))).put(audio, hash(models.getValue("models/$audio"))))
        assets["constants/model_hashes.json"] = hashes.toString().toByteArray()
        assets["constants/meta_classifier.json"] = "{\"decision_threshold\":0.6}".toByteArray()
        assets["constants/natural_diff_histogram.json"] = "{}".toByteArray()
        val contract = JSONObject().put("sha256", hash(models.getValue("models/$image"))).put("dtype", "FLOAT32")
            .put("input_shape", org.json.JSONArray("[1,32,32,3]")).put("output_shape", org.json.JSONArray("[1,1]"))
        val artifactHashes = JSONObject().put("constants/meta_classifier.json", hash(assets.getValue("constants/meta_classifier.json")))
            .put("constants/natural_diff_histogram.json", hash(assets.getValue("constants/natural_diff_histogram.json")))
        assets["constants/release_manifest.json"] = JSONObject().put("schema_version", 1).put("release_id", "test-unvalidated")
            .put("score_convention", "real=0_fake=1").put("models", JSONObject().put(image, contract)).put("artifacts", artifactHashes)
            .put("validated", false).put("validation_status", "unvalidated").toString().toByteArray()
        return assets
    }
    private fun report(assets: Map<String, ByteArray>) = ModelIntegrityChecker.verifyModelsDetailed(setOf(image)) { path ->
        ByteArrayInputStream(assets[path] ?: throw FileNotFoundException(path))
    }
    /** Contract fixtures exercise evidence linkage, not detector accuracy. */
    private fun validatedAudioFixture(): MutableMap<String, ByteArray> {
        val assets = fixture()
        val frontend = com.onndetect.detection.audio.LFCCExtractor.FRONTEND_ID
        val audioHash = hash(assets.getValue("models/$audio"))
        val sourceCommit = "b".repeat(40)
        val releaseId = "test-validated"
        assets["constants/lcnn_audio_threshold.json"] = "{\"threshold_at_eer\":0.4}".toByteArray()
        assets["source_snapshot.json"] = JSONObject().put("source_commit", sourceCommit).put("source_dirty", true)
            .put("files", JSONObject().put("ml/src/model.py", "c".repeat(64))).toString().toByteArray()
        assets["fusion/XGBoostMetaClassifier.kt"] = "companion object { const val RELEASE_ID = \"$releaseId\" }".toByteArray()
        val sourceHash = hash(assets.getValue("source_snapshot.json"))
        val fusionHash = hash(assets.getValue("fusion/XGBoostMetaClassifier.kt"))
        assets["evaluation_report.json"] = JSONObject().put("release_id", releaseId).put("source_snapshot_sha256", sourceHash)
            .put("models", JSONObject().put(audio, JSONObject().put("parity", JSONObject().put("passed", true))
                .put("calibration", JSONObject().put("model_hash", audioHash).put("frontend_id", frontend)))).toString().toByteArray()
        assets["device_evaluation.json"] = JSONObject().put("release_id", releaseId).put("passed", true)
            .put("kotlin_parity_passed", true).put("offline_verified", true).put("fusion_revision", releaseId)
            .put("source_snapshot_sha256", sourceHash).put("fusion_kotlin_sha256", fusionHash)
            .put("models", JSONObject().put(audio, JSONObject().put("sha256", audioHash).put("frontend_id", frontend).put("parity_passed", true)))
            .toString().toByteArray()
        val release = JSONObject(String(assets.getValue("constants/release_manifest.json")))
            .put("release_id", releaseId).put("validated", true).put("source_commit", sourceCommit)
            .put("source_snapshot_sha256", sourceHash).put("fusion_revision", releaseId)
            .put("data_manifest_hashes", JSONObject().put("audio", "a".repeat(64)))
            .put("evaluation_report_hash", hash(assets.getValue("evaluation_report.json")))
            .put("device_validation", JSONObject().put("passed", true).put("report_hash", hash(assets.getValue("device_evaluation.json"))))
            .put("calibration", JSONObject().put(audio, JSONObject().put("model_hash", audioHash).put("frontend_id", frontend)
                .put("threshold", 0.4).put("auc", 0.8).put("n_calibration", 100).put("split_manifest_hash", "a".repeat(64))))
        release.getJSONObject("models").put(audio, JSONObject().put("sha256", audioHash).put("dtype", "FLOAT32")
            .put("input_shape", org.json.JSONArray("[1,401,40,1]")).put("output_shape", org.json.JSONArray("[1,1]")).put("frontend_id", frontend))
        val artifacts = release.getJSONObject("artifacts")
        for (path in listOf("source_snapshot.json", "fusion/XGBoostMetaClassifier.kt", "evaluation_report.json", "device_evaluation.json", "constants/lcnn_audio_threshold.json")) {
            artifacts.put(path, hash(assets.getValue(path)))
        }
        assets["constants/release_manifest.json"] = release.toString().toByteArray()
        return assets
    }
    private fun audioReport(assets: Map<String, ByteArray>) = ModelIntegrityChecker.verifyModelsDetailed(setOf(audio)) { path ->
        ByteArrayInputStream(assets[path] ?: throw FileNotFoundException(path))
    }
    private fun replaceDeviceEvidence(assets: MutableMap<String, ByteArray>, field: String, value: String) {
        val device = JSONObject(String(assets.getValue("device_evaluation.json"))).put(field, value)
        assets["device_evaluation.json"] = device.toString().toByteArray()
        val digest = hash(assets.getValue("device_evaluation.json"))
        val manifest = JSONObject(String(assets.getValue("constants/release_manifest.json")))
        manifest.getJSONObject("device_validation").put("report_hash", digest)
        manifest.getJSONObject("artifacts").put("device_evaluation.json", digest)
        assets["constants/release_manifest.json"] = manifest.toString().toByteArray()
    }
    @Test fun `legacy bytes may be intact but cannot provide validated verdict`() {
        val report = report(fixture())
        assertTrue(report.allOk); assertFalse(report.validated); assertNull(report.threshold)
    }
    @Test fun `missing malformed and empty hash manifests fail closed`() {
        val missing = fixture().apply { remove("constants/model_hashes.json") }
        assertFalse(report(missing).allOk)
        for (json in listOf("bad", "{}", "{\"algorithm\":\"sha256\",\"models\":{}}")) {
            assertFalse(report(fixture().apply { put("constants/model_hashes.json", json.toByteArray()) }).allOk)
        }
    }
    @Test fun `model and constant tampering fail closed`() {
        assertFalse(report(fixture().apply { put("models/$image", byteArrayOf(9)) }).allOk)
        assertFalse(report(fixture().apply { put("constants/meta_classifier.json", "{}".toByteArray()) }).allOk)
    }
    @Test fun `unrelated audio failure does not block image diagnostics`() {
        assertTrue(report(fixture().apply { remove("models/$audio") }).allOk)
    }
    @Test fun `claiming validation without lineage is rejected`() {
        val assets = fixture()
        val manifest = JSONObject(String(assets.getValue("constants/release_manifest.json"))).put("validated", true)
        assets["constants/release_manifest.json"] = manifest.toString().toByteArray()
        assertFalse(report(assets).allOk)
    }
    @Test fun `missing release fails closed`() {
        val report = report(fixture().apply { remove("constants/release_manifest.json") })
        assertFalse(report.allOk); assertFalse(report.validated)
    }
    @Test fun `string validation flags cannot unlock a release`() {
        val assets = fixture()
        val manifest = JSONObject(String(assets.getValue("constants/release_manifest.json"))).put("validated", "false")
        assets["constants/release_manifest.json"] = manifest.toString().toByteArray()
        assertFalse(report(assets).allOk)
    }
    @Test fun `string tensor dimensions are rejected`() {
        val assets = fixture()
        val manifest = JSONObject(String(assets.getValue("constants/release_manifest.json")))
        manifest.getJSONObject("models").getJSONObject(image).put("input_shape", org.json.JSONArray("[\"1\",32,32,3]"))
        assets["constants/release_manifest.json"] = manifest.toString().toByteArray()
        assertFalse(report(assets).allOk)
    }
    @Test fun `audio-only validation requires the exact source snapshot and fusion tested on device`() {
        val intact = audioReport(validatedAudioFixture())
        assertTrue(intact.failures.joinToString(), intact.allOk)
        assertTrue(intact.validated)
        for (field in listOf("source_snapshot_sha256", "fusion_kotlin_sha256")) {
            val assets = validatedAudioFixture()
            replaceDeviceEvidence(assets, field, "d".repeat(64))
            val failure = audioReport(assets)
            assertFalse("Mismatched $field must reject a reused device report", failure.allOk)
            assertFalse(failure.validated)
        }
    }
    @Test fun `tampered source or fusion bytes invalidate even an audio-only release`() {
        for (path in listOf("source_snapshot.json", "fusion/XGBoostMetaClassifier.kt")) {
            val assets = validatedAudioFixture()
            assets[path] = assets.getValue(path) + byteArrayOf(32)
            assertFalse("Unverified source bytes: $path", audioReport(assets).allOk)
        }
    }
    @Test fun `desktop parity cannot be reused for different model bytes or frontend`() {
        for (field in listOf("model_hash", "frontend_id")) {
            val assets = validatedAudioFixture()
            val evaluation = JSONObject(String(assets.getValue("evaluation_report.json")))
            evaluation.getJSONObject("models").getJSONObject(audio).getJSONObject("calibration").put(field, "unrelated")
            assets["evaluation_report.json"] = evaluation.toString().toByteArray()
            val digest = hash(assets.getValue("evaluation_report.json"))
            val manifest = JSONObject(String(assets.getValue("constants/release_manifest.json")))
            manifest.put("evaluation_report_hash", digest)
            manifest.getJSONObject("artifacts").put("evaluation_report.json", digest)
            assets["constants/release_manifest.json"] = manifest.toString().toByteArray()
            assertFalse("Desktop report linkage changed: $field", audioReport(assets).allOk)
        }
    }
}
