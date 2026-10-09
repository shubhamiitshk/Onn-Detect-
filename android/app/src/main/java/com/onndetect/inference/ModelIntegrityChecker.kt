package com.onndetect.inference

import android.content.Context
import org.json.JSONObject
import java.security.MessageDigest

/** Byte integrity is distinct from a calibrated, evaluated detector release. */
object ModelIntegrityChecker {
    const val IMAGE_MODEL = "srm_forensic_cnn_int8.tflite"
    const val AUDIO_MODEL = "lcnn_audio_int8.tflite"
    val REQUIRED_MODELS = setOf(IMAGE_MODEL, AUDIO_MODEL)
    data class Report(val allOk: Boolean, val failures: List<String>, val validated: Boolean = false,
        val validationReason: String = "Release lineage/calibration is unavailable", val releaseId: String = "unknown",
        val frontendId: String = "unknown", val threshold: Float? = null)

    fun verifyAllModels(context: Context): Boolean = verifyAllModelsDetailed(context).allOk
    fun verifyAllModelsDetailed(context: Context): Report = verifyModelsDetailed(context, REQUIRED_MODELS)

    fun verifyModelsDetailed(context: Context, required: Set<String>): Report =
        verifyModelsDetailed(required) { path -> context.assets.open(path) }

    /** Reader injection permits fail-closed manifest tests without Android stubs. */
    fun verifyModelsDetailed(required: Set<String>, openAsset: (String) -> java.io.InputStream): Report {
        require(required.isNotEmpty() && REQUIRED_MODELS.containsAll(required))
        val failures = mutableListOf<String>()
        fun read(path: String) = JSONObject(openAsset(path).bufferedReader().use { it.readText() })
        fun numeric(json: JSONObject, key: String): Double {
            val number = json.get(key) as? Number ?: error("$key must be a JSON number")
            return number.toDouble().also { require(it.isFinite()) { "$key must be finite" } }
        }
        fun integer(json: JSONObject, key: String): Int {
            val n = numeric(json, key)
            require(n == n.toInt().toDouble()) { "$key must be an integer" }
            return n.toInt()
        }
        fun positiveShape(array: org.json.JSONArray) {
            require((0 until array.length()).all {
                val number = array.get(it) as? Number
                number != null && number.toDouble().isFinite() && number.toDouble() == number.toInt().toDouble() && number.toInt() > 0
            }) { "Tensor shape must contain positive integers" }
        }
        val hashes = try { read("constants/model_hashes.json") } catch (_: Exception) {
            return Report(false, listOf("Missing or malformed model hash manifest"))
        }
        try {
            require(hashes.getString("algorithm") == "sha256")
            val models = hashes.getJSONObject("models")
            require(REQUIRED_MODELS.all { models.has(it) }) { "Required model missing from manifest" }
            required.forEach { name ->
                val expected = models.getString(name)
                require(validHash(expected)) { "Malformed model hash" }
                val actual = openAsset("models/$name").use(::computeSHA256)
                require(actual.equals(expected, true)) { "$name: hash mismatch" }
            }
        } catch (e: Exception) { failures += e.message ?: "Model integrity failure" }
        if (failures.isNotEmpty()) return Report(false, failures)

        val release = try { read("constants/release_manifest.json") } catch (_: Exception) {
            return Report(false, listOf("Missing or malformed detector release manifest"))
        }
        return try {
            require(integer(release, "schema_version") == 1)
            val releaseId = release.getString("release_id").also { require(it.isNotBlank()) }
            require(release.getString("score_convention") == "real=0_fake=1")
            val models = release.getJSONObject("models")
            required.forEach { name ->
                val model = models.getJSONObject(name)
                require(model.getString("sha256").equals(hashes.getJSONObject("models").getString(name), true))
                require(model.getString("dtype") == "FLOAT32")
                val shape = model.getJSONArray("input_shape")
                positiveShape(shape)
                require(shape.length() == 4 && shape.getInt(0) == 1)
                val out = model.getJSONArray("output_shape")
                positiveShape(out)
                require(out.length() == 2 && out.getInt(0) == 1 && out.getInt(1) == 1)
            }
            val artifacts = release.getJSONObject("artifacts")
            val requiredArtifacts = if (IMAGE_MODEL in required) listOf("constants/meta_classifier.json", "constants/natural_diff_histogram.json") else emptyList()
            (requiredArtifacts + if (AUDIO_MODEL in required) listOf("constants/lcnn_audio_threshold.json") else emptyList()).forEach { path ->
                val expected = artifacts.getString(path)
                require(validHash(expected))
                require(openAsset(path).use(::computeSHA256).equals(expected, true)) { "$path: hash mismatch" }
            }
            require(release.get("validated") is Boolean) { "validated must be a Boolean" }
            if (release.get("validated") != true) return Report(true, emptyList(), false,
                release.optString("validation_status", "Unvalidated legacy detector"), releaseId)
            require(release.getString("source_commit").matches(Regex("[a-fA-F0-9]{40}")))
            val dataHashes = release.getJSONObject("data_manifest_hashes")
            require(dataHashes.length() > 0 && dataHashes.keys().asSequence().all { validHash(dataHashes.getString(it)) })
            require(validHash(release.getString("evaluation_report_hash")))
            fun boundArtifact(path: String, digest: String) {
                require(validHash(digest) && artifacts.getString(path) == digest) { "$path: artifact linkage mismatch" }
                require(openAsset(path).use(::computeSHA256) == digest) { "$path: artifact hash mismatch" }
            }
            fun boundReport(path: String, digest: String): JSONObject {
                boundArtifact(path, digest)
                return read(path).also { require(it.getString("release_id") == releaseId) { "Report release ID mismatch" } }
            }
            val sourceSnapshotHash = release.getString("source_snapshot_sha256")
            boundArtifact("source_snapshot.json", sourceSnapshotHash)
            val sourceSnapshot = read("source_snapshot.json")
            require(sourceSnapshot.getString("source_commit") == release.getString("source_commit") && sourceSnapshot.get("source_dirty") is Boolean) { "Source snapshot identity mismatch" }
            val sourceFiles = sourceSnapshot.getJSONObject("files")
            require(sourceFiles.length() > 0 && sourceFiles.keys().asSequence().all { validHash(sourceFiles.getString(it)) }) { "Source snapshot file hashes missing" }
            val evaluation = boundReport("evaluation_report.json", release.getString("evaluation_report_hash"))
            require(evaluation.getString("source_snapshot_sha256") == sourceSnapshotHash) { "Desktop evaluation source snapshot mismatch" }
            val deviceValidation = release.getJSONObject("device_validation")
            require(deviceValidation.get("passed") == true) { "Device validation missing" }
            val device = boundReport("device_evaluation.json", deviceValidation.getString("report_hash"))
            require(device.get("passed") == true && device.get("kotlin_parity_passed") == true && device.get("offline_verified") == true) { "Device/offline validation failed" }
            require(device.getString("fusion_revision") == release.getString("fusion_revision"))
            require(device.getString("source_snapshot_sha256") == sourceSnapshotHash) { "Device evaluation source snapshot mismatch" }
            val fusionKotlinHash = artifacts.getString("fusion/XGBoostMetaClassifier.kt")
            boundArtifact("fusion/XGBoostMetaClassifier.kt", fusionKotlinHash)
            require(device.getString("fusion_kotlin_sha256") == fusionKotlinHash) { "Device evaluation fusion source mismatch" }
            required.forEach { name ->
                val model = models.getJSONObject(name)
                val desktop = evaluation.getJSONObject("models").getJSONObject(name)
                require(desktop.getJSONObject("parity").get("passed") == true) { "Desktop model parity missing" }
                val desktopCalibration = desktop.getJSONObject("calibration")
                require(desktopCalibration.getString("model_hash") == model.getString("sha256") &&
                    desktopCalibration.getString("frontend_id") == model.getString("frontend_id")) { "Desktop model/frontend linkage mismatch" }
                val tested = device.getJSONObject("models").getJSONObject(name)
                require(tested.getString("sha256") == model.getString("sha256") && tested.getString("frontend_id") == model.getString("frontend_id") && tested.get("parity_passed") == true) { "Device model/frontend linkage mismatch" }
            }
            if (IMAGE_MODEL in required) require(release.getString("fusion_revision") == com.onndetect.detection.fusion.XGBoostMetaClassifier.RELEASE_ID &&
                com.onndetect.detection.fusion.XGBoostMetaClassifier.RELEASE_ID != "legacy-unvalidated") { "Fusion lineage mismatch" }
            val calibrations = release.getJSONObject("calibration")
            var threshold: Float? = null
            var frontend = "unknown"
            required.forEach { name ->
                val model = models.getJSONObject(name)
                val c = calibrations.getJSONObject(name)
                require(c.getString("model_hash") == model.getString("sha256"))
                frontend = model.getString("frontend_id")
                require(c.getString("frontend_id") == frontend)
                val runtimeFrontend = if (name == AUDIO_MODEL) com.onndetect.detection.audio.LFCCExtractor.FRONTEND_ID else
                    "rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-size${model.getJSONArray("input_shape").getInt(1)}-float32"
                require(frontend == runtimeFrontend) { "Frontend contract mismatch" }
                if (name == AUDIO_MODEL) require(model.getJSONArray("input_shape").toString() == "[1,401,40,1]")
                else require(model.getJSONArray("input_shape").getInt(1) == model.getJSONArray("input_shape").getInt(2) && model.getJSONArray("input_shape").getInt(3) == 3)
                val t = numeric(c, "threshold").toFloat()
                val auc = numeric(c, "auc")
                require(t.isFinite() && t in 0f..1f && auc.isFinite() && auc > 0.5 && auc <= 1.0)
                require(integer(c, "n_calibration") >= 100 && validHash(c.getString("split_manifest_hash")))
                require(dataHashes.keys().asSequence().any { dataHashes.getString(it) == c.getString("split_manifest_hash") }) { "Calibration split lineage mismatch" }
                val constantThreshold = if (name == AUDIO_MODEL) numeric(read("constants/lcnn_audio_threshold.json"), "threshold_at_eer").toFloat() else numeric(read("constants/meta_classifier.json"), "decision_threshold").toFloat()
                require(kotlin.math.abs(t - constantThreshold) < 1e-6f) { "Calibration asset/release threshold mismatch" }
                threshold = t
            }
            Report(true, emptyList(), true, "Validated release", releaseId, frontend, threshold)
        } catch (e: Exception) { Report(false, listOf("Release contract invalid: ${e.message}")) }
    }

    private fun validHash(value: String) = value.matches(Regex("[a-fA-F0-9]{64}"))
    fun computeSHA256(inputStream: java.io.InputStream): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(8192)
        while (true) { val n = inputStream.read(buffer); if (n < 0) break; digest.update(buffer, 0, n) }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
