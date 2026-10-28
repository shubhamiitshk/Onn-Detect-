package com.onndetect.detection.visual

import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import android.os.Build
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream

/**
 * Manages discovery, hardware profiling, Google Drive integration, and 1-tap local
 * zip/pack importation for optional on-device Vision-Language Model (VLM) feature packs.
 *
 * Strict Privacy Guarantee: All file handling occurs locally via ContentResolver.
 * No internet socket permissions are held or requested.
 */
object VlmPackManager {

    data class DeviceHardwareProfile(
        val totalRamGb: Float,
        val availableRamGb: Float,
        val abi: String,
        val recommendedPresetId: String,
    )

    data class VlmCatalogEntry(
        val id: String,
        val name: String,
        val description: String,
        val minRamGb: Float,
        val recommendedRamGb: Float,
        val downloadSizeStr: String,
        val defaultDriveUrl: String,
        val isRecommended: Boolean,
    )

    data class InstalledVlmPackDetails(
        val packId: String,
        val modelId: String,
        val totalSizeBytes: Long,
        val languageFileName: String,
        val visionFileName: String,
        val publisherKeyHex: String,
        val isOfficialPublisher: Boolean,
        val diagnosticOnly: Boolean,
    )

    // Pre-configured public Google Drive / repository links
    const val DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1_OnnDetect_Forensic_VLM_Packs"
    const val QWEN_08B_DRIVE_URL = "https://drive.google.com/file/d/1fXMpdUyDywQ7mjunDfK5LupzAYgBX25s/view?usp=sharing"
    const val SMOLVLM_500M_DRIVE_URL = "https://drive.google.com/uc?export=download&id=1_SmolVLM2_500M_Pack"
    const val QWEN_2B_DRIVE_URL = "https://drive.google.com/uc?export=download&id=1_Qwen35_2B_Pack"

    fun inspectDevice(context: Context): DeviceHardwareProfile {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(memInfo)
        val totalGb = memInfo.totalMem.toFloat() / (1024f * 1024f * 1024f)
        val availGb = memInfo.availMem.toFloat() / (1024f * 1024f * 1024f)
        val abi = Build.SUPPORTED_ABIS.firstOrNull() ?: "arm64-v8a"

        val recommendedPreset = when {
            totalGb >= 7.5f -> "qwen3.5-2b"
            totalGb >= 3.8f -> "qwen3.5-0.8b"
            else -> "smolvlm2-500m"
        }

        return DeviceHardwareProfile(
            totalRamGb = (Math.round(totalGb * 10f) / 10f),
            availableRamGb = (Math.round(availGb * 10f) / 10f),
            abi = abi,
            recommendedPresetId = recommendedPreset,
        )
    }

    fun getCatalog(profile: DeviceHardwareProfile): List<VlmCatalogEntry> {
        return listOf(
            VlmCatalogEntry(
                id = "qwen3.5-0.8b",
                name = "Qwen 3.5 0.8B Forensic Pack",
                description = "Balanced 4-bit quantized architecture. Optimal speed and dialectical reasoning for mid-range and modern phones.",
                minRamGb = 4.0f,
                recommendedRamGb = 6.0f,
                downloadSizeStr = "~550 MB",
                defaultDriveUrl = QWEN_08B_DRIVE_URL,
                isRecommended = profile.recommendedPresetId == "qwen3.5-0.8b",
            ),
            VlmCatalogEntry(
                id = "smolvlm2-500m",
                name = "SmolVLM2 500M Ultra-Lightweight",
                description = "Compact multimodal architecture for budget devices or minimum RAM footprint. Fast prefill with low battery draw.",
                minRamGb = 3.0f,
                recommendedRamGb = 4.0f,
                downloadSizeStr = "~490 MB",
                defaultDriveUrl = SMOLVLM_500M_DRIVE_URL,
                isRecommended = profile.recommendedPresetId == "smolvlm2-500m",
            ),
            VlmCatalogEntry(
                id = "qwen3.5-2b",
                name = "Qwen 3.5 2B High-Precision",
                description = "Deep 2-billion parameter multimodal engine for high-end flagship devices. Maximum nuance on subtle AI synthesis.",
                minRamGb = 7.0f,
                recommendedRamGb = 8.0f,
                downloadSizeStr = "~1.4 GB",
                defaultDriveUrl = QWEN_2B_DRIVE_URL,
                isRecommended = profile.recommendedPresetId == "qwen3.5-2b",
            ),
        )
    }

    fun getInstalledPackDetails(context: Context): InstalledVlmPackDetails? {
        val packDir = File(context.filesDir, "vlm_packs")
        if (!packDir.isDirectory) return null

        val manifestFile = File(packDir, "signed_manifest.json").takeIf { it.isFile }
            ?: File(packDir, "manifest.json").takeIf { it.isFile }
            ?: return null

        return runCatching {
            val manifest = OnDemandForensicVlmAnalyzer.SignedVlmPackManifest.fromJson(
                manifestFile.readText(Charsets.UTF_8)
            )
            val policyFile = File(packDir, "forensic_vlm_policy.json")
            val policyBundle = if (policyFile.isFile) {
                OnDemandForensicVlmAnalyzer.PolicyBundle.fromJson(policyFile.readText(Charsets.UTF_8))
            } else null

            var totalBytes = 0L
            packDir.listFiles()?.forEach { file ->
                if (file.isFile) totalBytes += file.length()
            }

            val langFile = packDir.listFiles()?.firstOrNull {
                it.isFile && it.extension == "gguf" && !it.name.contains("mmproj")
            }?.name ?: "${manifest.packId}-model.gguf"

            val visFile = packDir.listFiles()?.firstOrNull {
                it.isFile && it.extension == "gguf" && it.name.contains("mmproj")
            }?.name ?: "${manifest.packId}-mmproj.gguf"

            val isOfficial = manifest.publisherPublicKeyHex.equals(
                OnDemandForensicVlmAnalyzer.OFFICIAL_PUBLISHER_PUBLIC_KEY_HEX,
                ignoreCase = true
            )

            InstalledVlmPackDetails(
                packId = manifest.packId,
                modelId = policyBundle?.modelId ?: manifest.hfModelId,
                totalSizeBytes = totalBytes,
                languageFileName = langFile,
                visionFileName = visFile,
                publisherKeyHex = manifest.publisherPublicKeyHex,
                isOfficialPublisher = isOfficial,
                diagnosticOnly = policyBundle?.diagnosticOnly ?: manifest.diagnosticOnly,
            )
        }.getOrNull()
    }

    suspend fun importPackZip(
        context: Context,
        zipUri: Uri,
        allowCustomKey: Boolean = false,
        onProgress: (String) -> Unit = {},
    ): Result<InstalledVlmPackDetails> = withContext(Dispatchers.IO) {
        runCatching {
            onProgress("Opening pack archive...")
            val stagingDir = File(context.cacheDir, "vlm_import_staging_${System.currentTimeMillis()}")
            if (stagingDir.exists()) stagingDir.deleteRecursively()
            stagingDir.mkdirs()

            try {
                // 1. Stream & Extract ZIP archive with Zip-Slip path traversal defense
                onProgress("Extracting archive files to secure sandbox...")
                context.contentResolver.openInputStream(zipUri)?.use { input ->
                    extractZipStream(input, stagingDir)
                } ?: throw IllegalArgumentException("Cannot open stream for selected URI: $zipUri")

                // If contents are nested under a single directory, flatten to root
                val resolvedDir = resolveNestedPackDir(stagingDir)

                onProgress("Verifying cryptographic signature and file hashes...")
                val manifestFile = File(resolvedDir, "signed_manifest.json").takeIf { it.isFile }
                    ?: File(resolvedDir, "manifest.json").takeIf { it.isFile }
                    ?: throw IllegalArgumentException("Pack archive does not contain signed_manifest.json")

                val manifest = OnDemandForensicVlmAnalyzer.SignedVlmPackManifest.fromJson(
                    manifestFile.readText(Charsets.UTF_8)
                )

                // Verify contract & hashes
                val verified = OnDemandForensicVlmAnalyzer.verifySignedPackContract(
                    manifest = manifest,
                    expectedPublisherPublicKeyHex = if (allowCustomKey) null else OnDemandForensicVlmAnalyzer.OFFICIAL_PUBLISHER_PUBLIC_KEY_HEX,
                    allowUntrustedPublisherForTesting = allowCustomKey,
                )
                if (!verified) {
                    throw IllegalStateException("Cryptographic verification failed. Check publisher signature or toggle Developer Mode.")
                }

                // Verify policy
                val policyFile = File(resolvedDir, "forensic_vlm_policy.json")
                if (!policyFile.isFile) throw IllegalArgumentException("Missing forensic_vlm_policy.json")
                val policySha = OnDemandForensicVlmAnalyzer.computeFileSha256(policyFile)
                if (!policySha.equals(manifest.policySha256, ignoreCase = true)) {
                    throw IllegalStateException("Policy SHA-256 hash mismatch! Possible file corruption.")
                }

                onProgress("Validating model weights on disk...")
                val langFiles = resolvedDir.listFiles()?.filter {
                    it.isFile && it.extension == "gguf" && !it.name.contains("mmproj")
                } ?: emptyList()
                val langFile = langFiles.firstOrNull { it.length() == manifest.languageBytes }
                    ?: throw IllegalArgumentException("Language GGUF missing or size mismatch (expected ${manifest.languageBytes} bytes)")

                val visFiles = resolvedDir.listFiles()?.filter {
                    it.isFile && it.extension == "gguf" && it.name.contains("mmproj")
                } ?: emptyList()
                val visFile = visFiles.firstOrNull { it.length() == manifest.visionMmprojBytes }
                    ?: throw IllegalArgumentException("Vision mmproj GGUF missing or size mismatch (expected ${manifest.visionMmprojBytes} bytes)")

                // SHA-256 weight verification
                onProgress("Verifying SHA-256 checksums...")
                val actualLangSha = OnDemandForensicVlmAnalyzer.computeFileSha256(langFile)
                if (!actualLangSha.equals(manifest.languageWeightsSha256, ignoreCase = true)) {
                    throw IllegalStateException("Language model SHA-256 hash mismatch!")
                }
                val actualVisSha = OnDemandForensicVlmAnalyzer.computeFileSha256(visFile)
                if (!actualVisSha.equals(manifest.visionMmprojSha256, ignoreCase = true)) {
                    throw IllegalStateException("Vision mmproj SHA-256 hash mismatch!")
                }

                onProgress("Installing verified pack...")
                val targetDir = File(context.filesDir, "vlm_packs")
                // Reset any existing native backend session
                (OnDemandForensicVlmAnalyzer.activePrefillBackend as? AutoCloseable)?.close()
                OnDemandForensicVlmAnalyzer.activePrefillBackend = null

                if (targetDir.exists()) {
                    targetDir.deleteRecursively()
                }
                targetDir.mkdirs()

                // Copy verified files into filesDir/vlm_packs
                resolvedDir.listFiles()?.forEach { file ->
                    if (file.isFile) {
                        file.copyTo(File(targetDir, file.name), overwrite = true)
                    }
                }

                // Enable cascade in analyzer
                OnDemandForensicVlmAnalyzer.experimentalVlmCascadeEnabled = true
                OnDemandForensicVlmAnalyzer.allowUntrustedPublisherForTesting = allowCustomKey

                onProgress("Pack activated successfully!")
                getInstalledPackDetails(context)
                    ?: throw IllegalStateException("Failed to read installed pack details after copy")
            } finally {
                stagingDir.deleteRecursively()
            }
        }
    }

    fun uninstallPack(context: Context): Boolean {
        return runCatching {
            (OnDemandForensicVlmAnalyzer.activePrefillBackend as? AutoCloseable)?.close()
            OnDemandForensicVlmAnalyzer.activePrefillBackend = null
            OnDemandForensicVlmAnalyzer.experimentalVlmCascadeEnabled = false

            val packDir = File(context.filesDir, "vlm_packs")
            if (packDir.exists()) {
                packDir.deleteRecursively()
            } else true
        }.getOrDefault(false)
    }

    private fun extractZipStream(input: InputStream, outputDir: File) {
        val canonicalDest = outputDir.canonicalPath + File.separator
        ZipInputStream(input).use { zis ->
            var entry: ZipEntry? = zis.nextEntry
            val buffer = ByteArray(64 * 1024)
            while (entry != null) {
                val file = File(outputDir, entry.name)
                // Zip-Slip security defense: reject any entry attempting path traversal outside outputDir
                if (!file.canonicalPath.startsWith(canonicalDest)) {
                    throw SecurityException("Zip entry is outside of target dir: ${entry.name}")
                }
                if (entry.isDirectory) {
                    file.mkdirs()
                } else {
                    file.parentFile?.mkdirs()
                    FileOutputStream(file).use { fos ->
                        var len: Int
                        while (zis.read(buffer).also { len = it } > 0) {
                            fos.write(buffer, 0, len)
                        }
                    }
                }
                zis.closeEntry()
                entry = zis.nextEntry
            }
        }
    }

    private fun resolveNestedPackDir(dir: File): File {
        val files = dir.listFiles() ?: return dir
        // If the zip had a single root folder containing the pack
        if (files.size == 1 && files[0].isDirectory) {
            val subFiles = files[0].listFiles() ?: emptyArray()
            if (subFiles.any { it.name == "signed_manifest.json" || it.name == "manifest.json" }) {
                return files[0]
            }
        }
        return dir
    }
}
