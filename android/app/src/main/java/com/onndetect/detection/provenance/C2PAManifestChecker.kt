package com.onndetect.detection.provenance

import android.content.Context
import android.net.Uri
import javax.inject.Inject

class C2PAManifestChecker @Inject constructor() {

    /**
     * Lightweight Kotlin parser for Coalition for Content Provenance and Authenticity (C2PA).
     * Instead of running the heavy 15MB Rust `c2pa-rs` library via JNI which crashes low-end devices,
     * this performs a basic offline byte-scan for the "c2pa" JUMBF (JPEG Universal Metadata Box Format) URI.
     */
    fun analyze(context: Context, imageUri: Uri): Float {
        // 0.05 = the NEUTRAL_C2PA constant the meta-model was calibrated with
        // (see DetectionPipeline.Companion and feature_extractor.py). The
        // previous 0.5 "neutral" value was out-of-distribution for slot 7.
        var c2paScore = 0.05f

        try {
            context.contentResolver.openInputStream(imageUri)?.use { inputStream ->
                // Read up to the first 64KB fully - a single read() may return
                // fewer bytes than available and miss the JUMBF marker.
                val buffer = ByteArray(65536)
                var offset = 0
                while (offset < buffer.size) {
                    val n = inputStream.read(buffer, offset, buffer.size - offset)
                    if (n <= 0) break
                    offset += n
                }

                if (offset > 0) {
                    val headerString = String(buffer, 0, offset, Charsets.US_ASCII)

                    // Look for the C2PA JUMBF marker: "http://c2pa.org/..."
                    if (headerString.contains("c2pa.org")) {
                        // A C2PA manifest exists!
                        // In a full implementation, we would cryptographically verify the signature here.
                        // For now, presence of C2PA usually indicates AI Generation (DALL-E 3, Firefly)
                        // or a heavily edited image from a modern workflow.
                        c2paScore = 0.8f
                    }
                }
            }
        } catch (e: Exception) {
            c2paScore = 0.05f
        }

        return c2paScore
    }
}
