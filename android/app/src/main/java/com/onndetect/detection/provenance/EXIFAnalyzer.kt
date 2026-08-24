package com.onndetect.detection.provenance

import android.content.Context
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import javax.inject.Inject

class EXIFAnalyzer @Inject constructor() {

    fun analyze(context: Context, imageUri: Uri): Float {
        var anomalyScore = 0.0f

        try {
            // use{} guarantees the stream is closed even if ExifInterface
            // parsing throws mid-way (the previous manual close() was skipped
            // on any exception path, leaking a content-provider fd).
            context.contentResolver.openInputStream(imageUri)?.use { inputStream ->
                val exif = ExifInterface(inputStream)

                // 1. Check Software Tag for known manipulation tools
                val software = exif.getAttribute(ExifInterface.TAG_SOFTWARE)?.lowercase() ?: ""
                val knownManipulators = listOf("photoshop", "gimp", "midjourney", "dall-e", "stable diffusion", "lightroom", "canva")
                if (knownManipulators.any { software.contains(it) }) {
                    anomalyScore += 0.4f
                }

                // 2. Check for missing DateTimeOriginal vs DateTimeDigitized
                val dtOriginal = exif.getAttribute(ExifInterface.TAG_DATETIME_ORIGINAL)
                val dtDigitized = exif.getAttribute(ExifInterface.TAG_DATETIME_DIGITIZED)

                // Natural photos from cameras typically have both.
                // AI generated or heavily scrubbed photos often miss one or both.
                if (dtOriginal == null && dtDigitized != null) {
                    anomalyScore += 0.2f
                } else if (dtOriginal == null && dtDigitized == null) {
                    anomalyScore += 0.1f // Very common in social media, so mild penalty
                }

                // 3. Check for unusual resolutions (e.g. perfect 1024x1024 often indicates AI)
                val width = exif.getAttributeInt(ExifInterface.TAG_IMAGE_WIDTH, -1)
                val length = exif.getAttributeInt(ExifInterface.TAG_IMAGE_LENGTH, -1)
                if ((width == 512 && length == 512) || (width == 1024 && length == 1024)) {
                    anomalyScore += 0.3f
                }
            }
        } catch (e: Exception) {
            // Missing EXIF data entirely is suspicious but also common for PNGs or web downloads
            anomalyScore += 0.1f
        }

        return anomalyScore.coerceIn(0.0f, 1.0f)
    }
}
