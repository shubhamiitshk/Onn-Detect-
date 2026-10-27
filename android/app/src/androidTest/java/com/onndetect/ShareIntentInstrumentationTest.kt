package com.onndetect

import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.onndetect.ui.share.SharedMediaIntents
import com.onndetect.ui.share.SharedMediaResolver
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The share-sheet hand-off, on a device.
 *
 * The rules themselves are JVM-tested in `SharedMediaResolverTest`; what
 * needs a device is the PLUMBING: that the platform really hands over
 * `EXTRA_STREAM` the way the adapter expects, that a `ClipData` hand-off is
 * read, and that the hand-off is consumed so a rotation does not re-run the
 * scan.
 *
 * The URIs here are well-formed `content://` URIs that point at nothing.
 * That is deliberate: this test covers the HAND-OFF, not the scan. The scan
 * itself is covered by `PackagedInferenceTest`, which runs the real
 * packaged models over a real image.
 *
 * This needs a device. It belongs to the same run as the device evaluation
 * that the release still owes.
 */
@RunWith(AndroidJUnit4::class)
class ShareIntentInstrumentationTest {

    private val imageUri = Uri.parse("content://media/external/images/media/42")
    private val audioUri = Uri.parse("content://media/external/audio/media/7")

    @Test fun aSingleImageShareIsResolvedToTheImageRoute() {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "image/png"
            putExtra(Intent.EXTRA_STREAM, imageUri)
        }
        val shared = SharedMediaIntents.consume(intent)
        assertNotNull("the share hand-off was not resolved", shared)
        assertEquals(SharedMediaResolver.MODE_IMAGE, shared!!.mode)
        assertEquals(imageUri.toString(), shared.uri)
    }

    @Test fun anAudioShareIsResolvedToTheAudioRoute() {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "audio/mp4"
            putExtra(Intent.EXTRA_STREAM, audioUri)
        }
        val shared = SharedMediaIntents.consume(intent)
        assertNotNull(shared)
        assertEquals(SharedMediaResolver.MODE_AUDIO, shared!!.mode)
    }

    @Test fun aMultipleShareIsReadFromTheClipData() {
        val intent = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
            type = "image/png"
            clipData = android.content.ClipData.newRawUri("shared", imageUri)
        }
        val shared = SharedMediaIntents.consume(intent)
        assertNotNull("the clip-data hand-off was not resolved", shared)
        assertEquals(imageUri.toString(), shared!!.uri)
    }

    @Test fun theHandoffIsConsumedSoARotationDoesNotRerunTheScan() {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "image/png"
            putExtra(Intent.EXTRA_STREAM, imageUri)
        }
        assertNotNull(SharedMediaIntents.consume(intent))
        // The SAME intent object is re-delivered after a configuration
        // change, so the second consume must find nothing.
        assertNull("the share hand-off was not consumed", SharedMediaIntents.consume(intent))
    }

    @Test fun anUnrelatedIntentIsNotAShare() {
        assertNull(SharedMediaIntents.consume(Intent(Intent.ACTION_MAIN)))
        assertNull(SharedMediaIntents.consume(null))
    }

    @Test fun theAppIsOfferedAsAShareTargetForImagesAudioAndVideo() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        for (type in listOf("image/jpeg", "audio/mp4", "video/mp4")) {
            val intent = Intent(Intent.ACTION_SEND).apply { this.type = type }
            val handlers = context.packageManager.queryIntentActivities(
                intent, PackageManager.MATCH_DEFAULT_ONLY,
            )
            val ours = handlers.filter { it.activityInfo.packageName == context.packageName }
            assertTrue(
                "the app is not offered as a share target for $type",
                ours.isNotEmpty(),
            )
            assertTrue(
                "the share target for $type is not MainActivity: " +
                    ours.map { it.activityInfo.name },
                ours.any { it.activityInfo.name.endsWith("MainActivity") },
            )
        }
    }

    @Test fun theAppIsNOTOfferedAsAShareTargetForPlainText() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(Intent.ACTION_SEND).apply { type = "text/plain" }
        val ours = context.packageManager
            .queryIntentActivities(intent, PackageManager.MATCH_DEFAULT_ONLY)
            .filter { it.activityInfo.packageName == context.packageName }
        assertTrue("the app must not claim to be a text/plain share target", ours.isEmpty())
    }
}
