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

    @Test fun `a single image share is resolved to the image route`() {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "image/png"
            putExtra(Intent.EXTRA_STREAM, imageUri)
        }
        val shared = SharedMediaIntents.consume(intent)
        assertNotNull("the share hand-off was not resolved", shared)
        assertEquals(SharedMediaResolver.MODE_IMAGE, shared!!.mode)
        assertEquals(imageUri.toString(), shared.uri)
    }

    @Test fun `an audio share is resolved to the audio route`() {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "audio/mp4"
            putExtra(Intent.EXTRA_STREAM, audioUri)
        }
        val shared = SharedMediaIntents.consume(intent)
        assertNotNull(shared)
        assertEquals(SharedMediaResolver.MODE_AUDIO, shared!!.mode)
    }

    @Test fun `a multiple share is read from the clip data`() {
        val intent = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
            type = "image/png"
            clipData = android.content.ClipData.newRawUri("shared", imageUri)
        }
        val shared = SharedMediaIntents.consume(intent)
        assertNotNull("the clip-data hand-off was not resolved", shared)
        assertEquals(imageUri.toString(), shared!!.uri)
    }

    @Test fun `the hand-off is consumed so a rotation does not re-run the scan`() {
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "image/png"
            putExtra(Intent.EXTRA_STREAM, imageUri)
        }
        assertNotNull(SharedMediaIntents.consume(intent))
        // The SAME intent object is re-delivered after a configuration
        // change, so the second consume must find nothing.
        assertNull("the share hand-off was not consumed", SharedMediaIntents.consume(intent))
    }

    @Test fun `an unrelated intent is not a share`() {
        assertNull(SharedMediaIntents.consume(Intent(Intent.ACTION_MAIN)))
        assertNull(SharedMediaIntents.consume(null))
    }

    @Test fun `the app declares itself a share target for images and audio`() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val info = context.packageManager.getPackageInfo(
            context.packageName, PackageManager.GET_ACTIVITIES,
        )
        val actions = (info.activities ?: emptyArray())
            .filter { it.name.endsWith("MainActivity") }
            .flatMap { activity -> activity.intentFilters }
            .flatMap { filter -> filter.actions.map { action -> action.name } }
        assertTrue(
            "the app is not registered as a share target: $actions",
            actions.contains(Intent.ACTION_SEND),
        )
    }
}
