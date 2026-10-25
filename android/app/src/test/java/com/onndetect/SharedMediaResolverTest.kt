package com.onndetect

import com.onndetect.ui.share.SharedMediaResolver
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The share-sheet hand-off rules.
 *
 * A share sheet is an UNTRUSTED entry point: any app can hand this one a
 * URI of any shape. These tests pin the three things that matter - that the
 * mode is right, that only a `content://` URI is accepted, and that a
 * multiple share does not silently imply that every item was checked.
 */
class SharedMediaResolverTest {

    @Test fun `an image share resolves to the image route`() {
        val shared = SharedMediaResolver.resolve(
            streamUri = "content://media/external/images/media/42",
            clipUris = emptyList(),
            mimeType = "image/jpeg",
        )
        assertEquals("content://media/external/images/media/42", shared?.uri)
        assertEquals(SharedMediaResolver.MODE_IMAGE, shared?.mode)
    }

    @Test fun `an audio share resolves to the audio route`() {
        val shared = SharedMediaResolver.resolve(
            streamUri = "content://media/external/audio/media/7",
            clipUris = emptyList(),
            mimeType = "audio/mp4",
        )
        assertEquals(SharedMediaResolver.MODE_AUDIO, shared?.mode)
    }

    @Test fun `a video share resolves to the video route`() {
        val shared = SharedMediaResolver.resolve(
            streamUri = "content://media/external/video/media/99",
            clipUris = emptyList(),
            mimeType = "video/mp4",
        )
        assertEquals("content://media/external/video/media/99", shared?.uri)
        assertEquals(SharedMediaResolver.MODE_VIDEO, shared?.mode)
    }

    @Test fun `a file uri is refused`() {
        // A `file://` hand-off would let another app name a path in this
        // app's private storage.
        assertFalse(SharedMediaResolver.isSafeContentUri("file:///data/data/com.onndetect/files/x"))
        assertNull(
            SharedMediaResolver.resolve("file:///data/data/com.onndetect/files/x", emptyList(), "image/jpeg"),
        )
    }

    @Test fun `a network uri is refused`() {
        // The scanner must never become a network fetcher.
        assertFalse(SharedMediaResolver.isSafeContentUri("https://example.com/a.jpg"))
        assertFalse(SharedMediaResolver.isSafeContentUri("http://example.com/a.jpg"))
        assertNull(
            SharedMediaResolver.resolve("https://example.com/a.jpg", emptyList(), "image/jpeg"),
        )
    }

    @Test fun `an empty or bare content uri is refused`() {
        assertFalse(SharedMediaResolver.isSafeContentUri("content://"))
        assertNull(SharedMediaResolver.resolve("content://", emptyList(), "image/jpeg"))
    }

    @Test fun `an unsupported mime type is refused rather than guessed`() {
        // If the platform says this is text/plain, the platform is right.
        assertNull(SharedMediaResolver.resolve("content://x/1", emptyList(), "text/plain"))
        assertNull(SharedMediaResolver.resolve("content://x/1", emptyList(), "application/pdf"))
    }

    @Test fun `a missing mime type falls back to the extension`() {
        val image = SharedMediaResolver.resolve("content://x/1/photo.JPEG", emptyList(), null)
        assertEquals(SharedMediaResolver.MODE_IMAGE, image?.mode)
        val audio = SharedMediaResolver.resolve("content://x/1/voice.m4a", emptyList(), null)
        assertEquals(SharedMediaResolver.MODE_AUDIO, audio?.mode)
        val video = SharedMediaResolver.resolve("content://x/1/clip.mp4", emptyList(), null)
        assertEquals(SharedMediaResolver.MODE_VIDEO, video?.mode)
    }

    @Test fun `an unknown extension with no mime type is refused`() {
        assertNull(SharedMediaResolver.resolve("content://x/1/thing", emptyList(), null))
    }

    @Test fun `a multiple share scans the first item only`() {
        // The pipeline produces ONE verdict. Scanning only the first item
        // while implying all were checked would be a lie, so the share
        // sheet offers the app per item.
        val shared = SharedMediaResolver.resolve(
            streamUri = null,
            clipUris = listOf("content://x/1", "content://x/2", "content://x/3"),
            mimeType = "image/png",
        )
        assertEquals("content://x/1", shared?.uri)
        assertEquals(SharedMediaResolver.MODE_IMAGE, shared?.mode)
    }

    @Test fun `the extra stream wins over the clip data`() {
        val shared = SharedMediaResolver.resolve(
            streamUri = "content://x/stream",
            clipUris = listOf("content://x/clip"),
            mimeType = "image/png",
        )
        assertEquals("content://x/stream", shared?.uri)
    }

    @Test fun `no uri at all is refused`() {
        assertNull(SharedMediaResolver.resolve(null, emptyList(), "image/jpeg"))
        assertNull(SharedMediaResolver.resolve("   ", listOf("  "), "image/jpeg"))
    }

    @Test fun `the mode strings are the ones the scanner route uses`() {
        // The scanner dispatches on the literal strings, so a drift here
        // would silently send an image down the audio route.
        assertEquals("IMAGE_ONLY", SharedMediaResolver.MODE_IMAGE)
        assertEquals("AUDIO_ONLY", SharedMediaResolver.MODE_AUDIO)
        assertEquals("VIDEO_ONLY", SharedMediaResolver.MODE_VIDEO)
    }
}
