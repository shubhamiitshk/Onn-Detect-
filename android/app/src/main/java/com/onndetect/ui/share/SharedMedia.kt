package com.onndetect.ui.share

/**
 * A media hand-off from another app's share sheet.
 *
 * @param uri the `content://` URI of the shared item.
 * @param mode the scan mode the shared item's MIME type maps to. These are
 *   the same strings the scanner route uses (`IMAGE_ONLY`, `AUDIO_ONLY`).
 */
data class SharedMedia(val uri: String, val mode: String)

/**
 * Resolves a share-sheet hand-off into a scan request.
 *
 * This is deliberately PURE: it takes the raw strings the platform hands
 * over and returns the decision, so every rule below is unit-testable on a
 * plain JVM with no Robolectric and no device. The Android adapter that
 * pulls the strings out of an `Intent` lives in [SharedMediaIntents].
 *
 * The rules are conservative on purpose. A share sheet is an untrusted
 * entry point: any app can hand this one a URI of any shape, and a
 * hand-off is a place where it is easy to accidentally open a door.
 */
object SharedMediaResolver {

    const val MODE_IMAGE = "IMAGE_ONLY"
    const val MODE_AUDIO = "AUDIO_ONLY"

    private const val CONTENT_SCHEME = "content://"

    /**
     * Resolve a hand-off, or null when it is not something we can scan.
     *
     * @param streamUri `Intent.EXTRA_STREAM`, as a string.
     * @param clipUris the `ClipData` items, for a multiple share.
     * @param mimeType the intent's MIME type. When it is null the mode is
     *   inferred from the file extension.
     */
    fun resolve(streamUri: String?, clipUris: List<String>, mimeType: String?): SharedMedia? {
        // A multiple share is scanned one item at a time: the pipeline
        // produces ONE verdict, and silently scanning only the first item
        // while implying all were checked would be a lie. The share sheet
        // offers the app per item for this reason.
        val uri = streamUri?.takeIf { it.isNotBlank() }
            ?: clipUris.firstOrNull { it.isNotBlank() }
            ?: return null
        if (!isSafeContentUri(uri)) return null
        val mode = modeFor(mimeType, uri) ?: return null
        return SharedMedia(uri, mode)
    }

    /**
     * Only a `content://` URI is accepted.
     *
     * The importer copies the bytes through the `ContentResolver`, which is
     * the only scheme that carries the provider's own read grant. A
     * `file://` URI would let a hand-off name a path in the app's private
     * storage, and an `http(s)://` URI would turn the scanner into a
     * network fetcher - which this app must never be, offline by design.
     */
    fun isSafeContentUri(uri: String): Boolean =
        uri.length > CONTENT_SCHEME.length && uri.startsWith(CONTENT_SCHEME, ignoreCase = true)

    /**
     * The scan mode for a MIME type, or null when the type is not ours.
     *
     * An explicit, unsupported type is REFUSED rather than guessed from the
     * extension: if the platform says this is `text/plain`, the platform is
     * right and we do not scan it.
     */
    fun modeFor(mimeType: String?, uri: String): String? {
        val type = mimeType?.trim()?.lowercase()
        return when {
            type != null && type.isNotEmpty() && type.startsWith("image/") -> MODE_IMAGE
            type != null && type.isNotEmpty() && type.startsWith("audio/") -> MODE_AUDIO
            type != null && type.isNotEmpty() -> null
            else -> modeFromExtension(uri)
        }
    }

    /**
     * The mode inferred from a file extension, used only when the platform
     * gave no MIME type at all.
     */
    private fun modeFromExtension(uri: String): String? {
        val path = uri.substringBefore('?').substringBefore('#').lowercase()
        return when {
            IMAGE_EXTENSIONS.any { path.endsWith(it) } -> MODE_IMAGE
            AUDIO_EXTENSIONS.any { path.endsWith(it) } -> MODE_AUDIO
            else -> null
        }
    }

    private val IMAGE_EXTENSIONS = listOf(
        ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".gif",
    )
    private val AUDIO_EXTENSIONS = listOf(
        ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".amr", ".3ga",
    )
}
