package com.onndetect.detection.provenance

import android.content.Context
import android.net.Uri
import javax.inject.Inject

/** A real C2PA implementation must verify bindings, signatures and trust. */
class C2PAManifestChecker @Inject constructor() {
    @Suppress("UNUSED_PARAMETER")
    fun analyze(context: Context, imageUri: Uri): Float = -1f
    // Unavailable: a substring search is neither a parser nor a verifier.
}
