package com.onndetect.ui.share

import android.content.Intent
import android.net.Uri
import android.os.Build

/**
 * The Android adapter that pulls a share hand-off out of an `Intent`.
 *
 * It is a thin shell: every decision is made by [SharedMediaResolver],
 * which is pure and unit-tested on the JVM.
 */
object SharedMediaIntents {

    /**
     * Consume a share hand-off from [intent], or return null.
     *
     * The intent is CONSUMED: its extras are removed. Android re-delivers
     * the same Intent after a configuration change, and a scan must not run
     * twice because the user rotated the device.
     */
    fun consume(intent: Intent?): SharedMedia? {
        if (intent == null) return null
        val action = intent.action
        if (action != Intent.ACTION_SEND && action != Intent.ACTION_SEND_MULTIPLE) return null

        val resolved = SharedMediaResolver.resolve(
            streamUri = intent.streamUri(),
            clipUris = intent.clipUris(),
            mimeType = intent.type,
        )
        intent.removeExtra(Intent.EXTRA_STREAM)
        intent.clipData = null
        return resolved
    }

    @Suppress("DEPRECATION")
    private fun Intent.streamUri(): String? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)?.toString()
        } else {
            getParcelableExtra<Uri>(Intent.EXTRA_STREAM)?.toString()
        }

    private fun Intent.clipUris(): List<String> {
        val data = clipData ?: return emptyList()
        return (0 until data.itemCount).mapNotNull { index ->
            data.getItemAt(index)?.uri?.toString()
        }
    }
}
