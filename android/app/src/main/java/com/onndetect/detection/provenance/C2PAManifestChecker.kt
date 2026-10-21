package com.onndetect.detection.provenance

import android.content.Context
import android.net.Uri
import java.io.InputStream
import javax.inject.Inject

/**
 * Locates a C2PA "Content Credentials" manifest inside an asset.
 *
 * ## What this does and does NOT do
 *
 * This is a **DETECTOR**, not a verifier. It answers "does this asset carry
 * a C2PA manifest, and does that manifest name a generative digital source
 * type?" It does **not** verify the signature chain or the binding between
 * the manifest and the asset's bytes, because that requires the
 * `contentauth/c2pa-rs` verifier (via the `contentauth/c2pa-android`
 * SDK), which is not bundled yet.
 *
 * That distinction is load-bearing. The specification's **exclusion range**
 * allows parts of an asset to sit outside the signed region, so a manifest
 * that is merely PRESENT is not a claim: it can be forged, and it does not
 * prove that the parts of the file outside the signed range are genuine.
 *
 * Consequently [ProvenanceEvidence.C2PAClaim.signed] is always `false`
 * here, and the pipeline will never raise
 * [VerdictConfidence.PROVENANCE_CONFIRMED] from this path. The mechanism
 * is in place so that swapping in the real verifier is a one-line change.
 *
 * ## Where the manifest lives
 *
 *  * **JPEG** — an APP11 segment carrying a JUMBF box whose type is
 *    `c2pa`, or the C2PA claim UUID.
 *  * **ISO-BMFF** (MP4 / HEIF) — a top-level `c2pa` box.
 */
class C2PAManifestChecker @Inject constructor() {

    data class ManifestFinding(
        val present: Boolean,
        val digitalSourceType: String? = null,
        val container: String? = null,
    )

    /**
     * Inspect [imageUri] for a C2PA manifest.
     *
     * Returns a finding with `present = false` for anything unreadable;
     * an unreadable file is "no evidence", never "no AI".
     */
    fun analyze(context: Context, imageUri: Uri): ManifestFinding {
        return try {
            context.contentResolver.openInputStream(imageUri)?.use { stream ->
                inspect(stream)
            } ?: ManifestFinding(present = false)
        } catch (_: Exception) {
            ManifestFinding(present = false)
        }
    }

    /** Inspect a stream. Exposed for testing. */
    fun inspect(stream: InputStream): ManifestFinding {
        return try {
            val head = ByteArray(HEADER_BYTES)
            val read = stream.read(head)
            if (read < 12) return ManifestFinding(present = false)
            when {
                head[0] == 0xFF.toByte() && head[1] == 0xD8.toByte() ->
                    inspectJpeg(head)
                head[4] == 'f'.code.toByte() && head[5] == 't'.code.toByte() &&
                    head[6] == 'y'.code.toByte() && head[7] == 'p'.code.toByte() ->
                    inspectIsoBmff(stream)
                else -> ManifestFinding(present = false)
            }
        } catch (_: Exception) {
            ManifestFinding(present = false)
        }
    }

    private fun inspectJpeg(head: ByteArray): ManifestFinding {
        // Walk the JPEG marker segments looking for an APP11 that carries
        // the C2PA JUMBF box.
        var offset = 2
        while (offset + 4 <= head.size) {
            if (head[offset] != 0xFF.toByte()) { offset++; continue }
            val marker = head[offset + 1].toInt() and 0xFF
            if (marker == 0xD8 || marker == 0x01 || (marker in 0xD0..0xD7)) { offset += 2; continue }
            if (marker == 0xDA || marker == 0xD9) break
            val length = ((head[offset + 2].toInt() and 0xFF) shl 8) or (head[offset + 3].toInt() and 0xFF)
            if (length < 2) break
            val payloadStart = offset + 4
            val payloadEnd = (offset + 2 + length).coerceAtMost(head.size)
            if (payloadEnd > payloadStart) {
                val payload = head.copyOfRange(payloadStart, payloadEnd)
                if (containsC2PAMarker(payload)) {
                    return ManifestFinding(
                        present = true,
                        digitalSourceType = findDigitalSourceType(payload),
                        container = "jpeg",
                    )
                }
            }
            offset += 2 + length
        }
        return ManifestFinding(present = false)
    }

    private fun inspectIsoBmff(stream: InputStream): ManifestFinding {
        // A top-level 'c2pa' box in an ISO base media file.
        val buffer = ByteArray(64)
        val read = stream.read(buffer)
        if (read < 8) return ManifestFinding(present = false)
        for (i in 0 until read - 4) {
            if (buffer[i] == 'c'.code.toByte() && buffer[i + 1] == '2'.code.toByte() &&
                buffer[i + 2] == 'p'.code.toByte() && buffer[i + 3] == 'a'.code.toByte()
            ) {
                return ManifestFinding(
                    present = true,
                    digitalSourceType = findDigitalSourceType(buffer.copyOfRange(i, read)),
                    container = "isobmff",
                )
            }
        }
        return ManifestFinding(present = false)
    }

    private fun containsC2PAMarker(payload: ByteArray): Boolean {
        val needle = "c2pa".toByteArray(Charsets.US_ASCII)
        return indexOf(payload, needle) >= 0
    }

    private fun findDigitalSourceType(payload: ByteArray): String? {
        val text = String(payload, Charsets.ISO_8859_1)
        for (candidate in ProvenanceEvidence.AI_SOURCE_TYPES) {
            if (text.contains(candidate)) return candidate
        }
        return null
    }

    private fun indexOf(haystack: ByteArray, needle: ByteArray): Int {
        if (needle.isEmpty() || haystack.size < needle.size) return -1
        outer@ for (i in 0..haystack.size - needle.size) {
            for (j in needle.indices) {
                if (haystack[i + j] != needle[j]) continue@outer
            }
            return i
        }
        return -1
    }

    companion object {
        /** Bytes of the file inspected. The C2PA box is at the front. */
        const val HEADER_BYTES = 1024
    }
}
