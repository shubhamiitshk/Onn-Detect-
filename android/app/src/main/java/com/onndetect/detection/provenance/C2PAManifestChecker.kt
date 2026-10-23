package com.onndetect.detection.provenance

import android.content.Context
import android.net.Uri
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.security.MessageDigest
import javax.inject.Inject

/**
 * Locates and extracts the C2PA "Content Credentials" manifest store inside
 * an asset — spec-driven, full-file, F6 rewrite.
 *
 * What changed versus the pre-audit byte-scanner (defect F6, commit 66fce33):
 *  - The whole asset is walked, not the first 64 KB.
 *  - JPEG manifests are reassembled from multi-segment APP11 (0xFFEB) per
 *    ISO/IEC 19566-5 D.2: each segment carries the 5-byte signature "JUMBF",
 *    a 1-byte instance number En and a 1-byte continuation flag Z (Z=1: more
 *    segments for this En follow contiguously; Z=0: the box stream for this
 *    En is complete). Fragments of the same En are concatenated in order.
 *  - ISO-BMFF assets are walked box by box (4-byte size, 4-byte type,
 *    64-bit largesize when size==1, to-EOF when size==0); a manifest store
 *    is a top-level 'c2pa' box or a 'uuid' box whose 16-byte extended type
 *    identifies a C2PA store. We never descend into 'mdat' media payloads.
 *  - Inside the store, the JUMBF hierarchy is walked recursively and the
 *    claim ('clai'), signature (COSE_Sign1, 'cose') and certificate chain
 *    ('cert' and/or COSE x5chain) are extracted for [C2PAVerifier].
 *  - Data-hash assertions (c2pa.hash.data) are extracted so the verifier
 *    can recompute the hard binding between the manifest and these exact
 *    asset bytes.
 *  - The fictional `c2cs:` ASCII marker is gone: a manifest is located by
 *    structure, not by magic strings.
 */
/**
 * Extracts the digital_source_type VALUE from the claim bytes — the signed
 * claim is the only authoritative source, and the value is not restricted to
 * AI types: `digitalCapture` must survive intact so a verified manifest can
 * also evidence a genuine camera capture.
 *
 * Supports the CBOR 2.x shape (tstr key followed by its tstr value) and the
 * JSON 1.x shape, with a legacy AI-type scan as the last resort.
 */
private fun extractDigitalSourceType(claim: ByteArray?): String? {
    if (claim == null) return null
    val marker = "digital_source_type".toByteArray(Charsets.US_ASCII)
    run {
        var i = 0
        while (i + marker.size <= claim.size) {
            var matched = true
            for (j in marker.indices) {
                if (claim[i + j] != marker[j]) { matched = false; break }
            }
            if (matched) {
                val next = Cbor.decodeAt(claim, i + marker.size)?.first
                if (next is Cbor.Value.TStr) return next.text
                break
            }
            i++
        }
    }
    val text = String(claim, Charsets.ISO_8859_1)
    Regex("\"digital_source_type\"[ ]*:[ ]*\"([^\"]+)\"").find(text)?.let {
        return it.groupValues[1]
    }
    return ProvenanceEvidence.AI_SOURCE_TYPES.firstOrNull { text.contains(it) }
}

class C2PAManifestChecker @Inject constructor() {

    data class DataHash(
        /** Excluded byte ranges [start, end) in file coordinates. */
        val exclusions: List<Pair<Long, Long>>,
        /** sha-256 digest, raw 32 bytes; null when the stored value was unparsable. */
        val hash: ByteArray?,
        val alg: String?,
        val raw: String?,
    ) {
        override fun equals(other: Any?) = this === other
        override fun hashCode() = System.identityHashCode(this)
    }

    data class ManifestFinding(
        val present: Boolean,
        val digitalSourceType: String? = null,
        val container: String? = null,
        /** The claim bytes ('clai' box content, or the COSE payload). */
        val claimData: ByteArray? = null,
        /** The COSE_Sign1 bytes ('cose' box content). */
        val coseBytes: ByteArray? = null,
        /** Certificate chain as DER blobs, leaf first ('cert' box split). */
        val certificatesDer: List<ByteArray> = emptyList(),
        /** Parsed data-hash assertions. */
        val dataHashes: List<DataHash> = emptyList(),
        /** Byte ranges of the manifest store within the file. */
        val manifestRanges: List<Pair<Long, Long>> = emptyList(),
        /** The full asset bytes while hashing is possible; null when too large. */
        val fileBytes: ByteArray? = null,
        val note: String? = null,
    ) {
        fun toBuilderFields(): String = "container=$container claim=${claimData != null} " +
            "cose=${coseBytes != null} certs=${certificatesDer.size} hashes=${dataHashes.size}"
    }

    /** Inspects [imageUri] for a C2PA manifest store. */
    fun analyze(context: Context, imageUri: Uri): ManifestFinding {
        return try {
            context.contentResolver.openInputStream(imageUri)?.use { inspect(it) }
                ?: ManifestFinding(present = false, note = "asset unreadable")
        } catch (_: Exception) {
            ManifestFinding(present = false)
        }
    }

    /** Inspects a stream. Exposed for testing and for [C2PAVerifier]. */
    fun inspect(stream: InputStream): ManifestFinding {
        return try {
            val bytes = readFully(stream)
                ?: return ManifestFinding(present = false, note = "asset exceeds the on-device provenance budget")
            when {
                bytes.size >= 12 && bytes[0] == 0xFF.toByte() && bytes[1] == 0xD8.toByte() ->
                    inspectJpeg(bytes)
                bytes.size >= 12 && bytes[4] == 'f'.code.toByte() && bytes[5] == 't'.code.toByte() &&
                    bytes[6] == 'y'.code.toByte() && bytes[7] == 'p'.code.toByte() ->
                    inspectIsoBmff(bytes)
                else -> ManifestFinding(present = false, note = "unsupported container for provenance")
            }
        } catch (_: Exception) {
            ManifestFinding(present = false)
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Container walks
    // ─────────────────────────────────────────────────────────────────────

    private fun inspectJpeg(file: ByteArray): ManifestFinding {
        var offset = 2 // past SOI
        // En -> in-progress fragment buffer; closed when a Z=0 segment arrives.
        val streams = LinkedHashMap<Int, ByteArrayOutputStream>()
        val closed = LinkedHashMap<Int, ByteArray>()
        val ranges = mutableListOf<Pair<Long, Long>>()

        while (offset + 4 <= file.size) {
            if (file[offset] != 0xFF.toByte()) { offset++; continue }
            val marker = file[offset + 1].toInt() and 0xFF
            if (marker == 0xFF) { offset++; continue } // fill byte
            when {
                marker == 0xD8 || marker == 0x01 || marker in 0xD0..0xD7 -> { offset += 2; continue }
                marker == 0xDA || marker == 0xD9 -> return finishJpeg(closed, ranges, file)
            }
            val length = ((file[offset + 2].toInt() and 0xFF) shl 8) or (file[offset + 3].toInt() and 0xFF)
            if (length < 2) return ManifestFinding(present = false, note = "corrupt JPEG segment table")
            val segStart = offset + 4
            val segEnd = (offset + 2 + length).coerceAtMost(file.size)

            if (marker == 0xEB && segEnd - segStart >= 7) { // APP11 JUMBF: 5 magic + En + Z
                val payload = file.copyOfRange(segStart, segEnd)
                if (matchesApp11Signature(payload)) {
                    val en = payload[5].toInt() and 0xFF
                    val z = payload[6].toInt() and 0xFF
                    val fragment = payload.copyOfRange(7, payload.size)
                    val buf = streams.getOrPut(en) { ByteArrayOutputStream() }
                    buf.write(fragment)
                    ranges += offset.toLong() to (offset + 2 + length).toLong()
                    if (z == 0) { // this En's box stream is complete
                        streams.remove(en)?.let { closed[en] = it.toByteArray() }
                    }
                }
            }
            offset += 2 + length
        }
        return finishJpeg(closed, ranges, file)
    }

    private fun finishJpeg(
        closed: Map<Int, ByteArray>,
        ranges: List<Pair<Long, Long>>,
        file: ByteArray,
    ): ManifestFinding {
        // Only En groups that were CLOSED (Z=0 seen) may be parsed; an
        // unterminated group is a truncated manifest and must not be trusted.
        for ((en, store) in closed) {
            val parts = parseStore(store) ?: continue
            return parts.toFinding(container = "jpeg", ranges = ranges, fileBytes = file, note = "En=$en")
        }
        return ManifestFinding(present = false, fileBytes = file, manifestRanges = ranges)
    }

    private fun inspectIsoBmff(file: ByteArray): ManifestFinding {
        val ranges = mutableListOf<Pair<Long, Long>>()
        var offset = 0
        while (offset + 8 <= file.size) {
            var size = ((file[offset].toLong() and 0xFF) shl 24) or ((file[offset + 1].toLong() and 0xFF) shl 16) or
                ((file[offset + 2].toLong() and 0xFF) shl 8) or (file[offset + 3].toLong() and 0xFF)
            val type = String(file, offset + 4, 4, Charsets.US_ASCII)
            var header = 8
            if (size == 1L) {
                if (offset + 16 > file.size) break
                size = 0
                for (i in 0 until 8) size = (size shl 8) or (file[offset + 8 + i].toLong() and 0xFF)
                header = 16
            }
            if (size == 0L) size = (file.size - offset).toLong() // to EOF
            if (size < header || offset + size > file.size) break
            val contentStart = (offset + header).toLong()
            val contentEnd = (offset + size).toLong()

            val storeBytes: ByteArray? = when {
                type == "c2pa" -> file.copyOfRange(contentStart.toInt(), contentEnd.toInt())
                type == "uuid" && contentEnd - contentStart >= 16 &&
                    isC2paExtendedType(file, contentStart.toInt()) ->
                    file.copyOfRange(contentStart.toInt() + 16, contentEnd.toInt())
                else -> null
            }
            if (storeBytes != null) {
                ranges += offset.toLong() to (offset + size)
                val parts = parseStore(storeBytes) ?: continue
                return parts.toFinding(container = "isobmff", ranges = ranges, fileBytes = file, note = null)
            }
            // Never descend into media payloads; skip every other top-level box whole.
            offset += size.toInt()
        }
        return ManifestFinding(present = false, fileBytes = file, manifestRanges = ranges)
    }

    // ─────────────────────────────────────────────────────────────────────
    // JUMBF store walk (tolerant, recursive)
    // ─────────────────────────────────────────────────────────────────────

    private class StoreParts {
        var claim: ByteArray? = null
        var cose: ByteArray? = null
        val certs = mutableListOf<ByteArray>()
        val hashes = mutableListOf<DataHash>()
        var storeSize = 0

        fun toFinding(container: String, ranges: List<Pair<Long, Long>>, fileBytes: ByteArray, note: String?): ManifestFinding {
            // The C2PA claim is the COSE payload; a missing 'clai' box is tolerated.
            val claimData = claim ?: cose?.let { CoseSign1.parse(it)?.payload }
            val sourceType = extractDigitalSourceType(claimData)
            return ManifestFinding(
                present = true,
                digitalSourceType = sourceType,
                container = container,
                claimData = claimData,
                coseBytes = cose,
                certificatesDer = certs,
                dataHashes = hashes,
                manifestRanges = ranges,
                fileBytes = fileBytes,
                note = note,
            )
        }
    }

    /** Parses a manifest store: walks the JUMBF box hierarchy inside [store]. */
    private fun parseStore(store: ByteArray): StoreParts? {
        val parts = StoreParts()
        parts.storeSize = store.size
        if (store.size < 8) return null
        walkBoxes(store, 0, store.size, 0, parts)
        extractDataHashes(store, parts)
        if (parts.claim == null && parts.cose == null) return null
        return parts
    }

    private fun walkBoxes(bytes: ByteArray, start: Int, end: Int, depth: Int, parts: StoreParts) {
        if (depth > 8) return
        var p = start
        while (p + 8 <= end) {
            var size = ((bytes[p].toLong() and 0xFF) shl 24) or ((bytes[p + 1].toLong() and 0xFF) shl 16) or
                ((bytes[p + 2].toLong() and 0xFF) shl 8) or (bytes[p + 3].toLong() and 0xFF)
            var header = 8
            if (size == 1L) { // 64-bit largesize
                if (p + 16 > end) return
                size = 0
                for (i in 0 until 8) size = (size shl 8) or (bytes[p + 8 + i].toLong() and 0xFF)
                header = 16
            }
            if (size < 8 || p + size > end) return
            val typeBytes = bytes.copyOfRange(p + 4, p + 8)
            val type = String(typeBytes, Charsets.US_ASCII).trimEnd { it == '\u0000' || it == ' ' }
            val contentStart = p + header
            val contentEnd = p + size.toInt()

            when (type) {
                "clai", "claim" -> if (parts.claim == null) parts.claim = bytes.copyOfRange(contentStart, contentEnd)
                "cose", "sig", "sign" -> if (parts.cose == null) parts.cose = bytes.copyOfRange(contentStart, contentEnd)
                "cert", "certs", "chain" -> splitDerCertificates(bytes, contentStart, contentEnd, parts)
                "c2pa", "jumb", "jumd", "jumbf", "" -> walkBoxes(bytes, contentStart, contentEnd, depth + 1, parts)
                else -> if (size > 16) walkBoxes(bytes, contentStart, contentEnd, depth + 1, parts)
            }
            if (size == 0L) return
            p += size.toInt()
        }
    }

    /** Splits concatenated DER certificates in [start, end) into [parts.certs]. */
    private fun splitDerCertificates(bytes: ByteArray, start: Int, end: Int, parts: StoreParts) {
        var i = start
        while (i + 4 <= end) {
            if (bytes[i] == 0x30.toByte() && bytes[i + 1] == 0x82.toByte()) {
                val len = ((bytes[i + 2].toInt() and 0xFF) shl 8) or (bytes[i + 3].toInt() and 0xFF)
                val total = len + 4
                if (total in 200..16384 && i + total <= end && bytes[i + 4] == 0x30.toByte()) {
                    parts.certs += bytes.copyOfRange(i, i + total)
                    i += total
                    continue
                }
            }
            i++
        }
    }

    /**
     * Extracts c2pa.hash.data assertions from the store: the label is
     * located, then the following JSON (1.x) or CBOR (2.x) object is parsed
     * into exclusions + digest.
     */
    private fun extractDataHashes(store: ByteArray, parts: StoreParts) {
        val label = "c2pa.hash.data"
        var idx = indexOfAscii(store, label, 0)
        while (idx >= 0) {
            parseDataHashAfterLabel(store, idx + label.length)?.let { parts.hashes += it }
            idx = indexOfAscii(store, label, idx + 1)
        }
    }

    private fun parseDataHashAfterLabel(store: ByteArray, from: Int): DataHash? {
        // Find the start of the assertion object: '{' (JSON) or a CBOR map
        // header (0xA1..0xBF) within 16 bytes of the label.
        var objStart = -1
        var json = false
        for (i in from until minOf(from + 16, store.size)) {
            val b = store[i].toInt() and 0xFF
            if (b == 0x7B) { objStart = i; json = true; break }
            if (b in 0xA1..0xBF) { objStart = i; break }
        }
        if (objStart < 0) return null

        // A data-hash assertion MUST carry a digest. Occurrences of the label
        // inside claims (assertion references, c2pa_states lists) parse to no
        // object or to no hash, and are skipped: they are references, not
        // assertions.
        var candidate: DataHash? = null
        if (json) {
            // JSON (C2PA 1.x): {"exclusions":[{"start":N,"end":M},...],"alg":"sha256","hash":"b64/hex"}
            val depth = java.util.ArrayDeque<Int>()
            var end = -1
            var inString = false
            for (i in objStart until store.size) {
                val c = store[i].toInt() and 0xFF
                if (inString) { if (c == 0x5C) continue; if (c == 0x22) inString = false; continue }
                when (c) {
                    0x22 -> inString = true
                    0x7B -> depth.push(i)
                    0x7D -> { depth.pop(); if (depth.isEmpty()) { end = i + 1; break } }
                }
            }
            if (end < 0) return null
            val text = String(store, objStart, end - objStart, Charsets.UTF_8)
            candidate = try {
                val obj = org.json.JSONObject(text)
                val exclusions = mutableListOf<Pair<Long, Long>>()
                val exArr = obj.optJSONArray("exclusions")
                if (exArr != null) for (i in 0 until exArr.length()) {
                    val e = exArr.getJSONObject(i)
                    exclusions += e.getLong("start") to e.getLong("end")
                }
                DataHash(
                    exclusions = exclusions,
                    hash = decodeHashString(obj.optString("hash", "")),
                    alg = obj.optString("alg", "sha256"),
                    raw = obj.optString("hash", ""),
                )
            } catch (_: Exception) {
                null
            }
        } else {
            // CBOR (C2PA 2.x): {"exclusions":[{"start":uint,"end":uint}...],"alg":tstr,"hash":bstr}
            candidate = Cbor.decodeAt(store, objStart)?.first.let { value ->
                val map = value as? Cbor.Value.Map ?: return null
                val exclusions = mutableListOf<Pair<Long, Long>>()
                (map.get("exclusions") as? Cbor.Value.Arr)?.items?.forEach { entry ->
                    val m = entry as? Cbor.Value.Map ?: return@forEach
                    val s = (m.get("start") as? Cbor.Value.UInt)?.value ?: return@forEach
                    val e = (m.get("end") as? Cbor.Value.UInt)?.value ?: return@forEach
                    exclusions += s to e
                }
                val hash: ByteArray? = (map.get("hash") as? Cbor.Value.BStr)?.bytes
                    ?: (map.get("hash") as? Cbor.Value.TStr)?.text?.let(::decodeHashString)
                DataHash(
                    exclusions = exclusions,
                    hash = hash,
                    alg = (map.get("alg") as? Cbor.Value.TStr)?.text ?: "sha256",
                    raw = null,
                )
            }
        }
        return if (candidate?.hash != null) candidate else null
    }

    private fun decodeHashString(s: String): ByteArray? {
        if (s.length == 64 && s.all { it.isDigit() || it in 'a'..'f' || it in 'A'..'F' }) {
            val out = ByteArray(32)
            for (i in 0 until 32) out[i] = ((Character.digit(s[i * 2], 16) shl 4) + Character.digit(s[i * 2 + 1], 16)).toByte()
            return out
        }
        return runCatching { java.util.Base64.getDecoder().decode(s) }.getOrNull()
    }

    /** The signed digital source type comes from the CLAIM bytes only. */
    private fun findDigitalSourceType(claim: ByteArray): String? {
        val text = String(claim, Charsets.ISO_8859_1)
        return ProvenanceEvidence.AI_SOURCE_TYPES.firstOrNull { text.contains(it) }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────

    private fun matchesApp11Signature(payload: ByteArray): Boolean {
        for (i in APP11_HEADER.indices) if (payload[i] != APP11_HEADER[i]) return false
        return true
    }

    private fun isC2paExtendedType(file: ByteArray, extStart: Int): Boolean {
        // A C2PA JUMBF 'uuid' box carries a 16-byte extended type whose first
        // four bytes are the 'c2pa' fourcc (63 32 70 61) followed by the
        // JUMBF UUID bytes.
        return extStart + 4 <= file.size &&
            file[extStart] == 'c'.code.toByte() && file[extStart + 1] == '2'.code.toByte() &&
            file[extStart + 2] == 'p'.code.toByte() && file[extStart + 3] == 'a'.code.toByte()
    }

    private fun indexOfAscii(haystack: ByteArray, needle: String, from: Int): Int {
        val needleBytes = needle.toByteArray(Charsets.US_ASCII)
        if (needleBytes.isEmpty() || haystack.size < needleBytes.size) return -1
        outer@ for (i in from..haystack.size - needleBytes.size) {
            if (i < 0) return -1
            for (j in needleBytes.indices) if (haystack[i + j] != needleBytes[j]) continue@outer
            return i
        }
        return -1
    }

    private fun readFully(stream: InputStream): ByteArray? {
        val out = ByteArrayOutputStream()
        val buf = ByteArray(1 shl 20)
        var total = 0L
        while (true) {
            val n = stream.read(buf)
            if (n <= 0) break
            total += n
            if (total > MAX_VERIFY_BYTES) return null
            out.write(buf, 0, n)
        }
        return out.toByteArray()
    }

    companion object {
        /** "JUMBF" + En + Z is 7 bytes; the box stream fragment follows. */
        private val APP11_HEADER = "JUMBF".toByteArray(Charsets.US_ASCII) // 5 bytes; En/Z read separately

        /**
         * On-device provenance budget. Typical JPEG/PNG/MP4 shared through
         * messaging apps are far below this; 4K video masters are verified
         * on the desktop path instead.
         */
        const val MAX_VERIFY_BYTES: Long = 256L * 1024 * 1024
    }
}
