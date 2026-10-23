package com.onndetect.detection.provenance

/**
 * Minimal CBOR (RFC 8949) support for offline C2PA manifest verification.
 *
 * Decoder: the definite-length subset that C2PA manifests actually use —
 * unsigned/unsigned integers, byte strings, text strings, arrays, maps,
 * tags, and standalone simple values. Indefinite-length items are accepted
 * for arrays/maps (terminated by 0xFF) so that non-canonical encoders still
 * parse.
 *
 * Encoder: exactly what COSE Sig_structure (RFC 9052 4.2) needs.
 *
 * Pure JVM: no Android imports, unit-testable on the desktop.
 */
object Cbor {

    /** A decoded CBOR value tree. */
    sealed class Value {
        data class UInt(val value: Long) : Value()
        data class NInt(val value: Long) : Value() // value is the negative number itself
        data class BStr(val bytes: ByteArray) : Value() {
            override fun equals(other: Any?) = other is BStr && bytes.contentEquals(other.bytes)
            override fun hashCode() = bytes.contentHashCode()
        }
        data class TStr(val text: String) : Value()
        data class Arr(val items: List<Value>) : Value()
        data class Map(val entries: List<Pair<Value, Value>>) : Value() {
            fun get(key: String): Value? = entries.firstOrNull { it.first is TStr && (it.first as TStr).text == key }?.second
            fun get(key: Long): Value? = entries.firstOrNull {
                (it.first is UInt && (it.first as UInt).value == key) ||
                    (it.first is NInt && (it.first as NInt).value == key)
            }?.second
        }
        data class Tagged(val tag: Long, val inner: Value) : Value()
        data object Other : Value()
    }

    /** Decodes one CBOR item at [offset]; returns null when the bytes are not CBOR. */
    fun decodeAt(bytes: ByteArray, offset: Int, depth: Int = 0): Pair<Value, Int>? {
        if (offset < 0 || offset >= bytes.size || depth > 16) return null
        val initial = bytes[offset].toInt() and 0xFF
        val major = initial ushr 5
        val info = initial and 0x1F
        var cursor = offset + 1

        fun readLength(): Long? {
            return when (info) {
                in 0..23 -> info.toLong()
                24 -> ((bytes.getOrNull(cursor)?.toInt() ?: return null) and 0xFF).toLong().also { cursor += 1 }
                25 -> {
                    if (cursor + 2 > bytes.size) return null
                    (((bytes[cursor].toInt() and 0xFF) shl 8) or (bytes[cursor + 1].toInt() and 0xFF)).toLong()
                        .also { cursor += 2 }
                }
                26 -> {
                    if (cursor + 4 > bytes.size) return null
                    var v = 0L
                    for (i in 0 until 4) { v = (v shl 8) or (bytes[cursor + i].toLong() and 0xFF) }
                    cursor += 4; v
                }
                27 -> {
                    if (cursor + 8 > bytes.size) return null
                    var v = 0L
                    for (i in 0 until 8) { v = (v shl 8) or (bytes[cursor + i].toLong() and 0xFF) }
                    cursor += 8; v
                }
                31 -> -1L // indefinite
                else -> null
            }
        }

        when (major) {
            0 -> { // unsigned integer
                val len = readLength() ?: return null
                if (len == -1L) return null
                return Value.UInt(len) to cursor
            }
            1 -> { // negative integer: value = -1 - n
                val len = readLength() ?: return null
                if (len == -1L) return null
                return Value.NInt(-1L - len) to cursor
            }
            2 -> { // byte string
                val len = readLength() ?: return null
                if (len == -1L) { // indefinite: concatenate definite chunks until the break code
                    val out = java.io.ByteArrayOutputStream()
                    while (cursor < bytes.size) {
                        if (bytes[cursor].toInt() and 0xFF == 0xFF) { cursor += 1; break }
                        val chunk = decodeAt(bytes, cursor, depth + 1) ?: return null
                        val chunkBstr = chunk.first as? Value.BStr ?: return null
                        out.write(chunkBstr.bytes)
                        cursor = chunk.second
                    }
                    return Value.BStr(out.toByteArray()) to cursor
                }
                if (len < 0 || cursor + len > bytes.size) return null
                return Value.BStr(bytes.copyOfRange(cursor, cursor + len.toInt())) to (cursor + len.toInt())
            }
            3 -> { // text string
                val len = readLength() ?: return null
                if (len == -1L) return null
                if (len < 0 || cursor + len > bytes.size) return null
                val text = runCatching { String(bytes, cursor, len.toInt(), Charsets.UTF_8) }.getOrNull() ?: return null
                return Value.TStr(text) to (cursor + len.toInt())
            }
            4 -> { // array
                val len = readLength() ?: return null
                val items = mutableListOf<Value>()
                if (len == -1L) { // indefinite
                    while (cursor < bytes.size) {
                        if (bytes[cursor].toInt() and 0xFF == 0xFF) { cursor += 1; break }
                        val item = decodeAt(bytes, cursor, depth + 1) ?: return null
                        items += item.first; cursor = item.second
                    }
                } else {
                    if (len > 4096) return null
                    repeat(len.toInt()) {
                        val item = decodeAt(bytes, cursor, depth + 1) ?: return null
                        items += item.first; cursor = item.second
                    }
                }
                return Value.Arr(items) to cursor
            }
            5 -> { // map
                val len = readLength() ?: return null
                val entries = mutableListOf<Pair<Value, Value>>()
                if (len == -1L) { // indefinite
                    while (cursor < bytes.size) {
                        if (bytes[cursor].toInt() and 0xFF == 0xFF) { cursor += 1; break }
                        val key = decodeAt(bytes, cursor, depth + 1) ?: return null
                        val value = decodeAt(bytes, key.second, depth + 1) ?: return null
                        entries += key.first to value.first; cursor = value.second
                    }
                } else {
                    if (len > 4096) return null
                    repeat(len.toInt()) {
                        val key = decodeAt(bytes, cursor, depth + 1) ?: return null
                        val value = decodeAt(bytes, key.second, depth + 1) ?: return null
                        entries += key.first to value.first; cursor = value.second
                    }
                }
                return Value.Map(entries) to cursor
            }
            6 -> { // tag: skip the tag number, decode the inner item
                val len = readLength() ?: return null
                if (len == -1L) return null
                val inner = decodeAt(bytes, cursor, depth + 1) ?: return null
                return Value.Tagged(len, inner.first) to inner.second
            }
            else -> { // 7: float/simple — consume header only
                return when (info) {
                    20, 21, 22, 23 -> Value.Other to cursor
                    25 -> { if (cursor + 2 > bytes.size) return null; cursor += 2; Value.Other to cursor }
                    26 -> { if (cursor + 4 > bytes.size) return null; cursor += 4; Value.Other to cursor }
                    27 -> { if (cursor + 8 > bytes.size) return null; cursor += 8; Value.Other to cursor }
                    31 -> Value.Other to cursor // break code seen standalone
                    else -> if (info < 24) Value.Other to cursor else null
                }
            }
        }
    }

    /** Convenience: decode exactly one item covering the whole buffer. */
    fun decode(bytes: ByteArray): Value? {
        val parsed = decodeAt(bytes, 0) ?: return null
        return if (parsed.second == bytes.size) parsed.first else null
    }

    // ─────────────────────────────────────────────────────────────────────
    // Encoder: the Sig_structure bytes that COSE signatures are computed over
    // (RFC 9052 §4.2): ["Signature1", protected_bstr, external_aad_bstr,
    // payload_bstr] — a definite-length 4-element array.
    // ─────────────────────────────────────────────────────────────────────

    /** COSE_Sign1 ToBeSigned bytes: ["Signature1", protected, aad, payload]. */
    fun encodeSigStructure(protected: ByteArray, externalAad: ByteArray, payload: ByteArray): ByteArray {
        val out = java.io.ByteArrayOutputStream()
        out.write(0x84) // array(4)
        out.write(0x6A) // text(10)
        out.write("Signature1".toByteArray(Charsets.US_ASCII))
        writeByteString(out, protected)
        writeByteString(out, externalAad)
        writeByteString(out, payload)
        return out.toByteArray()
    }

    private fun writeByteString(out: java.io.ByteArrayOutputStream, bytes: ByteArray) {
        when {
            bytes.size < 24 -> out.write(0x40 + bytes.size)
            bytes.size < 256 -> { out.write(0x58); out.write(bytes.size) }
            bytes.size < 65536 -> {
                out.write(0x59)
                out.write((bytes.size shr 8) and 0xFF); out.write(bytes.size and 0xFF)
            }
            else -> {
                out.write(0x5A)
                for (shift in 24 downTo 0 step 8) out.write((bytes.size shr shift) and 0xFF)
            }
        }
        out.write(bytes)
    }

    /** Encodes a definite-length byte string (used by tests to build headers). */
    fun encodeByteString(bytes: ByteArray): ByteArray {
        val out = java.io.ByteArrayOutputStream()
        writeByteString(out, bytes)
        return out.toByteArray()
    }
}
