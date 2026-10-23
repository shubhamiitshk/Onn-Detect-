package com.onndetect.detection.provenance

/**
 * Lightweight, zero-dependency CBOR (RFC 8949) decoder for C2PA manifest assertions
 * and RFC 9052 COSE_Sign1 envelopes.
 *
 * Enforces strict bounds checking and memory safety against untrusted byte streams.
 */
object Cbor {

    sealed class Value {
        data class UnsignedInt(val value: Long) : Value()
        data class NegativeInt(val value: Long) : Value()
        data class ByteString(val bytes: ByteArray) : Value() {
            override fun equals(other: Any?): Boolean {
                if (this === other) return true
                if (javaClass != other?.javaClass) return false
                return bytes.contentEquals((other as ByteString).bytes)
            }
            override fun hashCode(): Int = bytes.contentHashCode()
        }
        data class TextString(val text: String) : Value()
        data class ArrayValue(val items: List<Value>) : Value()
        data class MapValue(val entries: Map<Value, Value>) : Value()
        data class Tag(val tagNumber: Long, val item: Value) : Value()
        data class BooleanValue(val value: Boolean) : Value()
        data class Simple(val value: Int) : Value()
        object NullValue : Value()
    }

    /**
     * Decodes a CBOR byte payload into a [Value].
     * Returns null if the byte stream is malformed or truncated.
     */
    fun decode(bytes: ByteArray): Value? {
        val parser = Parser(bytes)
        val value = parser.parseValue()
        return value
    }

    private class Parser(private val bytes: ByteArray) {
        var cursor = 0

        fun parseValue(): Value? {
            if (cursor >= bytes.size) return null
            val head = bytes[cursor].toInt() and 0xFF
            cursor += 1

            val major = head ushr 5
            val info = head and 0x1F

            val argument = when (info) {
                in 0..23 -> info.toLong()
                // Fix 1: Unsigned mask prevents Byte.toLong() sign-extension on lengths 128..255
                24 -> ((bytes.getOrNull(cursor)?.toInt() ?: return null) and 0xFF).toLong().also { cursor += 1 }
                25 -> {
                    val b0 = (bytes.getOrNull(cursor)?.toInt() ?: return null) and 0xFF
                    val b1 = (bytes.getOrNull(cursor + 1)?.toInt() ?: return null) and 0xFF
                    cursor += 2
                    ((b0 shl 8) or b1).toLong()
                }
                26 -> {
                    var v = 0L
                    for (i in 0 until 4) {
                        val b = (bytes.getOrNull(cursor + i)?.toInt() ?: return null) and 0xFF
                        v = (v shl 8) or b.toLong()
                    }
                    cursor += 4
                    v
                }
                27 -> {
                    var v = 0L
                    for (i in 0 until 8) {
                        val b = (bytes.getOrNull(cursor + i)?.toInt() ?: return null) and 0xFF
                        v = (v shl 8) or b.toLong()
                    }
                    cursor += 8
                    v
                }
                31 -> -1L // Indefinite length
                else -> return null
            }

            return when (major) {
                0 -> Value.UnsignedInt(argument)
                1 -> Value.NegativeInt(-1L - argument)
                2 -> {
                    if (argument < 0 || argument > bytes.size - cursor || argument > 16_777_216) return null
                    val len = argument.toInt()
                    val slice = bytes.copyOfRange(cursor, cursor + len)
                    cursor += len
                    Value.ByteString(slice)
                }
                3 -> {
                    if (argument < 0 || argument > bytes.size - cursor || argument > 16_777_216) return null
                    val len = argument.toInt()
                    val text = String(bytes, cursor, len, Charsets.UTF_8)
                    cursor += len
                    Value.TextString(text)
                }
                4 -> {
                    if (argument < 0 || argument > 65536) return null
                    val count = argument.toInt()
                    val list = ArrayList<Value>(count)
                    for (i in 0 until count) {
                        val elem = parseValue() ?: return null
                        list.add(elem)
                    }
                    Value.ArrayValue(list)
                }
                5 -> {
                    if (argument < 0 || argument > 65536) return null
                    val count = argument.toInt()
                    val map = LinkedHashMap<Value, Value>(count)
                    for (i in 0 until count) {
                        val k = parseValue() ?: return null
                        val v = parseValue() ?: return null
                        map[k] = v
                    }
                    Value.MapValue(map)
                }
                6 -> {
                    val taggedItem = parseValue() ?: return null
                    Value.Tag(argument, taggedItem)
                }
                7 -> {
                    when (argument.toInt()) {
                        20 -> Value.BooleanValue(false)
                        21 -> Value.BooleanValue(true)
                        22 -> Value.NullValue
                        else -> Value.Simple(argument.toInt())
                    }
                }
                else -> null
            }
        }
    }
}
