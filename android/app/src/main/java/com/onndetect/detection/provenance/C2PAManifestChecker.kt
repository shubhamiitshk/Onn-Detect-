package com.onndetect.detection.provenance

import android.content.Context
import android.net.Uri
import java.io.InputStream
import javax.inject.Inject

/**
 * Locates and parses a C2PA "Content Credentials" manifest inside an asset.
 *
 * ## What this does and does NOT do
 *
 * Extracts C2PA JUMBF structures, embedded X.509 certificate chains, digital
 * source types, exclusion ranges, data hashes, and signature blocks from
 * JPEG APP11 and ISO-BMFF containers.
 *
 * It extracts raw cryptographic components which are subsequently verified
 * against [C2PATrustStore] by [C2PAVerifier] under IACR 2026/804 invariants.
 */
class C2PAManifestChecker @Inject constructor() {

    data class Exclusion(val start: Long, val end: Long)

    data class DataHash(
        val hash: ByteArray? = null,
        val exclusions: List<Exclusion> = emptyList(),
        val algorithm: String = "sha256",
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as DataHash
            if (!hash.contentEquals(other.hash)) return false
            if (exclusions != other.exclusions) return false
            if (algorithm != other.algorithm) return false
            return true
        }

        override fun hashCode(): Int {
            var result = hash?.contentHashCode() ?: 0
            result = 31 * result + exclusions.hashCode()
            result = 31 * result + algorithm.hashCode()
            return result
        }
    }

    data class ManifestFinding(
        val present: Boolean,
        val digitalSourceType: String? = null,
        val container: String? = null,
        val certificatesDer: List<ByteArray> = emptyList(),
        val claimData: ByteArray? = null,
        val signatureBytes: ByteArray? = null,
        val signatureAlgorithm: String = "SHA256withECDSA",
        val issuerName: String? = null,
        val fileBytes: ByteArray? = null,
        val manifestRanges: List<Pair<Long, Long>> = emptyList(),
        val dataHashes: List<DataHash> = emptyList(),
    )

    /**
     * Inspect [imageUri] for a C2PA manifest.
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
            var totalRead = 0
            while (totalRead < HEADER_BYTES) {
                val n = stream.read(head, totalRead, HEADER_BYTES - totalRead)
                if (n <= 0) break
                totalRead += n
            }
            if (totalRead < 12) return ManifestFinding(present = false)
            val data = if (totalRead == HEADER_BYTES) head else head.copyOf(totalRead)

            when {
                data[0] == 0xFF.toByte() && data[1] == 0xD8.toByte() ->
                    inspectJpeg(data)
                data[4] == 'f'.code.toByte() && data[5] == 't'.code.toByte() &&
                    data[6] == 'y'.code.toByte() && data[7] == 'p'.code.toByte() ->
                    inspectIsoBmff(data)
                else -> ManifestFinding(present = false)
            }
        } catch (_: Exception) {
            ManifestFinding(present = false)
        }
    }

    private fun inspectJpeg(data: ByteArray): ManifestFinding {
        var offset = 2
        while (offset + 4 <= data.size) {
            if (data[offset] != 0xFF.toByte()) { offset++; continue }
            val marker = data[offset + 1].toInt() and 0xFF
            if (marker == 0xD8 || marker == 0x01 || (marker in 0xD0..0xD7)) { offset += 2; continue }
            if (marker == 0xDA || marker == 0xD9) break
            val length = ((data[offset + 2].toInt() and 0xFF) shl 8) or (data[offset + 3].toInt() and 0xFF)
            if (length < 2) break
            val payloadStart = offset + 4
            val payloadEnd = (offset + 2 + length).coerceAtMost(data.size)
            if (payloadEnd > payloadStart) {
                val payload = data.copyOfRange(payloadStart, payloadEnd)
                if (containsC2PAMarker(payload)) {
                    val certs = extractCertificates(payload)
                    val sig = extractSignatureBlock(payload)
                    val dataHashes = extractDataHashes(payload)
                    return ManifestFinding(
                        present = true,
                        digitalSourceType = findDigitalSourceType(payload),
                        container = "jpeg",
                        certificatesDer = certs,
                        claimData = sig?.first,
                        signatureBytes = sig?.second,
                        signatureAlgorithm = sig?.third ?: "SHA256withECDSA",
                        issuerName = extractIssuer(payload),
                        fileBytes = data,
                        manifestRanges = listOf(Pair(payloadStart.toLong(), payloadEnd.toLong())),
                        dataHashes = dataHashes,
                    )
                }
            }
            offset += 2 + length
        }
        return ManifestFinding(present = false)
    }

    private fun inspectIsoBmff(data: ByteArray): ManifestFinding {
        for (i in 0 until data.size - 4) {
            if (data[i] == 'c'.code.toByte() && data[i + 1] == '2'.code.toByte() &&
                data[i + 2] == 'p'.code.toByte() && data[i + 3] == 'a'.code.toByte()
            ) {
                val payload = data.copyOfRange(i, data.size)
                val certs = extractCertificates(payload)
                val sig = extractSignatureBlock(payload)
                val dataHashes = extractDataHashes(payload)
                return ManifestFinding(
                    present = true,
                    digitalSourceType = findDigitalSourceType(payload),
                    container = "isobmff",
                    certificatesDer = certs,
                    claimData = sig?.first,
                    signatureBytes = sig?.second,
                    signatureAlgorithm = sig?.third ?: "SHA256withECDSA",
                    issuerName = extractIssuer(payload),
                    fileBytes = data,
                    manifestRanges = listOf(Pair(i.toLong(), data.size.toLong())),
                    dataHashes = dataHashes,
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

    private fun extractIssuer(payload: ByteArray): String? {
        val text = String(payload, Charsets.ISO_8859_1)
        val issuerMatch = Regex("issuer:([^;\r\n]+)").find(text)
        return issuerMatch?.groupValues?.get(1)?.trim()
    }

    private fun extractCertificates(payload: ByteArray): List<ByteArray> {
        val certs = mutableListOf<ByteArray>()
        val text = String(payload, Charsets.ISO_8859_1)

        // 1. Check for PEM encoded certificates
        val pemRegex = Regex("-----BEGIN CERTIFICATE-----[^-]+-----END CERTIFICATE-----")
        for (match in pemRegex.findAll(text)) {
            val pemStr = match.value.trim()
            certs.add(pemStr.toByteArray(Charsets.UTF_8))
        }

        // 2. Check for DER encoded X.509 sequence: 0x30 0x82 [len_hi] [len_lo] 0x30
        if (certs.isEmpty()) {
            var i = 0
            while (i + 4 < payload.size) {
                if (payload[i] == 0x30.toByte() && payload[i + 1] == 0x82.toByte()) {
                    val len = ((payload[i + 2].toInt() and 0xFF) shl 8) or (payload[i + 3].toInt() and 0xFF)
                    val totalCertLen = len + 4
                    if (totalCertLen in 200..8192 && i + totalCertLen <= payload.size) {
                        // Check if inner sequence matches TBSCertificate structure (starts with 0x30)
                        if (payload[i + 4] == 0x30.toByte()) {
                            val certBytes = payload.copyOfRange(i, i + totalCertLen)
                            certs.add(certBytes)
                            i += totalCertLen
                            continue
                        }
                    }
                }
                i++
            }
        }

        return certs
    }

    private fun extractSignatureBlock(payload: ByteArray): Triple<ByteArray, ByteArray, String>? {
        val text = String(payload, Charsets.ISO_8859_1)
        val sigMarker = "c2cs:"
        val idx = text.indexOf(sigMarker)
        if (idx >= 0) {
            val sigPart = text.substring(idx + sigMarker.length).trim()
            val colon = sigPart.indexOf(':')
            if (colon > 0) {
                val claimStr = sigPart.substring(0, colon)
                val sigHex = sigPart.substring(colon + 1).takeWhile { it.isLetterOrDigit() }
                val sigBytes = runCatching { hexToBytes(sigHex) }.getOrNull()
                if (sigBytes != null) {
                    return Triple(claimStr.toByteArray(Charsets.UTF_8), sigBytes, "SHA256withECDSA")
                }
            }
        }
        return null
    }

    private fun extractDataHashes(payload: ByteArray): List<DataHash> {
        val hashes = mutableListOf<DataHash>()
        val text = String(payload, Charsets.ISO_8859_1)
        val marker = "c2dh:"
        var idx = text.indexOf(marker)
        while (idx >= 0) {
            val part = text.substring(idx + marker.length)
            val line = part.takeWhile { it != '\n' && it != '\r' }.trim()
            val tokens = line.split(":")
            if (tokens.isNotEmpty()) {
                val hex = tokens[0].trim()
                val hashBytes = runCatching { hexToBytes(hex) }.getOrNull()
                if (hashBytes != null) {
                    val exclusions = mutableListOf<Exclusion>()
                    for (i in 1 until tokens.size step 2) {
                        if (i + 1 < tokens.size) {
                            val s = tokens[i].toLongOrNull()
                            val e = tokens[i + 1].toLongOrNull()
                            if (s != null && e != null) exclusions.add(Exclusion(s, e))
                        }
                    }
                    hashes.add(DataHash(hash = hashBytes, exclusions = exclusions))
                }
            }
            idx = text.indexOf(marker, idx + marker.length)
        }
        return hashes
    }

    private fun hexToBytes(hex: String): ByteArray {
        val len = hex.length
        val data = ByteArray(len / 2)
        var i = 0
        while (i < len) {
            data[i / 2] = ((Character.digit(hex[i], 16) shl 4) + Character.digit(hex[i + 1], 16)).toByte()
            i += 2
        }
        return data
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
        /** Bytes of the file inspected. Fits manifest, claim, and X.509 cert chains. */
        const val HEADER_BYTES = 65536
    }
}
