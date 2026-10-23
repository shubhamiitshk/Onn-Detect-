package com.onndetect.detection.provenance

import java.security.KeyFactory
import java.security.PublicKey
import java.security.Signature
import java.security.spec.ECPoint
import java.security.spec.ECPublicKeySpec
import java.security.spec.PKCS8EncodedKeySpec
import java.security.spec.X509EncodedKeySpec

/**
 * COSE_Sign1 (RFC 9052 §4.2) parsing and verification for offline C2PA
 * claim signatures.
 *
 * Structure: tag 18 (0xD2) wrapping a 4-element array
 *   [ protected: bstr, unprotected: map, payload: bstr, signature: bstr ]
 *
 * The signature is computed over the RFC 9052 Sig_structure
 *   ["Signature1", protected, external_aad (empty for C2PA), payload]
 * and, for ES256 (alg -7, P-256), is transported as the raw 64-byte
 * r||s pair — which JCA expects in DER SEQUENCE form, hence the converters.
 *
 * Pure JVM: unit-testable on the desktop.
 */
object CoseSign1 {

    data class Parsed(
        val protected: ByteArray,
        val payload: ByteArray,
        val signature: ByteArray,
        /** COSE algorithm label, e.g. -7 (ES256), -35 (ES384), -257 (RS256). */
        val alg: Long?,
        /** x5chain (label 33): DER certificates, leaf first. */
        val x5chain: List<ByteArray>,
    ) {
        override fun equals(other: Any?) = this === other
        override fun hashCode() = System.identityHashCode(this)
    }

    /** Algorithm labels (RFC 9052 §4) mapped to JCA signature names. */
    private val ALG_JCA: Map<Long, String> = mapOf(
        -7L to "SHA256withECDSA",   // ES256
        -35L to "SHA384withECDSA",  // ES384
        -47L to "SHA512withECDSA",  // ES512
        -257L to "SHA256withRSA",   // RS256
        -258L to "SHA384withRSA",   // RS384
        -259L to "SHA512withRSA",   // RS512
    )

    /** Parses a COSE_Sign1 from [bytes]; tolerant of a missing tag byte and trailing padding. */
    fun parse(bytes: ByteArray): Parsed? {
        var cursor = 0
        // Byte.toInt() sign-extends: 0xD2 is -46 as a Byte, so mask before compare.
        val first = (bytes.getOrNull(0)?.toInt() ?: return null) and 0xFF
        if (first == 0xD2) cursor = 1 // COSE_Sign1 tag 18
        val decoded = Cbor.decodeAt(bytes, cursor)?.first ?: return null
        val arr = decoded as? Cbor.Value.Arr ?: return null
        if (arr.items.size < 4) return null
        val protected = (arr.items[0] as? Cbor.Value.BStr)?.bytes ?: return null
        val payload = (arr.items[2] as? Cbor.Value.BStr)?.bytes ?: return null
        val signature = (arr.items[3] as? Cbor.Value.BStr)?.bytes ?: return null
        if (signature.isEmpty()) return null

        var alg: Long? = null
        val x5chain = mutableListOf<ByteArray>()
        val protectedValue = Cbor.decode(protected) as? Cbor.Value.Map
        if (protectedValue != null) {
            (protectedValue.get(1L) as? Cbor.Value.NInt)?.let { alg = it.value }
            (protectedValue.get(1L) as? Cbor.Value.UInt)?.let { alg = it.value }
            when (val chain = protectedValue.get(33L)) { // x5chain
                is Cbor.Value.Arr -> chain.items.forEach { item ->
                    (item as? Cbor.Value.BStr)?.let { x5chain += it.bytes }
                }
                is Cbor.Value.BStr -> x5chain += chain.bytes
                else -> {}
            }
        }
        return Parsed(protected, payload, signature, alg, x5chain)
    }

    /**
     * Verifies the COSE signature over [payload] using [publicKey] and the
     * algorithm declared in the protected header. Empty external AAD (C2PA).
     */
    fun verify(parsed: Parsed, publicKey: PublicKey, payload: ByteArray = parsed.payload): Boolean {
        val jcaAlg = ALG_JCA[parsed.alg] ?: return false
        return try {
            val toBeSigned = Cbor.encodeSigStructure(parsed.protected, ByteArray(0), payload)
            var sigBytes = parsed.signature
            if (jcaAlg.endsWith("ECDSA")) {
                // COSE transports r||s; JCA wants DER. 64-byte P-256 case handled;
                // for P-384/P-521 raw lengths of 96/132 bytes are handled the same way.
                es256RawToDer(sigBytes)?.let { sigBytes = it } ?: return false
            }
            val verifier = Signature.getInstance(jcaAlg)
            verifier.initVerify(publicKey)
            verifier.update(toBeSigned)
            verifier.verify(sigBytes)
        } catch (_: Exception) {
            false
        }
    }

    /** The RFC 9052 ToBeSigned bytes for this parsed object (exposed for tests). */
    fun toBeSigned(parsed: Parsed, externalAad: ByteArray = ByteArray(0)): ByteArray =
        Cbor.encodeSigStructure(parsed.protected, externalAad, parsed.payload)

    // ─────────────────────────────────────────────────────────────────────
    // Raw (r||s) <-> DER conversion for ECDSA signatures.
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Converts a raw COSE ECDSA signature (fixed-width r||s, each component
     * half the total length: 64 for P-256, 96 for P-384) into the DER
     * SEQUENCE form JCA verifies. Returns null for shapes that are not a
     * raw COSE pair (they may already be DER).
     */
    fun es256RawToDer(raw: ByteArray): ByteArray? {
        if (raw.size < 16 || raw.size % 2 != 0) return null
        val half = raw.size / 2
        val r = stripLeadingZeros(raw.copyOfRange(0, half))
        val s = stripLeadingZeros(raw.copyOfRange(half, raw.size))
        val out = java.io.ByteArrayOutputStream()
        out.write(0x30)
        out.write(2 + r.size + 2 + s.size)
        out.write(0x02); out.write(r.size); out.write(r)
        out.write(0x02); out.write(s.size); out.write(s)
        return out.toByteArray()
    }

    /** Converts a DER ECDSA signature into the raw r||s pair of [coordLen] bytes per coordinate. */
    fun derToRaw(der: ByteArray, coordLen: Int = 32): ByteArray? {
        return try {
            var idx = 0
            if (der[idx] != 0x30.toByte()) return null
            idx += 1
            if (der[idx] == 0x81.toByte()) idx += 2 else idx += 1
            if (der[idx] != 0x02.toByte()) return null
            idx += 1
            val rLen = der[idx].toInt() and 0xFF; idx += 1
            val r = stripLeadingZeros(der.copyOfRange(idx, idx + rLen)); idx += rLen
            if (der[idx] != 0x02.toByte()) return null
            idx += 1
            val sLen = der[idx].toInt() and 0xFF; idx += 1
            val s = stripLeadingZeros(der.copyOfRange(idx, idx + sLen)); idx += sLen
            val out = ByteArray(2 * coordLen)
            System.arraycopy(r, 0, out, coordLen - r.size, r.size)
            System.arraycopy(s, 0, out, 2 * coordLen - s.size, s.size)
            out
        } catch (_: Exception) {
            null
        }
    }

    private fun stripLeadingZeros(bytes: ByteArray): ByteArray {
        var start = 0
        while (start < bytes.size - 1 && bytes[start] == 0.toByte()) start++
        return if (start == 0) bytes else bytes.copyOfRange(start, bytes.size)
    }
}
