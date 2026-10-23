package com.onndetect.detection.provenance

import org.bouncycastle.asn1.x500.X500Name
import org.bouncycastle.asn1.x509.BasicConstraints
import org.bouncycastle.asn1.x509.Extension
import org.bouncycastle.asn1.x509.KeyUsage
import org.bouncycastle.cert.jcajce.JcaX509CertificateConverter
import org.bouncycastle.cert.jcajce.JcaX509v3CertificateBuilder
import org.bouncycastle.operator.jcajce.JcaContentSignerBuilder
import java.io.ByteArrayOutputStream
import java.math.BigInteger
import java.security.KeyPair
import java.security.KeyPairGenerator
import java.security.SecureRandom
import java.security.Signature
import java.security.cert.X509Certificate
import java.security.spec.ECGenParameterSpec
import java.util.Base64

/**
 * JVM test fixtures that build REAL, spec-shaped C2PA structures with keys
 * generated at RUNTIME (F1: no embedded key material anywhere, including
 * tests). Everything here is test-only; production code never links it.
 */
object C2paTestFixtures {

    // ─────────────────────────────────────────────────────────────────────
    // Runtime CA
    // ─────────────────────────────────────────────────────────────────────

    class TestCa(val keyPair: KeyPair, val certificate: X509Certificate) {
        fun pem(): String = toPem(certificate.encoded)
        override fun toString() = certificate.subjectX500Principal.name
    }

    /** Generates an ephemeral EC P-256 self-signed CA, like the C2PA conformance roots. */
    fun newCa(cn: String = "ONN Test C2PA Root"): TestCa {
        val kpg = KeyPairGenerator.getInstance("EC")
        kpg.initialize(ECGenParameterSpec("secp256r1"))
        val kp = kpg.generateKeyPair()
        val name = X500Name("CN=$cn, O=ONN Detect Tests, C=IN")
        val notBefore = java.util.Date(System.currentTimeMillis() - 86400_000L)
        val notAfter = java.util.Date(System.currentTimeMillis() + 3650L * 86400_000L)
        val serial = BigInteger(63, SecureRandom())
        val builder = JcaX509v3CertificateBuilder(name, serial, notBefore, notAfter, name, kp.public)
            .addExtension(Extension.basicConstraints, true, BasicConstraints(true))
            .addExtension(Extension.keyUsage, true, KeyUsage(KeyUsage.keyCertSign or KeyUsage.cRLSign))
        val holder = builder.build(JcaContentSignerBuilder("SHA256withECDSA").build(kp.private))
        return TestCa(kp, JcaX509CertificateConverter().getCertificate(holder))
    }

    /** Signs a leaf certificate under [ca] (for intermediate-chain tests). */
    fun newLeafSignedBy(ca: TestCa, cn: String = "ONN Test Claim Signing"): X509Certificate {
        val kpg = KeyPairGenerator.getInstance("EC")
        kpg.initialize(ECGenParameterSpec("secp256r1"))
        val kp = kpg.generateKeyPair()
        val subject = X500Name("CN=$cn, O=ONN Detect Tests, C=IN")
        val issuer = X500Name(ca.certificate.subjectX500Principal.name)
        val notBefore = java.util.Date(System.currentTimeMillis() - 86400_000L)
        val notAfter = java.util.Date(System.currentTimeMillis() + 825L * 86400_000L)
        val builder = JcaX509v3CertificateBuilder(issuer, BigInteger(63, SecureRandom()), notBefore, notAfter, subject, kp.public)
            .addExtension(Extension.basicConstraints, true, BasicConstraints(false))
        val holder = builder.build(JcaContentSignerBuilder("SHA256withECDSA").build(ca.keyPair.private))
        return JcaX509CertificateConverter().getCertificate(holder)
    }

    fun toPem(der: ByteArray): String =
        "-----BEGIN CERTIFICATE-----\n" +
            Base64.getMimeEncoder(64, "\n".toByteArray()).encodeToString(der) +
            "\n-----END CERTIFICATE-----\n"

    // ─────────────────────────────────────────────────────────────────────
    // Minimal CBOR encoder (test-side; production only decodes + Sig_structure)
    // ─────────────────────────────────────────────────────────────────────

    object Enc {
        fun uint(v: Long): ByteArray = when {
            v < 24 -> byteArrayOf((0x00 or v.toInt()).toByte())
            v < 256 -> byteArrayOf(0x18, v.toByte())
            v < 65536 -> byteArrayOf(0x19, ((v shr 8) and 0xFF).toByte(), (v and 0xFF).toByte())
            else -> byteArrayOf(0x1A, ((v shr 24) and 0xFF).toByte(), ((v shr 16) and 0xFF).toByte(), ((v shr 8) and 0xFF).toByte(), (v and 0xFF).toByte())
        }

        fun nint(v: Long): ByteArray { // v is the negative value
            val n = -1L - v
            return when {
                n < 24 -> byteArrayOf((0x20 or n.toInt()).toByte())
                n < 256 -> byteArrayOf(0x38, n.toByte())
                else -> byteArrayOf(0x39, ((n shr 8) and 0xFF).toByte(), (n and 0xFF).toByte())
            }
        }

        fun tstr(s: String): ByteArray {
            val body = s.toByteArray(Charsets.UTF_8)
            val head = when {
                body.size < 24 -> byteArrayOf((0x60 or body.size).toByte())
                body.size < 256 -> byteArrayOf(0x78, body.size.toByte())
                else -> byteArrayOf(0x79, ((body.size shr 8) and 0xFF).toByte(), (body.size and 0xFF).toByte())
            }
            return head + body
        }

        fun bstr(b: ByteArray): ByteArray {
            val head = when {
                b.size < 24 -> byteArrayOf((0x40 or b.size).toByte())
                b.size < 256 -> byteArrayOf(0x58, b.size.toByte())
                else -> byteArrayOf(0x59, ((b.size shr 8) and 0xFF).toByte(), (b.size and 0xFF).toByte())
            }
            return head + b
        }

        fun arr(items: List<ByteArray>): ByteArray {
            val head = when {
                items.size < 24 -> byteArrayOf((0x80 or items.size).toByte())
                else -> byteArrayOf(0x98.toByte(), items.size.toByte())
            }
            return head + items.fold(ByteArray(0)) { acc, i -> acc + i }
        }

        fun map(entries: List<Pair<ByteArray, ByteArray>>): ByteArray {
            val head = when {
                entries.size < 24 -> byteArrayOf((0xA0 or entries.size).toByte())
                else -> byteArrayOf(0xB8.toByte(), entries.size.toByte())
            }
            return head + entries.fold(ByteArray(0)) { acc, (k, v) -> acc + k + v }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Claim + COSE_Sign1
    // ─────────────────────────────────────────────────────────────────────

    /** A CBOR claim with a digital source type and a generator string. */
    fun buildClaim(digitalSourceType: String = "trainedAlgorithmicMedia"): ByteArray = Enc.map(
        listOf(
            Enc.tstr("claim_generator") to Enc.tstr("onn_test/1.0"),
            Enc.tstr("digital_source_type") to Enc.tstr(digitalSourceType),
            Enc.tstr("claim_format") to Enc.map(
                listOf(Enc.tstr("c2pa_states") to Enc.arr(listOf(Enc.tstr("c2pa.hash.data"))))
            ),
        )
    )

    /**
     * Signs [claim] with [ca]: ES256 (alg -7) with the x5chain in the
     * protected header (label 33), exactly like a C2PA claim signature.
     */
    fun buildCoseSign1(claim: ByteArray, ca: TestCa, asLeafOf: X509Certificate? = null): ByteArray {
        val signerCert = asLeafOf ?: ca.certificate
        val protected = Enc.map(
            listOf(
                Enc.uint(1) to Enc.nint(-7), // alg: ES256
                Enc.uint(33) to Enc.arr(listOf(Enc.bstr(signerCert.encoded))), // x5chain
            )
        )
        val toBeSigned = Cbor.encodeSigStructure(protected, ByteArray(0), claim)
        val dsa = Signature.getInstance("SHA256withECDSA")
        dsa.initSign(ca.keyPair.private)
        dsa.update(toBeSigned)
        val der = dsa.sign()
        val raw = CoseSign1.derToRaw(der, 32) ?: error("fixture DER->raw conversion failed")
        return byteArrayOf(0xD2.toByte()) + Enc.arr(
            listOf(Enc.bstr(protected), Enc.map(emptyList()), Enc.bstr(claim), Enc.bstr(raw))
        )
    }

    // ─────────────────────────────────────────────────────────────────────
    // JUMBF store bytes
    // ─────────────────────────────────────────────────────────────────────

    private fun box(type: String, content: ByteArray): ByteArray {
        val out = ByteArrayOutputStream()
        val size = 8 + content.size
        out.write((size shr 24) and 0xFF); out.write((size shr 16) and 0xFF)
        out.write((size shr 8) and 0xFF); out.write(size and 0xFF)
        out.write(type.toByteArray(Charsets.US_ASCII))
        out.write(content)
        return out.toByteArray()
    }

    /** Manifest store: 'c2pa' superbox containing claim/COSE/certs (+ optional extra boxes). */
    fun buildStore(
        claim: ByteArray,
        cose: ByteArray,
        certsDer: List<ByteArray>,
        extraBoxes: List<Pair<String, ByteArray>> = emptyList(),
        storeType: String = "c2pa",
    ): ByteArray {
        var inner = box("clai", claim) + box("cose", cose)
        if (certsDer.isNotEmpty()) inner += box("cert", certsDer.fold(ByteArray(0)) { a, c -> a + c })
        for ((t, c) in extraBoxes) inner += box(t, c)
        return box(storeType, inner)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Containers
    // ─────────────────────────────────────────────────────────────────────

    class JpegLayout(val bytes: ByteArray, val manifestStart: Long, val manifestEnd: Long) {
        fun nonManifestBytes(): ByteArray {
            val out = ByteArrayOutputStream()
            out.write(bytes, 0, manifestStart.toInt())
            out.write(bytes, manifestEnd.toInt(), bytes.size - manifestEnd.toInt())
            return out.toByteArray()
        }
    }

    /**
     * Embeds [store] into a JPEG as multi-segment APP11 (ISO 19566-5 D.2):
     * "JUMBF" + En + Z per segment, fragments of the same En contiguous,
     * segments ≤ [maxSegPayload] bytes each — plus a JFIF APP0 in front so
     * the walker must skip real segments, and filler APP0 segments totalling
     * [prefixPadBytes] to push the manifest past the old 64 KB scan window.
     */
    fun jpegWithApp11(
        store: ByteArray,
        maxSegPayload: Int = 64,
        prefixPadBytes: Int = 0,
        trailingSuffix: ByteArray = byteArrayOf(0xFF.toByte(), 0xD9.toByte()), // EOI
    ): JpegLayout {
        val prefix = ByteArrayOutputStream()
        prefix.write(byteArrayOf(0xFF.toByte(), 0xD8.toByte())) // SOI
        // JFIF APP0 (18 bytes) — a real segment the walker must skip by length.
        prefix.write(byteArrayOf(
            0xFF.toByte(), 0xE0.toByte(), 0x00, 0x10,
            0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00
        ))
        // Filler APP0 segments so the manifest starts beyond [prefixPadBytes].
        var padded = 0
        while (padded < prefixPadBytes) {
            val take = minOf(65533 - 4, prefixPadBytes - padded).coerceAtLeast(4)
            prefix.write(byteArrayOf(0xFF.toByte(), 0xE0.toByte()))
            prefix.write(((take + 2) shr 8) and 0xFF); prefix.write((take + 2) and 0xFF)
            prefix.write(ByteArray(take)) // zero payload
            padded += take + 4
        }

        val out = ByteArrayOutputStream()
        out.write(prefix.toByteArray())
        val prefixLen = out.size()

        // Split the store into segments of at most [maxSegPayload] fragment bytes.
        val fragments = store.toList().chunked(maxSegPayload.coerceAtLeast(8))
        val en = 1
        fragments.forEachIndexed { index, fragment ->
            val z = if (index == fragments.lastIndex) 0 else 1
            val payload = ByteArray(7 + fragment.size)
            "JUMBF".toByteArray(Charsets.US_ASCII).copyInto(payload, 0)
            payload[5] = en.toByte()
            payload[6] = z.toByte()
            fragment.toByteArray().copyInto(payload, 7)
            out.write(byteArrayOf(0xFF.toByte(), 0xEB.toByte()))
            val lp = payload.size + 2
            out.write((lp shr 8) and 0xFF); out.write(lp and 0xFF)
            out.write(payload)
        }
        val manifestEnd = out.size()
        out.write(trailingSuffix)
        return JpegLayout(out.toByteArray(), prefixLen.toLong(), manifestEnd.toLong())
    }

    /** An ISO-BMFF file with the store in a 'uuid' box (16-byte 'c2pa'-prefixed ext type). */
    fun bmffWithUuidStore(store: ByteArray, noiseInMdat: Boolean = true): ByteArray {
        val out = ByteArrayOutputStream()
        fun boxBytes(type: String, content: ByteArray) {
            out.write(((content.size + 8) shr 24) and 0xFF); out.write(((content.size + 8) shr 16) and 0xFF)
            out.write(((content.size + 8) shr 8) and 0xFF); out.write((content.size + 8) and 0xFF)
            out.write(type.toByteArray(Charsets.US_ASCII))
            out.write(content)
        }
        boxBytes("ftyp", byteArrayOf(0, 0, 0, 0) + "isom".toByteArray() + ByteArray(8))
        // uuid box: 16-byte extended type beginning with the 'c2pa' fourcc.
        val extType = "c2pa".toByteArray(Charsets.US_ASCII) + ByteArray(12) + store
        boxBytes("uuid", extType)
        // mdat containing decoy 'c2pa' text + fake APP11-looking bytes: must be SKIPPED whole.
        val mdat = ("c2pa" + "JUMBF" + "trainedAlgorithmicMedia".repeat(20)).toByteArray(Charsets.ISO_8859_1)
        boxBytes("mdat", if (noiseInMdat) mdat else ByteArray(16))
        boxBytes("moov", ByteArray(8))
        return out.toByteArray()
    }

    // ─────────────────────────────────────────────────────────────────────
    // Data-hash assertion JSON (C2PA 1.x shape; verifier accepts 2.x CBOR too)
    // ─────────────────────────────────────────────────────────────────────

    fun dataHashJson(exclusions: List<Pair<Long, Long>>, digest: ByteArray): String {
        val hex = digest.joinToString("") { "%02x".format(it) }
        val ex = exclusions.joinToString(",") { """{"start":${it.first},"end":${it.second}}""" }
        return """{"exclusions":[$ex],"alg":"sha256","hash":"$hex"}"""
    }

    fun sha256(bytes: ByteArray): ByteArray =
        java.security.MessageDigest.getInstance("SHA-256").digest(bytes)
}
