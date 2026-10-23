package com.onndetect.detection.provenance

import java.io.ByteArrayInputStream
import java.security.Signature
import java.security.cert.CertPathValidator
import java.security.cert.CertificateFactory
import java.security.cert.PKIXParameters
import java.security.cert.TrustAnchor
import java.security.cert.X509Certificate
import java.util.Date

/**
 * Offline trust store for verifying C2PA "Content Credentials" signer chains.
 *
 * Hard rules enforced since the F1 fix (architecture audit, commit 66fce33):
 *
 *  1. NO private keys in the source tree. A trust store that also ships a
 *     signing key lets anyone forge "verified" provenance, which is the
 *     opposite of what a trust store is for. Test fixtures generate their
 *     own ephemeral keys at RUNTIME; nothing is embedded.
 *  2. NO synthetic or self-generated roots. Anchors are loaded exclusively
 *     from [TRUST_LIST_ASSET] (`assets/trust/c2pa_trust_anchors.pem`), which
 *     carries the official production C2PA trust list (Adobe, Google,
 *     DigiCert, Irdeto, Huawei, SSL.com, ... as published by the C2PA
 *     conformance program).
 *  3. FAIL CLOSED. With an empty or unparsable trust list, [verifyChain]
 *     returns false for everything: a manifest that cannot be anchored is
 *     reported as unverified, never as signed.
 *  4. Zero network. No CRL/OCSP: certificate validity is checked against the
 *     current device clock only.
 *
 * The store stays pure-JVM (String in, no Android imports) so the whole
 * C2PA verification path is unit-testable on the desktop.
 */
class C2PATrustStore(trustListPem: String? = null) {

    private val certFactory = CertificateFactory.getInstance("X.509")
    private val trustAnchors = mutableSetOf<TrustAnchor>()
    private val trustedCertificates = mutableListOf<X509Certificate>()

    init {
        trustListPem?.let(::loadPemAnchors)
    }

    /** Parses every `-----BEGIN CERTIFICATE-----` block in [pem] into a trust anchor. */
    fun loadPemAnchors(pem: String) {
        val matcher = PEM_CERT_REGEX.findAll(pem)
        for (match in matcher) {
            parseCertificate(match.value.toByteArray(Charsets.UTF_8))?.let(::addTrustAnchor)
        }
    }

    /** Registers an X.509 certificate as a trusted root anchor. */
    fun addTrustAnchor(certificate: X509Certificate) {
        synchronized(this) {
            trustAnchors.add(TrustAnchor(certificate, null))
            trustedCertificates.add(certificate)
        }
    }

    /** Parses an X.509 certificate from PEM or DER bytes; null when unparsable. */
    fun parseCertificate(bytes: ByteArray): X509Certificate? {
        return runCatching {
            certFactory.generateCertificate(ByteArrayInputStream(bytes)) as? X509Certificate
        }.getOrNull()
    }

    /** Parses a chain of X.509 certificates from PEM or concatenated DER bytes. */
    fun parseCertificateChain(bytes: ByteArray): List<X509Certificate> {
        return runCatching {
            certFactory.generateCertificates(ByteArrayInputStream(bytes))
                .filterIsInstance<X509Certificate>()
        }.getOrDefault(emptyList())
    }

    /** True when no anchor is loaded. Everything verifies as untrusted. */
    fun isEmpty(): Boolean = synchronized(this) { trustAnchors.isEmpty() }

    /** Number of loaded anchors (diagnostics and the asset sanity test). */
    fun size(): Int = synchronized(this) { trustAnchors.size }

    /**
     * Verifies that [leaf] chains back to a loaded production anchor, offline.
     * An empty store returns false immediately (fail closed).
     */
    fun verifyChain(leaf: X509Certificate, intermediates: List<X509Certificate> = emptyList()): Boolean {
        synchronized(this) {
            if (trustAnchors.isEmpty()) return false

            // A certificate that IS an anchor is trusted directly.
            if (isDirectlyTrusted(leaf)) return true

            // The leaf is signed by an intermediate that itself chains to an anchor.
            for (inter in intermediates) {
                var interValid = isDirectlyTrusted(inter)
                if (!interValid) {
                    for (anchor in trustedCertificates) {
                        try {
                            inter.verify(anchor.publicKey)
                            interValid = true
                            break
                        } catch (_: Exception) { }
                    }
                }
                if (interValid) {
                    try {
                        leaf.verify(inter.publicKey)
                        return true
                    } catch (_: Exception) { }
                }
            }

            // Standard PKIX path validation, revocation strictly disabled.
            return try {
                val fullChain = listOf(leaf) + intermediates
                val certPath = certFactory.generateCertPath(fullChain)
                val params = PKIXParameters(trustAnchors).apply {
                    isRevocationEnabled = false // strictly offline
                    date = Date()
                }
                CertPathValidator.getInstance("PKIX").validate(certPath, params)
                true
            } catch (_: Exception) {
                false
            }
        }
    }

    /** Verifies a raw signature over [data] with the given certificate's public key. */
    fun verifySignature(
        leaf: X509Certificate,
        algorithm: String,
        data: ByteArray,
        signatureBytes: ByteArray,
    ): Boolean {
        return try {
            val sig = Signature.getInstance(algorithm)
            sig.initVerify(leaf.publicKey)
            sig.update(data)
            sig.verify(signatureBytes)
        } catch (_: Exception) {
            false
        }
    }

    /** True when the certificate itself is registered as an anchor. */
    fun isDirectlyTrusted(certificate: X509Certificate): Boolean {
        return trustedCertificates.any { trusted ->
            trusted.subjectX500Principal == certificate.subjectX500Principal &&
                trusted.publicKey == certificate.publicKey
        }
    }

    companion object {
        /** Asset path of the production trust list inside the APK. */
        const val TRUST_LIST_ASSET = "trust/c2pa_trust_anchors.pem"

        private val PEM_CERT_REGEX =
            Regex("-----BEGIN CERTIFICATE-----[\\s\\S]*?-----END CERTIFICATE-----")
    }
}
