package com.onndetect.detection.provenance

import android.content.Context
import android.net.Uri
import java.io.InputStream
import java.security.MessageDigest
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Offline C2PA "Content Credentials" claim-signature verifier — F2 rewrite.
 *
 * The signed state is a FAIL-CLOSED CONJUNCTION of three independent checks:
 *
 *   signed = hashVerified && coseSignatureValid && certChainValid
 *
 *  - [hashVerified]: every c2pa.hash.data assertion recomputes over the
 *    actual asset bytes (minus the declared exclusions). This is the hard
 *    binding that stops manifest transplants: a manifest lifted from a
 *    genuine file onto a doctored one fails here.
 *  - [coseSignatureValid]: the COSE_Sign1 (RFC 9052) signature verifies over
 *    the claim payload with the signer's public key, ES256 raw r||s handled.
 *  - [certChainValid]: the signer certificate chains to a PRODUCTION anchor
 *    in the offline [C2PATrustStore] (official C2PA trust list; revocation
 *    strictly disabled, zero network).
 *
 * A certificate chain that validates on its own maps to an EXPLICIT
 * UNVERIFIED state — never to signed=true. Chain presence proves nothing
 * about the bytes; the pre-audit code that trusted it (defect F2, commit
 * 66fce33) let any manifest carrying a parseable chain through as
 * "verified", which is the definition of a spoofable provenance channel.
 */
@Singleton
class C2PAVerifier @Inject constructor(
    private val manifestChecker: C2PAManifestChecker,
    private val trustStore: C2PATrustStore = C2PATrustStore(),
) {

    data class VerificationResult(
        val evidence: ProvenanceEvidence,
        val issuer: String? = null,
        val validationNotes: String? = null,
    )

    /** Verify an asset from a content Uri. */
    fun verify(context: Context, uri: Uri): VerificationResult {
        return try {
            context.contentResolver.openInputStream(uri)?.use { stream -> verifyStream(stream) }
                ?: VerificationResult(evidence = ProvenanceEvidence.Absent)
        } catch (_: Exception) {
            VerificationResult(evidence = ProvenanceEvidence.Absent)
        }
    }

    /** Verify a byte stream. */
    fun verifyStream(stream: InputStream): VerificationResult {
        val finding = manifestChecker.inspect(stream)
        if (!finding.present) {
            return VerificationResult(evidence = ProvenanceEvidence.Absent)
        }

        val notes = StringBuilder()
        var issuer: String? = null
        var certChainValid = false
        var coseSignatureValid = false
        var hashVerified = false

        if (trustStore.isEmpty()) {
            notes.append("C2PA trust list not loaded; nothing can be verified (fail closed). ")
        }

        // ── 1. Signer certificate chain ───────────────────────────────────
        val coseParsed = finding.coseBytes?.let(CoseSign1::parse)
        val chainBytes = buildList {
            coseParsed?.x5chain?.let(::addAll)   // x5chain from the protected header, leaf first
            addAll(finding.certificatesDer)      // and/or the 'cert' box
        }
        val parsedCerts = chainBytes.mapNotNull { trustStore.parseCertificate(it) }
        var leaf: java.security.cert.X509Certificate? = null
        if (parsedCerts.isNotEmpty()) {
            leaf = parsedCerts.first()
            issuer = leaf.subjectX500Principal.name
            certChainValid = trustStore.verifyChain(leaf, parsedCerts.drop(1))
            if (!certChainValid) {
                notes.append("Signer certificate does not chain to a production C2PA trust anchor ($issuer). ")
            }
        } else {
            notes.append("No X.509 signer certificate found in the manifest. ")
        }

        // ── 2. COSE_Sign1 signature over the claim payload ────────────────
        if (coseParsed != null && leaf != null) {
            // The signed claim is the COSE payload; a divergent 'clai' box
            // is ignored in favour of the bytes that were actually signed.
            coseSignatureValid = CoseSign1.verify(coseParsed, leaf.publicKey)
            if (!coseSignatureValid) {
                notes.append("COSE_Sign1 signature invalid over the claim payload. ")
            }
        } else if (coseParsed == null) {
            notes.append("No COSE_Sign1 signature found in the manifest. ")
        }

        // ── 3. Data-hash hard binding to these exact asset bytes ─────────
        val fileBytes = finding.fileBytes
        if (finding.dataHashes.isNotEmpty() && fileBytes != null) {
            hashVerified = finding.dataHashes.all { dh ->
                val recomputed = recomputeDataHash(fileBytes, dh, finding.manifestRanges)
                recomputed != null && dh.hash != null && recomputed.contentEquals(dh.hash)
            }
            if (!hashVerified) {
                notes.append("Data-hash binding failed: asset bytes or exclusion ranges do not match the signed manifest (possible transplant or edit). ")
            }
        } else {
            notes.append("Manifest carries no data-hash binding to this asset; it cannot be trusted as evidence for these bytes. ")
        }

        // ── F2 fail-closed conjunction ────────────────────────────────────
        val signed = hashVerified && coseSignatureValid && certChainValid
        if (signed) {
            notes.insert(0, "Verified against the production C2PA trust list (chain + COSE_Sign1 + data-hash, fail-closed conjunction). ")
        } else if (certChainValid) {
            notes.insert(0, "Certificate chain anchored to a production C2PA trust root, but the manifest stays UNVERIFIED: ")
        }

        val isAiType = finding.digitalSourceType in ProvenanceEvidence.AI_SOURCE_TYPES
        val claim = ProvenanceEvidence.C2PAClaim(
            manifestPresent = true,
            signed = signed,
            digitalSourceType = finding.digitalSourceType,
            ingredientCount = if (signed) finding.dataHashes.size.coerceAtLeast(1) else 0,
        )
        if (signed && isAiType) {
            notes.append("Confirmed AI generation source (${finding.digitalSourceType}).")
        }

        return VerificationResult(
            evidence = claim,
            issuer = issuer,
            validationNotes = notes.toString().trim(),
        )
    }

    /**
     * Recomputes a c2pa.hash.data digest over [fileBytes] with the declared
     * exclusion ranges removed, in file order, enforcing the IACR ePrint 2026/804
     * exclusion-range invariant: every excluded range must lie inside the
     * manifest store's own container hull (±64 B slack), total excluded bytes
     * must not exceed manifestTotalBytes + 64, and at least 64 asset bytes must
     * remain covered by the digest.
     */
    private fun recomputeDataHash(
        fileBytes: ByteArray,
        dh: C2PAManifestChecker.DataHash,
        manifestRanges: List<Pair<Long, Long>> = emptyList(),
    ): ByteArray? {
        if (dh.exclusions.isEmpty()) return runCatching {
            MessageDigest.getInstance("SHA-256").digest(fileBytes)
        }.getOrNull()
        if (manifestRanges.isEmpty()) return null
        val hullStart = (manifestRanges.minOf { it.first } - 64L).coerceAtLeast(0L)
        val hullEnd = (manifestRanges.maxOf { it.second } + 64L).coerceAtMost(fileBytes.size.toLong())
        val maxExcludedBytes = manifestRanges.sumOf { (it.second - it.first).coerceAtLeast(0L) } + 64L

        val sorted = dh.exclusions.sortedBy { it.first }
        var prevEnd = 0L
        var totalExcluded = 0L
        val md = MessageDigest.getInstance("SHA-256")
        for ((start, end) in sorted) {
            if (start < prevEnd || end <= start || end > fileBytes.size) return null
            if (start < hullStart || end > hullEnd) return null
            totalExcluded += (end - start)
            if (totalExcluded > maxExcludedBytes) return null
            if (start > prevEnd) md.update(fileBytes, prevEnd.toInt(), (start - prevEnd).toInt())
            prevEnd = end
        }
        if (fileBytes.size - totalExcluded < 64L) return null
        if (prevEnd < fileBytes.size) md.update(fileBytes, prevEnd.toInt(), (fileBytes.size - prevEnd).toInt())
        return md.digest()
    }
}
