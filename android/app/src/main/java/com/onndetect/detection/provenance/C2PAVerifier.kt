package com.onndetect.detection.provenance

import android.content.Context
import android.net.Uri
import java.io.InputStream
import java.security.MessageDigest
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Offline C2PA "Content Credentials" signature and assertion verifier.
 *
 * ## Principles
 * 1. Offline Operation: Validates cryptographic assertions and digital source
 *    types without internet access. Revocation checks are strictly local.
 * 2. Exclusion Range Discipline (IACR 2026/804): Excluded byte ranges must lie
 *    within the manifest container hull (+/-64B slack) and cannot exceed the manifest size.
 * 3. Root Anchor Discipline: Certificates must chain to an embedded root
 *    in [C2PATrustStore]. Untrusted self-signed certificates are rejected.
 * 4. Fail-Closed Conjunction: A manifest is signed ONLY if hashVerified && coseSignatureValid && certChainValid.
 *    A certificate chain alone without a cryptographic signature over the claim is unverified.
 * 5. Categorical Evidence: Verified AI claims raise confidence to
 *    [VerdictConfidence.PROVENANCE_CONFIRMED]. Absence of C2PA never lowers
 *    statistical scores.
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

    /**
     * Verify an asset from a content Uri.
     */
    fun verify(context: Context, uri: Uri): VerificationResult {
        return try {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                verifyStream(stream)
            } ?: VerificationResult(evidence = ProvenanceEvidence.Absent)
        } catch (_: Exception) {
            VerificationResult(evidence = ProvenanceEvidence.Absent)
        }
    }

    /**
     * Verify a byte stream.
     */
    fun verifyStream(stream: InputStream): VerificationResult {
        val finding = manifestChecker.inspect(stream)
        if (!finding.present) {
            return VerificationResult(evidence = ProvenanceEvidence.Absent)
        }

        var isVerifiedSignature = false
        var issuer: String? = null
        val notes = StringBuilder()

        val parsedCerts = finding.certificatesDer.mapNotNull { trustStore.parseCertificate(it) }

        if (parsedCerts.isNotEmpty()) {
            val leaf = parsedCerts.first()
            val intermediates = if (parsedCerts.size > 1) parsedCerts.drop(1) else emptyList()

            val chainValid = trustStore.verifyChain(leaf, intermediates)
            if (chainValid) {
                val fileBytes = finding.fileBytes
                var hashVerified = true
                if (finding.dataHashes.isNotEmpty()) {
                    if (fileBytes != null && finding.manifestRanges.isNotEmpty()) {
                        hashVerified = finding.dataHashes.all { dh ->
                            val recomputed = recomputeDataHash(fileBytes, dh, finding.manifestRanges)
                            recomputed != null && dh.hash != null && recomputed.contentEquals(dh.hash)
                        }
                        if (!hashVerified) {
                            notes.append("Data-hash binding failed: asset bytes or exclusion ranges violate the signed manifest. ")
                        }
                    } else {
                        hashVerified = false
                        notes.append("Manifest carries no data-hash binding to this asset; it cannot be trusted as evidence for these bytes. ")
                    }
                }

                val sigBytes = finding.signatureBytes
                val claimData = finding.claimData
                if (sigBytes != null && claimData != null) {
                    val sigValid = trustStore.verifySignature(
                        leaf,
                        finding.signatureAlgorithm,
                        claimData,
                        sigBytes,
                    )
                    if (sigValid && hashVerified) {
                        isVerifiedSignature = true
                        issuer = leaf.subjectX500Principal.name
                        notes.append("Cryptographically verified against trusted C2PA root ($issuer). ")
                    } else if (!sigValid) {
                        notes.append("C2PA signature mismatch: claim data tampered or corrupted. ")
                    }
                } else {
                    // Fail-closed conjunction: A valid cert chain alone WITHOUT a valid signature over the claim is unverified
                    isVerifiedSignature = false
                    issuer = leaf.subjectX500Principal.name
                    notes.append("C2PA certificate chain valid ($issuer) but manifest lacks cryptographic claim signature (unverified). ")
                }
            } else {
                notes.append("Untrusted certificate authority: signer does not chain to an embedded C2PA trust anchor. ")
            }
        } else {
            notes.append("C2PA manifest found but unverified: missing X.509 certificate chain (exclusion-range safe). ")
        }

        val isAiType = finding.digitalSourceType in ProvenanceEvidence.AI_SOURCE_TYPES
        val claim = ProvenanceEvidence.C2PAClaim(
            manifestPresent = true,
            signed = isVerifiedSignature,
            digitalSourceType = finding.digitalSourceType,
            ingredientCount = if (isVerifiedSignature) 1 else 0,
        )

        if (isVerifiedSignature && isAiType) {
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
     * exclusion ranges removed, enforcing the IACR 2026/804 exclusion invariant:
     * every excluded range must sit within the manifest container hull (+/-64 B slack)
     * and total excluded bytes must not exceed manifestTotalBytes + 64.
     */
    fun recomputeDataHash(
        fileBytes: ByteArray,
        dh: C2PAManifestChecker.DataHash,
        manifestRanges: List<Pair<Long, Long>>,
    ): ByteArray? {
        if (dh.exclusions.isEmpty() || manifestRanges.isEmpty()) return null
        val hullStart = (manifestRanges.minOf { it.first } - 64L).coerceAtLeast(0L)
        val hullEnd = (manifestRanges.maxOf { it.second } + 64L).coerceAtMost(fileBytes.size.toLong())
        val maxExcludedBytes = manifestRanges.sumOf { (it.second - it.first).coerceAtLeast(0L) } + 64L
        val sorted = dh.exclusions.sortedBy { it.start }
        var prevEnd = 0L
        var totalExcluded = 0L
        val md = MessageDigest.getInstance("SHA-256")
        for (ex in sorted) {
            val start = ex.start
            val end = ex.end
            if (start < prevEnd || end <= start || end > fileBytes.size) return null
            // IACR 2026/804 invariant: exclusion must lie inside the JUMBF container hull
            if (start < hullStart || end > hullEnd) return null
            totalExcluded += (end - start)
            if (totalExcluded > maxExcludedBytes) return null
            if (start > prevEnd) md.update(fileBytes, prevEnd.toInt(), (start - prevEnd).toInt())
            prevEnd = end
        }
        // Reject degenerate exclusions that leave fewer than 64 asset bytes covered
        if (fileBytes.size - totalExcluded < 64L) return null
        if (prevEnd < fileBytes.size) md.update(fileBytes, prevEnd.toInt(), (fileBytes.size - prevEnd).toInt())
        return md.digest()
    }
}
