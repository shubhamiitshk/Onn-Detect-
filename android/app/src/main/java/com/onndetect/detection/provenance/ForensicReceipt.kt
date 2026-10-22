package com.onndetect.detection.provenance

import java.security.KeyFactory
import java.security.PublicKey
import java.security.Signature
import java.security.spec.X509EncodedKeySpec
import java.util.Base64

/**
 * A hardware-attested, cryptographically verifiable forensic scan receipt.
 */
data class ForensicReceipt(
    val canonicalPayloadJson: String,
    val signatureBase64: String,
    val publicKeyPem: String,
    val securityTier: String, // "STRONGBOX", "TEE", or "SOFTWARE_TEST"
    val certificateChainBase64: List<String> = emptyList(),
) {
    fun toExportJson(): String {
        val root = org.json.JSONObject()
        root.put("schema_version", CanonicalReceiptPayload.SCHEMA_VERSION)
        root.put("security_tier", securityTier)
        root.put("signature_algorithm", "SHA256withECDSA")
        root.put("canonical_payload", org.json.JSONObject(canonicalPayloadJson))
        root.put("signature_base64", signatureBase64)
        root.put("public_key_pem", publicKeyPem)
        val chain = org.json.JSONArray()
        certificateChainBase64.forEach { chain.put(it) }
        root.put("certificate_chain", chain)
        return root.toString(2)
    }
}

/**
 * Platform-independent verifier for ONN-DETECT forensic receipts.
 * Can be run client-side, in JVM unit tests, or on an enterprise backend.
 */
object ReceiptVerifier {

    private const val ALGORITHM = "SHA256withECDSA"

    /** The tiers that prove the signing key was hardware-isolated. */
    private val HARDWARE_TIERS = setOf("STRONGBOX", "TEE")

    /**
     * Verifies a receipt the way an enterprise server must.
     *
     * Three things are required, and the signature alone is NOT one of them:
     *
     *  1. the ECDSA P-256 signature verifies against the canonical payload;
     *  2. the receipt carries a hardware-isolated tier (StrongBox or TEE);
     *  3. the receipt carries the device's ATTESTATION CHAIN.
     *
     * Point 2 and 3 are what a receipt from a rooted phone with a software
     * key fails, and they are exactly what the enterprise claim rests on. A
     * verifier that only checks the signature accepts a forged receipt,
     * because the receipt carries the public key it would verify against.
     *
     * Validating the CHAIN against Google's hardware root CA is the
     * verifier's next step and is deliberately not implemented here: it
     * needs the attestation root list, the app's signing certificate and a
     * revocation check, all of which belong on the server.
     */
    fun verify(receipt: ForensicReceipt): Boolean {
        if (receipt.securityTier !in HARDWARE_TIERS) return false
        if (receipt.certificateChainBase64.isEmpty()) return false
        return verifySignature(
            canonicalPayloadJson = receipt.canonicalPayloadJson,
            signatureBase64 = receipt.signatureBase64,
            publicKeyPem = receipt.publicKeyPem,
        )
    }

    /**
     * Verifies ONLY the signature.
     *
     * This proves that the payload was not altered after signing. It does
     * NOT prove that the signer was a real device: the public key travels
     * with the receipt. Use [verify] for an acceptance decision.
     */
    fun verifySignature(receipt: ForensicReceipt): Boolean = verifySignature(
        canonicalPayloadJson = receipt.canonicalPayloadJson,
        signatureBase64 = receipt.signatureBase64,
        publicKeyPem = receipt.publicKeyPem,
    )

    fun verifySignature(canonicalPayloadJson: String, signatureBase64: String, publicKeyPem: String): Boolean {
        return runCatching {
            val publicKey = parsePublicKey(publicKeyPem)
            val signatureBytes = Base64.getDecoder().decode(signatureBase64)
            val payloadBytes = canonicalPayloadJson.toByteArray(Charsets.UTF_8)

            val signer = Signature.getInstance(ALGORITHM)
            signer.initVerify(publicKey)
            signer.update(payloadBytes)
            signer.verify(signatureBytes)
        }.getOrDefault(false)
    }

    /**
     * Parses an EC public key in standard X.509 PEM or raw Base64 format.
     */
    fun parsePublicKey(publicKeyPem: String): PublicKey {
        val clean = publicKeyPem
            .replace("-----BEGIN PUBLIC KEY-----", "")
            .replace("-----END PUBLIC KEY-----", "")
            .replace("\\s".toRegex(), "")
        val decoded = Base64.getDecoder().decode(clean)
        val spec = X509EncodedKeySpec(decoded)
        val kf = KeyFactory.getInstance("EC")
        return kf.generatePublic(spec)
    }

    /**
     * Formats an encoded public key into standard PEM format.
     */
    fun formatPublicKeyToPem(keyBytes: ByteArray): String {
        val b64 = Base64.getEncoder().encodeToString(keyBytes)
        val chunks = b64.chunked(64).joinToString("\n")
        return "-----BEGIN PUBLIC KEY-----\n$chunks\n-----END PUBLIC KEY-----"
    }
}
