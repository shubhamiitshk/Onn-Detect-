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
)

/**
 * Platform-independent verifier for ONN-DETECT forensic receipts.
 * Can be run client-side, in JVM unit tests, or on an enterprise backend.
 */
object ReceiptVerifier {

    private const val ALGORITHM = "SHA256withECDSA"

    /**
     * Verifies the ECDSA P-256 signature against the canonical JSON payload and public key.
     */
    fun verify(receipt: ForensicReceipt): Boolean {
        return verify(
            canonicalPayloadJson = receipt.canonicalPayloadJson,
            signatureBase64 = receipt.signatureBase64,
            publicKeyPem = receipt.publicKeyPem,
        )
    }

    fun verify(canonicalPayloadJson: String, signatureBase64: String, publicKeyPem: String): Boolean {
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
