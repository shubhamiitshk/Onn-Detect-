package com.onndetect.detection.provenance

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.Signature
import java.security.spec.ECGenParameterSpec
import java.util.Base64
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Signs canonical forensic scan receipts inside Android Hardware Keystore (StrongBox KeyMint or TEE).
 * Third-party verifiers (enterprise banking/insurance servers) verify the ECDSA P-256 signature
 * and validate the device's hardware attestation certificate chain against Google's Root CA.
 */
interface ReceiptSigner {
    val securityTier: String
    fun sign(canonicalPayloadJson: String, challengeNonce: String? = null): ForensicReceipt
}

@Singleton
class StrongBoxReceiptSigner @Inject constructor(
    private val context: Context,
) : ReceiptSigner {

    companion object {
        const val KEY_ALIAS = "onn_forensic_signing_key_v1"
        const val TIER_STRONGBOX = "STRONGBOX"
        const val TIER_TEE = "TEE"
    }

    private var activeSecurityTier: String = TIER_TEE

    override val securityTier: String
        get() = activeSecurityTier

    /**
     * Signs [canonicalPayloadJson] using the hardware-backed private key.
     * Generates the key pair on first invocation with StrongBox preference and TEE fallback.
     */
    override fun sign(canonicalPayloadJson: String, challengeNonce: String?): ForensicReceipt {
        ensureKeyPair(challengeNonce)

        val keyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        val privateKey = (keyStore.getEntry(KEY_ALIAS, null) as KeyStore.PrivateKeyEntry).privateKey
        val certificate = keyStore.getCertificate(KEY_ALIAS)
        val certChain = keyStore.getCertificateChain(KEY_ALIAS) ?: arrayOf(certificate)

        val signer = Signature.getInstance("SHA256withECDSA")
        signer.initSign(privateKey)
        signer.update(canonicalPayloadJson.toByteArray(Charsets.UTF_8))
        val signatureBytes = signer.sign()

        val publicKeyPem = ReceiptVerifier.formatPublicKeyToPem(certificate.publicKey.encoded)
        val certChainBase64 = certChain.map { Base64.getEncoder().encodeToString(it.encoded) }

        return ForensicReceipt(
            canonicalPayloadJson = canonicalPayloadJson,
            signatureBase64 = Base64.getEncoder().encodeToString(signatureBytes),
            publicKeyPem = publicKeyPem,
            securityTier = activeSecurityTier,
            certificateChainBase64 = certChainBase64,
        )
    }

    private fun ensureKeyPair(challengeNonce: String?) {
        val keyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        if (keyStore.containsAlias(KEY_ALIAS) && challengeNonce == null) {
            return
        }

        val hasStrongBox = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            context.packageManager.hasSystemFeature(PackageManager.FEATURE_STRONGBOX_KEYSTORE)
        } else {
            false
        }

        if (hasStrongBox) {
            try {
                generateEcKey(isStrongBox = true, challengeNonce = challengeNonce)
                activeSecurityTier = TIER_STRONGBOX
                return
            } catch (e: Exception) {
                // StrongBox key generation failed or unsupported by driver -> Fallback to TEE
            }
        }

        // Standard hardware TEE generation
        generateEcKey(isStrongBox = false, challengeNonce = challengeNonce)
        activeSecurityTier = TIER_TEE
    }

    private fun generateEcKey(isStrongBox: Boolean, challengeNonce: String?) {
        val kpg = KeyPairGenerator.getInstance(KeyProperties.KEY_ALGORITHM_EC, "AndroidKeyStore")
        val builder = KeyGenParameterSpec.Builder(
            KEY_ALIAS,
            KeyProperties.PURPOSE_SIGN or KeyProperties.PURPOSE_VERIFY,
        )
            .setAlgorithmParameterSpec(ECGenParameterSpec("secp256r1"))
            .setDigests(KeyProperties.DIGEST_SHA256)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P && isStrongBox) {
            builder.setIsStrongBoxBacked(true)
        }

        if (challengeNonce != null) {
            builder.setAttestationChallenge(challengeNonce.toByteArray(Charsets.UTF_8))
        }

        kpg.initialize(builder.build())
        kpg.generateKeyPair()
    }
}
