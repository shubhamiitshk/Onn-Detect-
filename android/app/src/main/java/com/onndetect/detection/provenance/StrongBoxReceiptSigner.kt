package com.onndetect.detection.provenance

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.security.KeyFactory
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
class StrongBoxReceiptSigner @JvmOverloads constructor(
    private val context: Context? = null,
) : ReceiptSigner {

    companion object {
        const val KEY_ALIAS = "onn_forensic_signing_key_v1"
        const val TIER_STRONGBOX = "STRONGBOX"
        const val TIER_TEE = "TEE"
        const val TIER_SOFTWARE_TEST = "SOFTWARE_TEST"
    }

    private val keyLock = Any()

    @Volatile
    private var isInitialized: Boolean = false

    /**
     * The tier is DERIVED from the key that actually exists, not a mutable
     * flag that a concurrent scan could race with.
     */
    @Volatile
    private var activeSecurityTier: String = TIER_TEE

    @Volatile
    private var testKeyPair: java.security.KeyPair? = null

    override val securityTier: String
        get() {
            ensureKeyPair()
            return activeSecurityTier
        }

    /**
     * Signs [canonicalPayloadJson] using the hardware-backed private key.
     * Generates the key pair on first invocation with StrongBox preference and TEE fallback.
     * Falls back to standard JVM EC keys in test environments where AndroidKeyStore is absent.
     */
    override fun sign(canonicalPayloadJson: String, challengeNonce: String?): ForensicReceipt {
        ensureKeyPair()
        return try {
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

            ForensicReceipt(
                canonicalPayloadJson = canonicalPayloadJson,
                signatureBase64 = Base64.getEncoder().encodeToString(signatureBytes),
                publicKeyPem = publicKeyPem,
                securityTier = activeSecurityTier,
                certificateChainBase64 = certChainBase64,
            )
        } catch (e: Exception) {
            // The software path exists ONLY where there is no AndroidKeyStore
            // at all - a JVM unit test. On a device a keystore failure is a
            // FAILURE, and it is reported as one: a receipt signed by a
            // software key must never be produced silently, because the
            // enterprise claim is that the key is hardware-isolated.
            if (androidKeystoreAvailable()) throw e
            val kp = getOrGenerateTestKeyPairLocked()

            val signer = Signature.getInstance("SHA256withECDSA")
            signer.initSign(kp.private)
            signer.update(canonicalPayloadJson.toByteArray(Charsets.UTF_8))
            val signatureBytes = signer.sign()

            ForensicReceipt(
                canonicalPayloadJson = canonicalPayloadJson,
                signatureBase64 = Base64.getEncoder().encodeToString(signatureBytes),
                publicKeyPem = ReceiptVerifier.formatPublicKeyToPem(kp.public.encoded),
                securityTier = TIER_SOFTWARE_TEST,
                certificateChainBase64 = emptyList(),
            )
        }
    }

    /**
     * Thread-safe keypair initialization with double-checked locking.
     *
     * The signing key is created ONCE and then reused.
     *
     * The attestation challenge belongs to the KEY's certificate, not to each
     * receipt: the per-receipt nonce is already inside the signed payload.
     * Regenerating the key per receipt would rotate the device's identity on
     * every scan, and every previously issued receipt would become
     * unverifiable against the current key.
     */
    private fun ensureKeyPair() {
        if (isInitialized) return
        synchronized(keyLock) {
            if (isInitialized) return

            if (!androidKeystoreAvailable()) {
                activeSecurityTier = TIER_SOFTWARE_TEST
                getOrGenerateTestKeyPairLocked()
                isInitialized = true
                return
            }

            val keyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
            if (keyStore.containsAlias(KEY_ALIAS)) {
                activeSecurityTier = tierOf(keyStore)
                isInitialized = true
                return
            }

            val hasStrongBox = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P && context != null) {
                try {
                    context.packageManager.hasSystemFeature(PackageManager.FEATURE_STRONGBOX_KEYSTORE)
                } catch (_: Exception) {
                    false
                }
            } else {
                false
            }

            if (hasStrongBox) {
                try {
                    generateEcKey(isStrongBox = true)
                    activeSecurityTier = TIER_STRONGBOX
                    isInitialized = true
                    return
                } catch (e: Exception) {
                    // StrongBox key generation failed or unsupported by driver -> Fallback to TEE
                }
            }

            // Standard hardware TEE generation
            generateEcKey(isStrongBox = false)
            activeSecurityTier = TIER_TEE
            isInitialized = true
        }
    }

    /**
     * Safely retrieves or generates the software test keypair using double-checked locking.
     */
    private fun getOrGenerateTestKeyPairLocked(): java.security.KeyPair {
        return testKeyPair ?: synchronized(keyLock) {
            testKeyPair ?: KeyPairGenerator.getInstance("EC").apply {
                initialize(ECGenParameterSpec("secp256r1"))
            }.generateKeyPair().also { testKeyPair = it }
        }
    }

    /** The tier the EXISTING key actually provides, never an assumed one. */
    private fun tierOf(keyStore: KeyStore): String = try {
        if (isStrongBoxBacked(keyStore)) TIER_STRONGBOX else TIER_TEE
    } catch (_: Exception) {
        TIER_TEE
    }

    private fun isStrongBoxBacked(keyStore: KeyStore): Boolean = try {
        val entry = keyStore.getEntry(KEY_ALIAS, null) as KeyStore.PrivateKeyEntry
        val factory = KeyFactory.getInstance(entry.privateKey.algorithm, "AndroidKeyStore")
        val keyInfo = factory.getKeySpec(entry.privateKey, android.security.keystore.KeyInfo::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            keyInfo.securityLevel == KeyProperties.SECURITY_LEVEL_STRONGBOX
        } else {
            false
        }
    } catch (_: Exception) {
        false
    }

    /** True where an AndroidKeyStore exists at all (i.e. a real device). */
    private fun androidKeystoreAvailable(): Boolean = try {
        KeyStore.getInstance("AndroidKeyStore").load(null)
        true
    } catch (_: Exception) {
        false
    }

    private fun generateEcKey(isStrongBox: Boolean) {
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

        kpg.initialize(builder.build())
        kpg.generateKeyPair()
    }
}
