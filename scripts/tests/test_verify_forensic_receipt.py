from __future__ import annotations

import base64
import datetime
import hashlib
import unittest

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from scripts.verify_forensic_receipt import (
    serialize_canonical_payload,
    verify_receipt_bundle,
)


class VerifyForensicReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self.private_key.public_key()
        self.public_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ONN-DETECT StrongBox Key")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(self.public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
            .sign(self.private_key, hashes.SHA256())
        )
        self.cert_b64 = base64.b64encode(cert.public_bytes(serialization.Encoding.DER)).decode("ascii")
        self.media_bytes = b"authentic-camera-jpeg-payload-bytes"
        self.media_sha256 = hashlib.sha256(self.media_bytes).hexdigest()

        self.payload_dict = {
            "challenge_nonce": "bank-vcip-nonce-9912",
            "confidence": 0.965,
            "device_security_tier": "STRONGBOX",
            "media_sha256": self.media_sha256,
            "model_hashes": {
                "siglip_probe_int8.tflite": "1b8c4f3de7822196c3c0ebf59f683d86f24d5fa2476fb1d9312a1e14b1519c7b",
                "srm_forensic_cnn_int8.tflite": "08397e56b53e506ef89b290f114f167babb2ca530a52320ec770b519bf307ca1",
            },
            "model_release": "onn-prod-v10",
            "overall_score": 0.9412,
            "schema_version": "onn-receipt-v1",
            "signals": {
                "semantic_probe": 0.981,
                "srm": 0.874,
            },
            "timestamp_epoch_ms": 1792963200000,
            "verdict": "LIKELY_AI",
        }
        self.canonical_str = serialize_canonical_payload(self.payload_dict)
        sig_bytes = self.private_key.sign(self.canonical_str.encode("utf-8"), ec.ECDSA(hashes.SHA256()))
        self.sig_b64 = base64.b64encode(sig_bytes).decode("ascii")

    def _make_receipt(self, **overrides):
        base = {
            "schema_version": "onn-receipt-v1",
            "security_tier": "STRONGBOX",
            "signature_algorithm": "SHA256withECDSA",
            "canonical_payload": dict(self.payload_dict),
            "signature_base64": self.sig_b64,
            "public_key_pem": self.public_key_pem,
            "certificate_chain": [self.cert_b64],
        }
        base.update(overrides)
        return base

    def test_valid_strongbox_receipt_passes_all_checks(self) -> None:
        receipt = self._make_receipt()
        res = verify_receipt_bundle(
            receipt,
            media_bytes=self.media_bytes,
            expected_nonce="bank-vcip-nonce-9912",
        )
        self.assertTrue(res.valid)
        self.assertTrue(res.signature_valid)
        self.assertTrue(res.hardware_backed)
        self.assertTrue(res.cert_chain_present)
        self.assertTrue(res.leaf_key_matches)
        self.assertTrue(res.media_hash_matches)
        self.assertTrue(res.nonce_matches)
        self.assertEqual(res.reasons, [])

    def test_tampered_score_invalidates_signature(self) -> None:
        receipt = self._make_receipt()
        receipt["canonical_payload"] = dict(self.payload_dict, overall_score=0.0500)
        res = verify_receipt_bundle(receipt)
        self.assertFalse(res.valid)
        self.assertFalse(res.signature_valid)

    def test_software_tier_rejected_under_enterprise_policy(self) -> None:
        receipt = self._make_receipt(security_tier="SOFTWARE_TEST", certificate_chain=[])
        res = verify_receipt_bundle(receipt, allow_software_test_tier=False)
        self.assertFalse(res.valid)
        self.assertFalse(res.hardware_backed)

    def test_mismatched_media_bytes_rejected(self) -> None:
        receipt = self._make_receipt()
        res = verify_receipt_bundle(receipt, media_bytes=b"substituted-doctored-file")
        self.assertFalse(res.valid)
        self.assertFalse(res.media_hash_matches)


if __name__ == "__main__":
    unittest.main()
