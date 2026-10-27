from __future__ import annotations

import base64
import concurrent.futures
import datetime
import hashlib
import json
import unittest

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from scripts.verify_forensic_receipt import (
    serialize_canonical_payload,
    verify_receipt_bundle,
)


class VerifyForensicReceiptAdversarialTests(unittest.TestCase):
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

    def test_concurrent_receipt_verification(self) -> None:
        """Adversarially challenge verifier under 32 concurrent worker threads."""
        receipt = self._make_receipt()
        thread_count = 32

        def verify_worker(idx: int) -> bool:
            # Alternate valid receipts and tampered receipts
            if idx % 2 == 0:
                res = verify_receipt_bundle(
                    receipt,
                    media_bytes=self.media_bytes,
                    expected_nonce="bank-vcip-nonce-9912",
                )
                return res.valid
            else:
                tampered = self._make_receipt(signature_base64="AAAA")
                res = verify_receipt_bundle(
                    tampered,
                    media_bytes=self.media_bytes,
                    expected_nonce="bank-vcip-nonce-9912",
                )
                return not res.valid

        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(verify_worker, i) for i in range(thread_count)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        self.assertEqual(len(results), thread_count)
        self.assertTrue(all(results), "All concurrent verifications must yield deterministic pass/fail")

    def test_signature_byte_level_bit_flips(self) -> None:
        """Every single-byte bit flip in the DER signature must be rejected."""
        raw_sig = base64.b64decode(self.sig_b64)
        positions = [0, len(raw_sig) // 4, len(raw_sig) // 2, len(raw_sig) - 1]

        for pos in positions:
            mutated = bytearray(raw_sig)
            mutated[pos] ^= 0x01
            mutated_b64 = base64.b64encode(mutated).decode("ascii")

            tampered = self._make_receipt(signature_base64=mutated_b64)
            res = verify_receipt_bundle(tampered)
            self.assertFalse(res.valid, f"Mutated signature at pos {pos} must fail verification")
            self.assertFalse(res.signature_valid)

    def test_foreign_key_in_certificate_chain_rejected(self) -> None:
        """Receipt is rejected if leaf certificate carries a foreign public key."""
        foreign_priv = ec.generate_private_key(ec.SECP256R1())
        foreign_pub = foreign_priv.public_key()
        subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Foreign Key")])
        foreign_cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(foreign_pub)
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
            .sign(foreign_priv, hashes.SHA256())
        )
        foreign_cert_b64 = base64.b64encode(foreign_cert.public_bytes(serialization.Encoding.DER)).decode("ascii")

        receipt = self._make_receipt(certificate_chain=[foreign_cert_b64])
        res = verify_receipt_bundle(receipt)
        self.assertFalse(res.valid)
        self.assertFalse(res.leaf_key_matches)
        self.assertIn("Leaf X.509 certificate public key does not match receipt public_key_pem", res.reasons)

    def test_canonical_serializer_rejects_malformed_inputs(self) -> None:
        """Canonical serializer must raise ValueError for NaN, Inf, and bad media digests."""
        bad_nan = dict(self.payload_dict, confidence=float("nan"))
        with self.assertRaises(ValueError):
            serialize_canonical_payload(bad_nan)

        bad_inf = dict(self.payload_dict, overall_score=float("inf"))
        with self.assertRaises(ValueError):
            serialize_canonical_payload(bad_inf)

        bad_zeros = dict(self.payload_dict, media_sha256="0" * 64)
        with self.assertRaises(ValueError):
            serialize_canonical_payload(bad_zeros)

        bad_non_hex = dict(self.payload_dict, media_sha256="z" * 64)
        with self.assertRaises(ValueError):
            serialize_canonical_payload(bad_non_hex)

        bad_short = dict(self.payload_dict, media_sha256="deadbeef")
        with self.assertRaises(ValueError):
            serialize_canonical_payload(bad_short)


if __name__ == "__main__":
    unittest.main()
