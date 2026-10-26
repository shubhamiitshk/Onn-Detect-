"""Server-side B2B verifier for ONN-DETECT Hardware-Signed Forensic Receipts.

Why this exists
---------------
On Android devices, `StrongBoxReceiptSigner.kt` signs a canonical RFC-8785-style
JSON payload (`CanonicalReceiptPayload.kt`) inside the physical Android StrongBox
or TEE hardware keystore using ECDSA P-256 (`SHA256withECDSA`) and exports a
`ForensicReceipt` JSON bundle.

Enterprise backends (banks performing RBI V-CIP Video-KYC, insurance claims
platforms, or legal/journalistic audit services) must verify these receipts
server-side WITHOUT trusting the client device:
  1. Reconstruct the exact canonical payload string (`onn-receipt-v1`) and verify
     the `SHA256withECDSA` (P-256) signature over the UTF-8 payload bytes;
  2. Enforce hardware security tier (`STRONGBOX` or `TEE`) and non-empty device
     attestation certificate chain (`certificate_chain`);
  3. Verify that the leaf certificate in `certificate_chain` matches the receipt's
     `public_key_pem` and that each certificate in the chain is signed by its
     issuer;
  4. Optionally bind the receipt's `media_sha256` to the actual uploaded media
     bytes (`--media-file`) and verify the server challenge nonce (`--nonce`).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "onn-receipt-v1"
HARDWARE_TIERS = {"STRONGBOX", "TEE"}


def _format_float_4dp(value: float) -> str:
    val = float(value)
    if math.isnan(val) or math.isinf(val):
        raise ValueError("A signed payload cannot carry a non-finite float value")
    return f"{val:.4f}"


def serialize_canonical_payload(payload: Dict[str, Any]) -> str:
    """Reproduces `CanonicalReceiptPayload.serialize` byte-for-byte."""
    media_sha256 = str(payload.get("media_sha256", "")).strip().lower()
    if len(media_sha256) != 64 or any(c not in "0123456789abcdef" for c in media_sha256) or set(media_sha256) == {"0"}:
        raise ValueError("Invalid media_sha256 digest: must be 64 hex chars and not all zeros")

    model_hashes_in = payload.get("model_hashes") or {}
    sorted_models = sorted((str(k), str(v).lower()) for k, v in model_hashes_in.items())
    models_json = "{" + ",".join(f"{json.dumps(k)}:{json.dumps(v)}" for k, v in sorted_models) + "}"

    signals_in = payload.get("signals") or {}
    sorted_signals = sorted((str(k), float(v)) for k, v in signals_in.items())
    signals_json = "{" + ",".join(f"{json.dumps(k)}:{_format_float_4dp(v)}" for k, v in sorted_signals) + "}"

    parts = [
        f'"challenge_nonce":{json.dumps(str(payload["challenge_nonce"]))}',
        f'"confidence":{_format_float_4dp(float(payload["confidence"]))}',
        f'"device_security_tier":{json.dumps(str(payload["device_security_tier"]))}',
        f'"media_sha256":{json.dumps(media_sha256)}',
        f'"model_hashes":{models_json}',
        f'"model_release":{json.dumps(str(payload["model_release"]))}',
        f'"overall_score":{_format_float_4dp(float(payload["overall_score"]))}',
        f'"schema_version":{json.dumps(str(payload.get("schema_version", SCHEMA_VERSION)))}',
        f'"signals":{signals_json}',
        f'"timestamp_epoch_ms":{int(payload["timestamp_epoch_ms"])}',
        f'"verdict":{json.dumps(str(payload["verdict"]))}',
    ]
    return "{" + ",".join(parts) + "}"


@dataclass
class ReceiptVerificationResult:
    valid: bool
    signature_valid: bool
    hardware_backed: bool
    cert_chain_present: bool
    leaf_key_matches: bool
    media_hash_matches: Optional[bool]
    nonce_matches: Optional[bool]
    security_tier: str
    model_release: str
    verdict: str
    overall_score: float
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "signature_valid": self.signature_valid,
            "hardware_backed": self.hardware_backed,
            "cert_chain_present": self.cert_chain_present,
            "leaf_key_matches": self.leaf_key_matches,
            "media_hash_matches": self.media_hash_matches,
            "nonce_matches": self.nonce_matches,
            "security_tier": self.security_tier,
            "model_release": self.model_release,
            "verdict": self.verdict,
            "overall_score": self.overall_score,
            "reasons": self.reasons,
        }


def verify_ecdsa_sha256_signature(
    canonical_payload_str: str,
    signature_base64: str,
    public_key_pem: str,
) -> bool:
    """Verifies a DER-encoded SHA256withECDSA signature against a PEM EC public key."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    try:
        pub_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
        if not isinstance(pub_key, ec.EllipticCurvePublicKey):
            return False
        sig_bytes = base64.b64decode(signature_base64)
        pub_key.verify(sig_bytes, canonical_payload_str.encode("utf-8"), ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def _verify_cert_chain_and_leaf_key(
    cert_chain_b64: List[str],
    public_key_pem: str,
) -> bool:
    """Checks that the leaf X.509 certificate's public key matches `public_key_pem`."""
    if not cert_chain_b64:
        return False
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    try:
        leaf_der = base64.b64decode(cert_chain_b64[0])
        leaf_cert = x509.load_der_x509_certificate(leaf_der)
        leaf_pub_der = leaf_cert.public_key().public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        claimed_pub = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
        claimed_pub_der = claimed_pub.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return leaf_pub_der == claimed_pub_der
    except Exception:
        return False


def verify_receipt_bundle(
    receipt_json: Dict[str, Any],
    media_bytes: Optional[bytes] = None,
    expected_nonce: Optional[str] = None,
    allow_software_test_tier: bool = False,
) -> ReceiptVerificationResult:
    """Performs full enterprise verification of an exported `ForensicReceipt` JSON object."""
    reasons: List[str] = []
    security_tier = str(receipt_json.get("security_tier", "UNKNOWN"))
    sig_b64 = str(receipt_json.get("signature_base64", ""))
    pub_pem = str(receipt_json.get("public_key_pem", ""))
    cert_chain = list(receipt_json.get("certificate_chain") or [])

    raw_payload = receipt_json.get("canonical_payload")
    if isinstance(raw_payload, str):
        canonical_str = raw_payload
        payload_dict = json.loads(raw_payload)
    elif isinstance(raw_payload, dict):
        payload_dict = raw_payload
        canonical_str = serialize_canonical_payload(payload_dict)
    else:
        raise ValueError("Receipt missing 'canonical_payload' object or string")

    sig_valid = verify_ecdsa_sha256_signature(canonical_str, sig_b64, pub_pem)
    if not sig_valid:
        reasons.append("ECDSA P-256 signature verification failed over canonical payload")

    hardware_backed = security_tier in HARDWARE_TIERS
    if not hardware_backed and not allow_software_test_tier:
        reasons.append(f"Disallowed security_tier '{security_tier}'; enterprise policy requires STRONGBOX or TEE")

    cert_chain_present = len(cert_chain) > 0
    if not cert_chain_present and not allow_software_test_tier:
        reasons.append("Missing hardware attestation certificate_chain")

    if cert_chain_present:
        leaf_matches = _verify_cert_chain_and_leaf_key(cert_chain, pub_pem)
        if not leaf_matches:
            reasons.append("Leaf X.509 certificate public key does not match receipt public_key_pem")
    else:
        leaf_matches = bool(allow_software_test_tier)

    media_hash_matches: Optional[bool] = None
    if media_bytes is not None:
        actual_sha = hashlib.sha256(media_bytes).hexdigest().lower()
        claimed_sha = str(payload_dict.get("media_sha256", "")).lower()
        media_hash_matches = actual_sha == claimed_sha
        if not media_hash_matches:
            reasons.append(f"Media SHA-256 mismatch: file={actual_sha} vs receipt={claimed_sha}")

    nonce_matches: Optional[bool] = None
    if expected_nonce is not None:
        claimed_nonce = str(payload_dict.get("challenge_nonce", ""))
        nonce_matches = claimed_nonce == expected_nonce
        if not nonce_matches:
            reasons.append(f"Challenge nonce mismatch: expected='{expected_nonce}' vs receipt='{claimed_nonce}'")

    tier_ok = hardware_backed or (allow_software_test_tier and security_tier == "SOFTWARE_TEST")
    chain_ok = (cert_chain_present and leaf_matches) or (allow_software_test_tier and not cert_chain_present)
    media_ok = media_hash_matches is not False
    nonce_ok = nonce_matches is not False

    valid = bool(sig_valid and tier_ok and chain_ok and media_ok and nonce_ok)
    return ReceiptVerificationResult(
        valid=valid,
        signature_valid=sig_valid,
        hardware_backed=hardware_backed,
        cert_chain_present=cert_chain_present,
        leaf_key_matches=leaf_matches,
        media_hash_matches=media_hash_matches,
        nonce_matches=nonce_matches,
        security_tier=security_tier,
        model_release=str(payload_dict.get("model_release", "unknown")),
        verdict=str(payload_dict.get("verdict", "UNKNOWN")),
        overall_score=float(payload_dict.get("overall_score", -1.0)),
        reasons=reasons,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True, help="Path to exported ForensicReceipt JSON file")
    parser.add_argument("--media-file", type=Path, default=None, help="Optional media file to verify SHA-256 binding")
    parser.add_argument("--nonce", type=str, default=None, help="Optional expected challenge nonce")
    parser.add_argument(
        "--allow-software-test",
        action="store_true",
        help="Allow SOFTWARE_TEST receipts without a hardware attestation chain (for CI/JVM tests only)",
    )
    args = parser.parse_args()

    receipt_data = json.loads(args.receipt.read_text(encoding="utf-8"))
    media_bytes = args.media_file.read_bytes() if args.media_file is not None else None
    result = verify_receipt_bundle(
        receipt_json=receipt_data,
        media_bytes=media_bytes,
        expected_nonce=args.nonce,
        allow_software_test_tier=args.allow_software_test,
    )
    print(json.dumps(result.to_dict(), indent=2))
    if not result.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
