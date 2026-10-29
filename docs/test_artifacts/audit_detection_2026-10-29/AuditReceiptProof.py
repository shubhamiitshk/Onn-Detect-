"""Audit-only: reuse existing test fixture to check claimed hardware provenance."""
from pathlib import Path
import base64
import sys

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from cryptography import x509
from scripts.tests.test_verify_forensic_receipt_adversarial import (
    VerifyForensicReceiptAdversarialTests,
)
from scripts.verify_forensic_receipt import verify_receipt_bundle

fixture = VerifyForensicReceiptAdversarialTests()
fixture.setUp()
receipt = fixture._make_receipt()
result = verify_receipt_bundle(
    receipt,
    media_bytes=fixture.media_bytes,
    expected_nonce="bank-vcip-nonce-9912",
)
leaf = x509.load_der_x509_certificate(base64.b64decode(fixture.cert_b64))
try:
    leaf.extensions.get_extension_for_oid(x509.ObjectIdentifier("1.3.6.1.4.1.11129.2.1.17"))
    attestation_present = True
except x509.ExtensionNotFound:
    attestation_present = False

text = (
    "PROOF: locally generated software EC key from existing test fixture\n"
    f"self_signed_certificate={leaf.issuer == leaf.subject}\n"
    f"android_attestation_extension_present={attestation_present}\n"
    f"claimed_tier={receipt['security_tier']}\n"
    f"verifier_valid={result.valid}\n"
    f"verifier_hardware_backed={result.hardware_backed}\n"
    f"reasons={result.reasons}\n"
)
print(text, end="")
Path(__file__).with_name("receipt-proof.txt").write_text(text, encoding="utf-8")
assert result.valid and result.hardware_backed and not attestation_present
