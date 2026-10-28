#!/usr/bin/env bash
# Refreshes the production C2PA trust anchors bundled with the app.
#
# The list is the OFFICIAL C2PA trust list (the one the conformance program
# and Adobe Content Authenticity Inspect use), published by the c2pa-org
# conformance-public repository. Refresh it before every release so newly
# enrolled signers (new cameras, new generators) verify offline.
#
# F1 rules (unchangeable):
#   - anchors come ONLY from this official distribution;
#   - the app NEVER ships a private key or a synthetic root;
#   - an empty/unparsable list fails closed (nothing verifies as signed).
set -euo pipefail

DEST="android/app/src/main/assets/trust/c2pa_trust_anchors.pem"
SRC="https://raw.githubusercontent.com/c2pa-org/conformance-public/refs/heads/main/trust-list/C2PA-TRUST-LIST.pem"

curl -fsSL --max-time 60 "$SRC" -o "$DEST"

anchors=$(grep -c -- '-----BEGIN CERTIFICATE-----' "$DEST" || true)
if [ "${anchors:-0}" -lt 30 ]; then
    echo "REFUSING: suspiciously small trust list (${anchors} anchors) — keeping would fail closed anyway." >&2
    exit 1
fi

# Optional pinning: set C2PA_TRUST_ANCHORS_SHA256 to the expected digest.
# When unset, print the digest so it can be reviewed and pinned in CI.
actual_sha=$(sha256sum "$DEST" | awk '{print $1}')
if [ -n "${C2PA_TRUST_ANCHORS_SHA256:-}" ]; then
    if [ "$actual_sha" != "$C2PA_TRUST_ANCHORS_SHA256" ]; then
        echo "REFUSING: trust-anchor SHA-256 mismatch (expected $C2PA_TRUST_ANCHORS_SHA256, got $actual_sha)" >&2
        exit 1
    fi
    echo "OK: SHA-256 verified (${actual_sha})"
else
    echo "WARNING: C2PA_TRUST_ANCHORS_SHA256 not set; staged digest is ${actual_sha}. Pin it before release." >&2
fi

echo "OK: ${anchors} production anchors staged at ${DEST}"