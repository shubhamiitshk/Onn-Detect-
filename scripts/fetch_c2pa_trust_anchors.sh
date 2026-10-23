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
if [ "${anchors:-0}" -lt 20 ]; then
  echo "REFUSING: suspiciously small trust list (${anchors} anchors) — keeping would fail closed anyway." >&2
  exit 1
fi

echo "OK: ${anchors} production anchors staged at ${DEST}"
