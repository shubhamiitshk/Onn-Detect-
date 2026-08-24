#!/usr/bin/env bash
# Package exported models + calibration constants into the Android assets
# directory and print SHA-256 digests for verification.
#
# Usage (from repo root): ./scripts/package_for_android.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPORTED="$ROOT/ml/exported"
CONST_EXPORTED="$ROOT/ml/exported"
ASSETS_MODELS="$ROOT/android/app/src/main/assets/models"
ASSETS_CONST="$ROOT/android/app/src/main/assets/constants"

mkdir -p "$ASSETS_MODELS" "$ASSETS_CONST"

for m in srm_forensic_cnn_int8.tflite lcnn_audio_int8.tflite; do
  if [ -f "$EXPORTED/$m" ]; then
    cp "$EXPORTED/$m" "$ASSETS_MODELS/$m"
    echo "shipped models/$m ($(du -k "$ASSETS_MODELS/$m" | cut -f1) KB)"
  else
    echo "warning: $EXPORTED/$m not found (run 'make export' first)" >&2
  fi
done

for c in natural_diff_histogram.json lcnn_audio_threshold.json meta_classifier.json; do
  [ -f "$CONST_EXPORTED/$c" ] && cp "$CONST_EXPORTED/$c" "$ASSETS_CONST/$c" && echo "shipped constants/$c"
done

if [ -d "$ASSETS_MODELS" ]; then
  echo "--- model hashes ---"
  ( cd "$ASSETS_MODELS" && sha256sum *.tflite 2>/dev/null || true )
fi

echo "Regenerating constants/model_hashes.json is done by:"
echo "  python ml/scripts/generate_android_constants.py"
