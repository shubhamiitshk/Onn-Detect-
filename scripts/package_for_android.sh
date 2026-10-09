#!/usr/bin/env bash
# Install one complete validated release transaction. Missing inputs are errors.
# Usage: ./scripts/package_for_android.sh /absolute/path/to/bundle
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/validated/bundle" >&2
  exit 2
fi
exec "${ONN_PYTHON:-python}" "$ROOT/ml/scripts/release_bundle.py" \
  --bundle "$1" --install "$ROOT/android/app/src/main/assets" \
  --kotlin-destination "$ROOT/android/app/src/main/java/com/onndetect/detection/fusion/XGBoostMetaClassifier.kt"
