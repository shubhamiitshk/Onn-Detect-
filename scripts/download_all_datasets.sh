#!/usr/bin/env bash
# An explicit pinned corpus is required; no automatic bulk downloads.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ONN_PYTHON:-python}" "$ROOT/ml/scripts/download_all_datasets.py" "$@"
