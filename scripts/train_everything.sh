#!/usr/bin/env bash
# Use the pinned environment; select data/config explicitly and install separately.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/ml"
exec "${ONN_PYTHON:-python}" scripts/run_training_pipeline.py "$@"
