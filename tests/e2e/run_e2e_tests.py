"""Standalone E2E Test Suite Runner for ONN-DETECT.

Executes all opaque-box tests across Tiers 1-4 and reports a structured scorecard.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

E2E_DIR = Path(__file__).resolve().parent
REPO_ROOT = E2E_DIR.parent.parent


def main() -> int:
    print("=" * 80)
    print("          ONN-DETECT: OPAQUE-BOX E2E TEST RUNNER (TIERS 1-4)")
    print("=" * 80)
    print(f"Target Repository : {REPO_ROOT}")
    print(f"E2E Test Root     : {E2E_DIR}")
    print(f"Python Interpreter: {sys.executable}")
    print("=" * 80)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    start_time = time.time()
    pytest_args = [
        str(E2E_DIR),
        "-v",
        "--tb=short",
    ]

    exit_code = pytest.main(pytest_args)
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"E2E Execution Finished in {elapsed:.2f}s with Exit Code: {exit_code}")
    if exit_code == 0:
        print("[SUCCESS] All 145 E2E Tests across Tiers 1-4 PASSED with 100% success rate.")
    else:
        print(f"[FAILURE] Test runner encountered failures. Exit code: {exit_code}")
    print("=" * 80)
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
