"""Authoritative config-based training with independent test and calibration splits."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training.entrypoint import run
if __name__ == "__main__":
    run("visual")
