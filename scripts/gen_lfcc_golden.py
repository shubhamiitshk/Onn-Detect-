"""Generate golden LFCC vectors for the Kotlin LFCCExtractor unit test.

By default this uses the NumPy reference (``ml/src/data/lfcc_reference.py``),
so goldens can be regenerated with no PyTorch install. Pass ``--torch`` to
cross-check the reference against the pinned PyTorch front end
(``ml/src/data/lfcc_frontend.py``) when a torch environment is available;
that check FAILS LOUDLY on divergence and should be run before any front-end
change is trusted. (Deliberately NOT ``torchaudio.transforms.LFCC``: as of
torchaudio 2.11 that hardcodes ``top_db=80.0`` - see the ``lfcc_frontend``
module docstring.)

    python scripts/gen_lfcc_golden.py
    python scripts/gen_lfcc_golden.py --torch
    python scripts/gen_lfcc_golden.py --json scripts/lfcc_golden.json
    python scripts/gen_lfcc_golden.py --check scripts/lfcc_golden.json

The emitted numbers assert CMVN (per-coefficient over TIME), matching
``ml/src/data/lfcc_extractor.py``. Frame indices in the output are LITERAL -
a previous version labeled the last frame "frame99" and the JVM test caught
the off-by-one.

``--json`` writes the golden set to a machine-readable file and ``--check``
compares a freshly computed set against a committed one, exiting non-zero on
any drift. The Kotlin literals in ``LFCCExtractorTest.kt`` were previously
hand-transcribed from printed output with nothing comparing the two, so a stale
transcription that happened to sit inside the 2e-3 tolerance would pass. The
committed JSON closes that loop; CI runs ``--check``.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "ml", "src"))

from data.lfcc_reference import (  # noqa: E402
    N_FFT,
    N_FILTER,
    N_LFCC,
    HOP_LENGTH,
    SAMPLE_RATE,
    lfcc_raw,
    normalize_cmn,
    two_tone_signal,
)


def _golden_dict(out: np.ndarray) -> dict:
    """Extract the golden slices from a (T, N_LFCC) normalized feature matrix.

    Frame indices are LITERAL. A previous version read out[n_frames - 1] (the
    LAST frame, index 100) while labeling it "frame99" - the JVM test caught it
    because real frame 99 differs. Always index explicitly; never
    shape[0]-1 under a different name.
    """
    n_frames = out.shape[0]
    assert n_frames > 99, f"probe signal too short: {n_frames} frames"
    return {
        "shape": list(out.shape),
        "frame50": [round(float(v), 6) for v in out[50, :8]],
        "frame0": [round(float(v), 6) for v in out[0, :4]],
        "frame99": [round(float(v), 6) for v in out[99, :4]],
        "frame_last": [round(float(v), 6) for v in out[n_frames - 1, :4]],
        # CMVN -> per-coefficient zero mean over time, so this is (T-1)/T.
        "energy": round(float((out ** 2).mean()), 6),
    }


def reference_goldens(sig: np.ndarray) -> dict:
    return _golden_dict(normalize_cmn(lfcc_raw(sig)).T)   # (T, N_LFCC)


def pinned_frontend_goldens(sig: np.ndarray) -> dict:
    """Cross-check against the PINNED PyTorch front end.

    (Previously named torchaudio_goldens — renamed because it deliberately does
    NOT use torchaudio.transforms.LFCC: as of torchaudio 2.11 that hardcodes
    ``top_db=80.0``, which floors ~91 % of spectrogram cells and produces
    features that disagree with the Kotlin runtime by up to 10.4. See the
    ``lfcc_frontend`` module docstring. If you want to reproduce the old
    (wrong) numbers, use ``scripts/verify_front_end_parity.py`` section 3.)
    """
    import torch

    from data.lfcc_frontend import LFCCFrontend

    fe = LFCCFrontend()
    with torch.no_grad():
        norm = fe(torch.tensor(sig, dtype=torch.float64).unsqueeze(0))[0].numpy()
    return _golden_dict(norm)


GOLDEN_TOLERANCE = 1e-6   # goldens are stored at 6 decimals


def _golden_document(ref: dict) -> dict:
    """Wrap the golden slices with the parameters that produced them."""
    from data.lfcc_frontend import FRONTEND_ID
    return {
        "schema_version": 1,
        "probe": "two_tone_signal",
        "frontend_id": FRONTEND_ID,
        "parameters": {
            "sample_rate": SAMPLE_RATE,
            "n_fft": N_FFT,
            "n_filter": N_FILTER,
            "n_lfcc": N_LFCC,
            "hop_length": HOP_LENGTH,
            "normalization": "cmvn-per-coefficient-over-time",
        },
        "goldens": ref,
    }


def _check_goldens(path: str, computed: dict) -> None:
    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)
    problems = []
    if stored.get("schema_version") != computed.get("schema_version"):
        problems.append(f"schema_version {stored.get('schema_version')} != {computed.get('schema_version')}")
    if stored.get("frontend_id") != computed.get("frontend_id"):
        problems.append(
            f"frontend_id changed: stored {stored.get('frontend_id')!r} != {computed.get('frontend_id')!r}. "
            "Regenerate the goldens and the Kotlin literals together, and retrain any checkpoint.")
    if stored.get("parameters") != computed.get("parameters"):
        problems.append(f"parameters changed: {stored.get('parameters')} != {computed.get('parameters')}")
    expected = computed.get("goldens", {})
    actual = stored.get("goldens")
    if actual is None:
        problems.append("stored document has no 'goldens' key")
    else:
        for key in ("shape", "energy"):
            if actual.get(key) != expected.get(key):
                problems.append(f"{key}: stored {actual.get(key)} != computed {expected.get(key)}")
        for key in ("frame50", "frame0", "frame99", "frame_last"):
            a, e = actual.get(key), expected.get(key)
            if a is None or len(a) != len(e):
                problems.append(f"{key}: length mismatch {a} vs {e}")
                continue
            worst = max(abs(float(x) - float(y)) for x, y in zip(a, e))
            if worst > GOLDEN_TOLERANCE:
                problems.append(f"{key}: max drift {worst:.3e} > {GOLDEN_TOLERANCE:.0e} ({a} vs {e})")
    if problems:
        for problem in problems:
            print(f"  [FAIL] {problem}")
        raise SystemExit(
            "Golden drift detected. The committed goldens no longer match the NumPy "
            "reference. If the front end changed deliberately, regenerate with --json, "
            "update LFCCExtractorTest.kt literals in the same change, and retrain."
        )
    print(f"  [PASS] committed goldens match the NumPy reference ({path})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--torch",
        action="store_true",
        help="cross-check the NumPy reference against the pinned PyTorch front end",
    )
    ap.add_argument("--json", metavar="PATH", help="write the golden set to PATH")
    ap.add_argument(
        "--check",
        metavar="PATH",
        help="compare freshly computed goldens against the committed PATH; non-zero on drift",
    )
    args = ap.parse_args()

    sig = two_tone_signal()
    ref = reference_goldens(sig)
    document = _golden_document(ref)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"Wrote goldens to {args.json}")

    if args.check:
        print(f"Checking goldens against {args.check}")
        _check_goldens(args.check, document)

    print(f"signal          : {sig.shape[0]} samples @ {SAMPLE_RATE} Hz")
    print(f"features        : {ref['shape']}  (T, {N_LFCC})")
    print(f"normalization   : CMVN per-coefficient over TIME (matches training)")
    print()
    print("FRAME50 (first 8) =", ref["frame50"])
    print("FRAME0  (first 4) =", ref["frame0"])
    print("FRAME99 (first 4) =", ref["frame99"])
    print("FRAME_LAST (first 4) =", ref["frame_last"])
    print("ENERGY            =", ref["energy"], " (expect (T-1)/T)")
    print()

    if args.torch:
        try:
            auth = pinned_frontend_goldens(sig)
        except ImportError:
            raise SystemExit("Requested --torch verification requires PyTorch; cross-check was not run")
        print("pinned PyTorch front-end cross-check")
        print("  shape  :", auth["shape"], "vs", ref["shape"])
        worst = 0.0
        for key in ("frame50", "frame0", "frame99", "frame_last"):
            d = np.abs(np.asarray(auth[key]) - np.asarray(ref[key])).max()
            worst = max(worst, float(d))
            print(f"  {key:<11} max |numpy - torch| = {d:.2e}")
        e = abs(float(auth["energy"]) - float(ref["energy"]))
        print(f"  energy   abs diff = {e:.2e}")
        print()
        if worst > 1e-2:
            raise SystemExit(
                f"REFERENCE DISAGREES WITH THE PINNED FRONT END (max {worst:.2e}). "
                "Do not regenerate goldens from a broken reference."
            )
        print("Reference agrees with the pinned front end. Goldens are trustworthy.")
    else:
        print("Run with --torch to confirm this reference matches the training front end.")


if __name__ == "__main__":
    main()
