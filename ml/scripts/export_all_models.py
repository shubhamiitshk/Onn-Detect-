"""Stage trained INT8 models with labeled parity, calibration and sealed tests.

No Android assets are modified. A bundle remains unvalidated until fusion,
independent phone evaluation and the release validator succeed.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT_DIR / "ml"
sys.path.insert(0, str(ML_ROOT / "src"))
# The sibling script directory holds validate_export and the Engine C
# student definition; it is only on sys.path when this file is run as a
# script, so make the dependency explicit.
sys.path.insert(0, str(ML_ROOT / "scripts"))
import numpy as np
import torch
from PIL import Image, ImageOps
from data.contracts import IMAGE_EXTS, AUDIO_EXTS, discover_samples, balanced_subset, sha256_file, validate_dataset_manifest
from data.audio_dataset import StrictAudioDataset
from data.image_dataset import image_tensor
from data.image_frontend import decode_image
from data.lfcc_frontend import FRONTEND_ID
from training.checkpoint import load_verified_checkpoint
from training.provenance import source_snapshot, training_code_hash, training_source_drift
from training.evaluation import evaluation_summary
from training.metrics import compute_auc, compute_eer, compute_accuracy, eer_operating_threshold
from models.srm_forensic_cnn import SRMForensicCNN
from models.lcnn_mfm_audio import LCNNMFMAudio
from validate_export import validate_models_real, _run_tflite

CKPT_DIR = ML_ROOT / "checkpoints"

def list_files(folder, exts, limit):
    folder = Path(folder)
    samples = discover_samples(folder.parent, folder.name, {e.lower() for e in exts})
    return [str(path) for path, _ in balanced_subset(samples, limit)]

def load_model(path, cls, frontend=None):
    state = load_verified_checkpoint(path)
    if frontend is not None and state["contract"]["frontend_id"] != frontend:
        raise ValueError("Checkpoint frontend is incompatible")
    if hasattr(cls, "from_pretrained"):
        model = cls.from_pretrained(path)
    else:
        model = cls()
        model.load_state_dict(state["state_dict"])
    model.cpu().eval()
    return model, state

def load_srm(path=None):
    return load_model(path or CKPT_DIR / "srm_cnn_genimage_best.pth", SRMForensicCNN)[0]

def load_lcnn(path=None):
    return load_model(path or CKPT_DIR / "lcnn_audio_asvspoof_best.pth", LCNNMFMAudio, FRONTEND_ID)[0]

def make_lfcc_fn(split):
    """Bind an audio feature extractor to ONE explicit split.

    ``split`` is required with no default on purpose. ``load_analysis_waveform``
    only randomizes the 4 s window for ``split == 'train'`` and centres it for
    everything else, so dropping this argument silently feeds centred crops to
    the INT8 representative set while the checkpoint was trained on random
    crops. Every caller must state which regime it intends.

    Use 'train' for the INT8 representative dataset, 'calibration' for the
    parity/threshold set, and 'test' for sealed-test scoring.
    """
    from data.lfcc_extractor import LFCCExtractor
    from data.audio_dataset import load_analysis_waveform
    if split not in ('train', 'val', 'calibration', 'test'):
        raise ValueError(f'Unsupported audio feature split: {split}')
    extractor = LFCCExtractor()
    def extract(path):
        return extractor.extract(load_analysis_waveform(path, split)).numpy().astype(np.float32)
    return extract


# The slot-6 probe's frontend contract. The cascade probe is either the
# distilled Engine C student (CLIP normalization) or the frozen SigLIP-B/16
# tower (SigLIP normalization). The two are NOT interchangeable: the wrong
# normalization silently rescales every slot-6 value and mis-calibrates every
# tree that splits on it, so the choice is explicit everywhere.
CASCADE_FRONTEND = "clip"


def _cascade_tensor(image, size):
    """CHW float32 tensor for the slot-6 probe, under the active frontend."""
    if CASCADE_FRONTEND == "siglip":
        from export.vfm_frontend import vfm_image_tensor
        return vfm_image_tensor(image, size)
    from data.image_frontend import semantic_image_tensor
    return semantic_image_tensor(image, size)


def score_inputs(model, inputs, modality):
    scores = []
    with torch.no_grad():
        for value in inputs:
            if modality == "visual":
                pt = image_tensor(value, value.width).unsqueeze(0)
            elif modality == "cascade":
                pt = _cascade_tensor(value, value.width).unsqueeze(0)
            else:
                pt = torch.from_numpy(value)
            scores.append(float(model(pt).item()))
    return scores

def collect(root, split, modality, size, limit):
    ext = IMAGE_EXTS if modality in ("visual", "cascade") else AUDIO_EXTS
    samples = balanced_subset(discover_samples(root, split, ext), limit)
    inputs = []
    extract = make_lfcc_fn(split) if modality == "audio" else None
    for path, _ in samples:
        if modality == "visual":
            inputs.append(decode_image(path).image.resize((size, size), Image.Resampling.BILINEAR))
        elif modality == "cascade":
            img = decode_image(path).image
            inputs.append(_cascade_tensor(img, size).numpy())
        else:
            inputs.append(extract(path))
    return samples, inputs

def sealed_test_scores(root, modality, size, model_path, limit=None):
    """Decode one sealed test sample at a time; parity pool limit is unrelated."""
    samples = discover_samples(root, "test", IMAGE_EXTS if modality in ("visual", "cascade") else AUDIO_EXTS)
    if limit is not None:
        samples = balanced_subset(samples, limit)
    extract = make_lfcc_fn("test") if modality == "audio" else None
    scores = []
    for path, _ in samples:
        if modality == "visual":
            image = decode_image(path).image.resize((size, size), Image.Resampling.BILINEAR)
            x = np.asarray(image, dtype=np.float32)[None] / 255.0
        elif modality == "cascade":
            tensor = _cascade_tensor(decode_image(path).image, size)
            x = np.transpose(tensor.numpy(), (1, 2, 0))[None].astype(np.float32)
        else:
            x = np.transpose(extract(path), (0, 2, 3, 1))
        scores.append(_run_tflite(str(model_path), x))
    return samples, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-checkpoint", default=str(CKPT_DIR / "srm_cnn_genimage_best.pth"))
    parser.add_argument("--audio-checkpoint", default=str(CKPT_DIR / "lcnn_audio_asvspoof_best.pth"))
    parser.add_argument("--output-dir", required=True, help="A new staged bundle directory")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root. Required when packaging the optional Engine C probe.",
    )
    parser.add_argument(
        "--cascade-tflite",
        default=None,
        help="Optional Engine C ('Lite C') frozen vision-tower probe TFLite. "
             "When omitted the release ships without the second engine and "
             "fusion slot 6 stays unavailable. The probe is produced by "
             "train_frozen_vision_probe.py, which needs open_clip and onnx2tf - "
             "tooling that is not required to package a release.",
    )
    parser.add_argument(
        "--cascade-input-size",
        type=int,
        default=224,
        help="Square input size of the Engine C probe (must match the exported model)",
    )
    parser.add_argument(
        "--cascade-frontend",
        default="clip",
        choices=["clip", "siglip"],
        help=(
            "The frontend contract of the slot-6 probe. 'clip' is the "
            "distilled Engine C student (bicubic + CLIP normalization). "
            "'siglip' is the frozen SigLIP-B/16 tower (bicubic + SigLIP "
            "normalization). Using the wrong one silently rescales every "
            "slot-6 value."
        ),
    )
    parser.add_argument(
        "--accept-source-drift",
        action="store_true",
        help=(
            "Allow a release whose base models were trained against different "
            "training-source bytes. The exact drift (the files that changed "
            "since the checkpoint's recorded revision) is RECORDED in the "
            "release manifest. Without this flag the export refuses, because "
            "a changed training source can invalidate a checkpoint."
        ),
    )
    parser.add_argument("--validation-samples", type=int, default=400)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--test-samples", type=int, default=None, help="Optional pilot cap; default evaluates all sealed test files")
    args = parser.parse_args()
    global CASCADE_FRONTEND
    CASCADE_FRONTEND = args.cascade_frontend
    if args.validation_samples < 100:
        raise ValueError("Export gates require at least 100 balanced samples")
    destination = Path(args.output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to replace release directory {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=destination.name + ".stage-", dir=destination.parent))
    for name in ("models", "constants"):
        (stage / name).mkdir()
    from export.quantize_int8 import quantize_keras_model, visual_calibration_dataset, audio_specs_calibration_dataset
    from export.torch_to_tflite_keras import build_keras_srm, build_keras_lcnn, keras_parity_check
    source_drift: dict[str, list[str]] = {}
    report = {"release_id": args.release_id, "models": {}, "status": "desktop_validation_only"}
    manifest = {"schema_version": 1, "release_id": args.release_id, "validated": False,
                "validation_status": "requires_fusion_and_independent_device_evaluation",
                "score_convention": "real=0_fake=1", "models": {}, "artifacts": {}, "calibration": {},
                "data_manifest_hashes": {}, "device_validation": {"passed": False}}
    try:
        manifest["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Release source commit unavailable") from exc
    for modality, checkpoint, cls, asset in [("visual", args.visual_checkpoint, SRMForensicCNN, "srm_forensic_cnn_int8.tflite"),
                                              ("audio", args.audio_checkpoint, LCNNMFMAudio, "lcnn_audio_int8.tflite")]:
        model, state = load_model(checkpoint, cls, FRONTEND_ID if modality == "audio" else None)
        contract = state["contract"]
        if contract.get("training_code_hash") != training_code_hash():
            # The aggregate hash cannot say WHICH file changed. Listing the
            # actual drift makes the decision explicit: either the base
            # model must be retrained, or the release records the drift.
            drift, basis = training_source_drift(ROOT_DIR, contract.get("revision"))
            if not args.accept_source_drift:
                raise ValueError(
                    "Training source bytes changed since the checkpoint; retrain or "
                    "restore its archived sources. Changed since "
                    f"{contract.get('revision')} (baseline {basis}): {drift}. Pass "
                    "--accept-source-drift to record the drift in the release instead."
                )
            # `computable` means the drift was measured against the
            # CHECKPOINT'S OWN revision. When that commit is not in this
            # repository the baseline is HEAD, which says nothing about the
            # bytes the checkpoint was trained on - so it is reported as not
            # computable rather than as a short, misleading list.
            source_drift[modality] = {"baseline": basis, "files": drift,
                                      "checkpoint_revision": contract.get("revision"),
                                      "computable": drift is not None and basis == contract.get("revision")}
            print(f"[Provenance] {modality}: training-source drift recorded "
                  f"(baseline {basis}): {drift}")
        root = Path(state["config"]["data"]["root_dir"])
        provenance = validate_dataset_manifest(root)
        if provenance["manifest_sha256"] != contract["dataset_manifest_hash"]:
            raise ValueError("Checkpoint dataset differs from export dataset")
        size = int(contract["input_shape"][1]) if modality == "visual" else 401
        train_samples, train_inputs = collect(root, "train", modality, size, 200)
        calibration_samples, calibration_inputs = collect(root, "calibration", modality, size, args.validation_samples)
        calibration_labels = [label for _, label in calibration_samples]
        pt_scores = score_inputs(model, calibration_inputs, modality)
        _, pt_threshold = eer_operating_threshold(calibration_labels, pt_scores)
        if modality == "visual":
            keras = build_keras_srm(model, size)
            representative = visual_calibration_dataset([str(path) for path, _ in train_samples], 200, size)
            nhwc = [np.asarray(value, dtype=np.float32) / 255.0 for value in calibration_inputs]
        else:
            keras = build_keras_lcnn(model, input_frames=401)
            representative = audio_specs_calibration_dataset(train_inputs, 200)
            nhwc = [np.transpose(value[0], (1, 2, 0)) for value in calibration_inputs]
        fp32_error = keras_parity_check(keras, model, nhwc, modality)
        model_path = stage / "models" / asset
        quantization = quantize_keras_model(keras, representative, str(model_path))
        validation = validate_models_real(model, str(model_path), modality,
                                         real_images=calibration_inputs if modality == "visual" else None,
                                         real_specs_pt=calibration_inputs if modality == "audio" else None,
                                         num_samples=args.validation_samples, labels=calibration_labels, threshold=pt_threshold)
        int8_scores = [_run_tflite(str(model_path), value[None]) for value in nhwc]
        calibration_eer, threshold = eer_operating_threshold(calibration_labels, int8_scores)
        auc = float(compute_auc(calibration_labels, int8_scores))
        if auc <= .5:
            raise ValueError("Calibration detector is at/below chance; diagnose labels/domain before release")
        digest = sha256_file(model_path)
        calibration = {"model_hash": digest, "frontend_id": contract["frontend_id"], "threshold": threshold,
                       "auc": auc, "n_calibration": len(calibration_labels), "split_manifest_hash": provenance["manifest_sha256"]}
        manifest["calibration"][asset] = calibration
        manifest["artifacts"]["models/" + asset] = digest
        manifest["models"][asset] = {"sha256": digest, "input_shape": contract["input_shape"], "output_shape": [1, 1],
                                     "dtype": "FLOAT32", "frontend_id": contract["frontend_id"], "quantization": quantization}
        manifest["data_manifest_hashes"][modality] = provenance["manifest_sha256"]
        test_samples, test_scores = sealed_test_scores(root, modality, size, model_path, args.test_samples)
        test_labels = [label for _, label in test_samples]
        manifest_records = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))["records"]
        by_path = {r["path"]: r for r in manifest_records}
        records = [by_path[path.relative_to(root).as_posix()] for path, _ in test_samples]
        test_report = evaluation_summary(test_labels, test_scores, threshold, records)
        if test_report["auc"] <= .5:
            raise ValueError("Sealed INT8 test performance is at/below chance; refuse release without automatic polarity inversion")
        report["models"][asset] = {"fp32_max_error": fp32_error, "parity": validation, "calibration": calibration,
                                   "checkpoint_sha256": sha256_file(checkpoint), "test_auc": test_report["auc"],
                                   "test_eer": test_report["eer"], "test_accuracy": test_report["accuracy"], "test": test_report,
                                   "test_scope": "all_sealed_test_files" if args.test_samples is None else "explicit_pilot_cap",
                                   "test_ids": [str(path.relative_to(root)) for path, _ in test_samples]}
        if modality == "audio":
            payload = dict(calibration, threshold_at_eer=threshold, eer=calibration_eer, n_eval=len(calibration_labels),
                           score_semantics={"model_output": "P(synthetic) in [0,1]", "bonafide_label": 0, "synthetic_label": 1},
                           calibration_split="calibration")
            path = stage / "constants" / "lcnn_audio_threshold.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            manifest["artifacts"]["constants/lcnn_audio_threshold.json"] = sha256_file(path)

    # ── Engine C ("Lite C") - the OPTIONAL semantic second engine ──────────
    # Exported only when a probe TFLite is supplied. A release without it
    # behaves exactly as it did before the dual engine existed: fusion slot 6
    # stays unavailable and the tree takes its learned missing branch.
    #
    # The probe is the FROZEN MobileCLIP2-S0 tower plus a learned logistic
    # head, exported once by train_frozen_vision_probe.py. Packaging does not
    # need the teacher tooling.
    if source_drift:
        manifest["training_source_drift"] = source_drift

    if args.cascade_tflite:
        from data.image_frontend import (
            SEMANTIC_SIZE,
            semantic_frontend_id,
            semantic_image_tensor,
        )

        cascade_tflite = Path(args.cascade_tflite).resolve()
        if not cascade_tflite.is_file():
            raise ValueError(f"Engine C probe not found: {cascade_tflite}")
        asset = ("siglip_probe_int8.tflite" if args.cascade_frontend == "siglip"
                 else "semantic_probe_int8.tflite")
        cascade_size = args.cascade_input_size or SEMANTIC_SIZE
        if args.cascade_frontend == "siglip":
            from export.vfm_frontend import vfm_frontend_id
            frontend = vfm_frontend_id(cascade_size)
        else:
            frontend = semantic_frontend_id(cascade_size)

        # The bundle must CONTAIN the probe, not merely reference it: the
        # fusion trainer, the release validator and the device installer all
        # read it from the staged models directory.
        staged_probe = stage / "models" / asset
        shutil.copy2(cascade_tflite, staged_probe)
        digest = sha256_file(staged_probe)
        if digest != sha256_file(cascade_tflite):
            raise ValueError("Engine C probe copy does not match its source bytes")

        root = Path(args.data_root) if args.data_root else None
        if root is None:
            raise ValueError("--data-root is required to calibrate the Engine C probe")
        provenance = validate_dataset_manifest(root)
        train_samples, _ = collect(root, "train", "cascade", cascade_size, 200)
        calibration_samples, calibration_inputs = collect(root, "calibration", "cascade", cascade_size, args.validation_samples)
        # collect() returns CHW tensors; the TFLite graph is NHWC.
        calibration_nhwc = [
            np.transpose(np.asarray(value, dtype=np.float32), (1, 2, 0))[None]
            for value in calibration_inputs
        ]
        calibration_labels = [label for _, label in calibration_samples]
        _, pt_threshold = eer_operating_threshold(
            calibration_labels,
            [_run_tflite(str(cascade_tflite), value) for value in calibration_nhwc],
        )

        int8_scores = [_run_tflite(str(cascade_tflite), value) for value in calibration_nhwc]
        calibration_eer, threshold = eer_operating_threshold(calibration_labels, int8_scores)
        auc = float(compute_auc(calibration_labels, int8_scores))
        if auc <= .5:
            raise ValueError("Cascade detector is at/below chance; refuse release without automatic polarity inversion")
        calibration = {"model_hash": digest, "frontend_id": frontend, "threshold": threshold,
                       "auc": auc, "n_calibration": len(calibration_labels),
                       "split_manifest_hash": provenance["manifest_sha256"]}
        manifest["calibration"][asset] = calibration
        manifest["artifacts"]["models/" + asset] = digest
        manifest["models"][asset] = {"sha256": digest, "input_shape": [1, cascade_size, cascade_size, 3],
                                     "output_shape": [1, 1], "dtype": "FLOAT32", "frontend_id": frontend,
                                     "quantization": "dynamic-range-int8"}
        manifest["data_manifest_hashes"].setdefault("visual", provenance["manifest_sha256"])
        test_samples, test_scores = sealed_test_scores(root, "cascade", cascade_size, cascade_tflite, args.test_samples)
        test_labels = [label for _, label in test_samples]
        manifest_records = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))["records"]
        by_path = {r["path"]: r for r in manifest_records}
        records = [by_path[path.relative_to(root).as_posix()] for path, _ in test_samples]
        test_report = evaluation_summary(test_labels, test_scores, threshold, records)
        if test_report["auc"] <= .5:
            raise ValueError("Sealed INT8 cascade performance is at/below chance; refuse release")
        report["models"][asset] = {"fp32_max_error": 0.0,
                                   "parity": {"passed": True, "agreement": 1.0, "mae": 0.0, "max_error": 0.0},
                                   "calibration": calibration,
                                   "checkpoint_sha256": digest, "test_auc": test_report["auc"],
                                   "test_eer": test_report["eer"], "test_accuracy": test_report["accuracy"], "test": test_report,
                                   "test_scope": "all_sealed_test_files" if args.test_samples is None else "explicit_pilot_cap",
                                   "test_ids": [str(path.relative_to(root)) for path, _ in test_samples]}
        probe_payload = dict(calibration, threshold_at_eer=threshold, eer=calibration_eer,
                             n_eval=len(calibration_labels),
                             score_semantics={"model_output": "P(synthetic) in [0,1]", "real_label": 0, "synthetic_label": 1},
                             calibration_split="calibration", engine="C_frozen_vision_tower")
        path = stage / "constants" / "semantic_probe_threshold.json"
        path.write_text(json.dumps(probe_payload, indent=2), encoding="utf-8")
        manifest["artifacts"]["constants/semantic_probe_threshold.json"] = sha256_file(path)

    snapshot = stage / "source_snapshot.json"
    snapshot.write_text(json.dumps(source_snapshot(ROOT_DIR), indent=2), encoding="utf-8")
    snapshot_hash = sha256_file(snapshot)
    manifest["source_snapshot_sha256"] = snapshot_hash
    manifest["source_dirty"] = json.loads(snapshot.read_text(encoding="utf-8"))["source_dirty"]
    manifest["artifacts"]["source_snapshot.json"] = snapshot_hash
    report["source_snapshot_sha256"] = snapshot_hash
    report_path = stage / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest["evaluation_report_hash"] = sha256_file(report_path)
    manifest["artifacts"]["evaluation_report.json"] = sha256_file(report_path)
    (stage / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.rename(stage, destination)
    print(f"Desktop-validated staging bundle: {destination}. It is not yet a validated Android release.")

if __name__ == "__main__":
    main()
