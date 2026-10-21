"""Attach measured fusion constants to a staged release; never write Android assets."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.contracts import sha256_file
from training.acceptance import (
    ACADEMIC_TARGETS,
    MINIMUM_ACCEPTANCE,
    MINIMUM_FUSION_ACCEPTANCE,
)

CASCADE_MODEL = "siglip_probe_int8.tflite"


def read_object(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _meets(test, target):
    """True when the measured report clears every component of a target."""
    if not target or not isinstance(test, dict):
        return False
    min_tpr_1, min_tpr_01, min_auc = target
    auc = test.get("auc")
    tpr_1 = test.get("tpr_at_fpr_1pct")
    tpr_01 = test.get("tpr_at_fpr_0_1pct")
    if auc is None or tpr_1 is None or tpr_01 is None:
        return False
    return auc >= min_auc and tpr_1 >= min_tpr_1 and tpr_01 >= min_tpr_01


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--fusion-dir", required=True)
    parser.add_argument(
        "--cascade-dir",
        default=None,
        help="Directory holding semantic_probe_threshold.json and cascade_policy.json "
             "(from evaluate_cascade.py). Required when the bundle ships Engine C.",
    )
    args = parser.parse_args()
    bundle, fusion = Path(args.bundle).resolve(), Path(args.fusion_dir).resolve()
    manifest_path = bundle / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads((fusion / "meta_results.json").read_text(encoding="utf-8"))
    if manifest["release_id"] != results["release_id"]:
        raise ValueError("Fusion and neural model release IDs differ")
    srm_name = "srm_forensic_cnn_int8.tflite"
    visual = manifest["models"][srm_name]
    if results["identity"]["visual_model_hash"] != visual["sha256"] or results["identity"]["dataset_manifest_hash"] != manifest["data_manifest_hashes"]["visual"]:
        raise ValueError("Fusion model/data lineage differs from staged neural release")
    constant = {"features": results["features"], "decision_threshold": results["calibration"]["threshold"],
                "calibration_auc": results["calibration"]["auc"], "test_auc": results["test_auc"],
                "test_accuracy": results["test_accuracy"], "n_trees": results["n_trees"],
                "base_margin_intercept": results["base_margin_intercept"], "release_id": results["release_id"],
                "fusion_model_hash": results["model_hash"], "visual_model_hash": visual["sha256"]}
    (bundle / "constants" / "meta_classifier.json").write_text(json.dumps(constant, indent=2), encoding="utf-8")
    (bundle / "fusion").mkdir(exist_ok=True)
    for source, relative in [("natural_diff_histogram.json", "constants/natural_diff_histogram.json"),
                             ("XGBoostMetaClassifier.kt", "fusion/XGBoostMetaClassifier.kt"),
                             ("xgboost_meta.json", "fusion/xgboost_meta.json"), ("meta_results.json", "fusion/meta_results.json")]:
        shutil.copy2(fusion / source, bundle / relative)
    hashes = {"algorithm": "sha256", "models": {name: data["sha256"] for name, data in manifest["models"].items()}}
    (bundle / "constants" / "model_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    # ── Engine C ("Lite C") constants ──────────────────────────────────────
    # The probe threshold and the measured cascade policy are hash-bound
    # artifacts, exactly like the fusion threshold. They are attached only
    # when the release actually ships the second engine.
    if CASCADE_MODEL in manifest["models"]:
        cascade_model = manifest["models"][CASCADE_MODEL]
        # The fusion trainer writes the measured policy and the probe
        # threshold next to the fusion artifacts.
        for source, relative in [("semantic_probe_threshold.json", "constants/semantic_probe_threshold.json"),
                                 ("cascade_policy.json", "constants/cascade_policy.json")]:
            origin = fusion / source
            if not origin.is_file():
                raise ValueError(
                    f"Missing cascade artifact: {origin}. The fusion trainer writes it "
                    f"when --semantic-tflite is supplied."
                )
            shutil.copy2(origin, bundle / relative)
        probe_constant = json.loads((bundle / "constants" / "semantic_probe_threshold.json").read_text(encoding="utf-8"))
        if probe_constant.get("model_hash") != cascade_model["sha256"]:
            raise ValueError("Cascade threshold is bound to a different model")
        manifest["calibration"][CASCADE_MODEL] = {
            "model_hash": cascade_model["sha256"],
            "frontend_id": cascade_model["frontend_id"],
            "threshold": probe_constant["threshold_at_eer"],
            "auc": probe_constant["auc"],
            "n_calibration": probe_constant["n_calibration"],
            "split_manifest_hash": probe_constant["split_manifest_hash"],
        }

    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path != manifest_path and path.parts[-2] != "models":
            manifest["artifacts"][path.relative_to(bundle).as_posix()] = sha256_file(path)
    manifest["fusion_revision"] = results["release_id"]
    # The FUSION's calibration belongs to the fusion, under its own key.
    #
    # The previous code wrote it into `calibration[srm_name]`, so the
    # manifest recorded the FUSION's operating point as Engine A's and
    # Engine A's own INT8 calibration (auc 0.6033, threshold 0.53125)
    # existed nowhere in the release. The SRM key now keeps whatever
    # export_all_models.py wrote - Engine A's own calibration.
    manifest["calibration"]["fusion_xgboost"] = dict(
        results["calibration"],
        model_hash=results["model_hash"],
        frontend_id=visual["frontend_id"],
    )
    # Record what was measured against both the ENFORCED floor and the
    # ASPIRATIONAL target, so the gap between the shipped system and the
    # literature is visible in the release itself and not only in a document.
    acceptance = {}
    for name, entry in (read_object(bundle / "evaluation_report.json").get("models") or {}).items():
        test = entry.get("test") if isinstance(entry, dict) else None
        if not isinstance(test, dict):
            continue
        acceptance[name] = {
            "measured_auc": test.get("auc"),
            "measured_tpr_at_fpr_1pct": test.get("tpr_at_fpr_1pct"),
            "measured_tpr_at_fpr_0_1pct": test.get("tpr_at_fpr_0_1pct"),
            "enforced_floor": list(MINIMUM_ACCEPTANCE.get(name, ())),
            "academic_target": list(ACADEMIC_TARGETS.get(name, ())),
            "meets_academic_target": _meets(test, ACADEMIC_TARGETS.get(name)),
        }
    fusion_test = results.get("test") or {}
    acceptance["fusion_xgboost"] = {
        "measured_auc": fusion_test.get("auc"),
        "measured_tpr_at_fpr_1pct": fusion_test.get("tpr_at_fpr_1pct"),
        "measured_tpr_at_fpr_0_1pct": fusion_test.get("tpr_at_fpr_0_1pct"),
        "enforced_floor": list(MINIMUM_FUSION_ACCEPTANCE),
        "academic_target": list(ACADEMIC_TARGETS.get("fusion_xgboost", ())),
        "meets_academic_target": _meets(fusion_test, ACADEMIC_TARGETS.get("fusion_xgboost")),
    }
    manifest["acceptance"] = acceptance
    manifest["validation_status"] = "desktop_pipeline_passed_requires_independent_device_evaluation"
    manifest["validated"] = False
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(f"Measured constants attached to {bundle}; mobile validation remains required.")

if __name__ == "__main__":
    main()
