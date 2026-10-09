"""Reproducible staged training/export pipeline; never installs an unvalidated release."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import yaml
ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT / "src"))
from data.contracts import validate_dataset_manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-config", required=True)
    parser.add_argument("--audio-config", required=True)
    parser.add_argument("--fusion-config", default=str(ML_ROOT / "configs/xgboost_meta.yaml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", help="Parent of prepared dataset folders; relative config roots use their basename here")
    parser.add_argument("--release-id", default=None)
    parser.add_argument("--skip-training", action="store_true", help="Use already verified checkpoints in this output directory")
    args = parser.parse_args()
    try:
        repository = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=ML_ROOT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('Full staged releases require Git and the complete checkout; mount the repository when using Docker') from exc
    if Path(repository).resolve() != ML_ROOT.parent.resolve():
        raise ValueError('Training pipeline must run from the complete repository containing this ml/ directory')
    output = Path(args.output_dir).resolve()
    if (output / 'bundle').exists() or (output / 'fusion').exists():
        raise FileExistsError('Use a new run directory; existing staged release artifacts are never replaced')
    if not args.skip_training and (output / 'checkpoints').exists() and any((output / 'checkpoints').iterdir()):
        raise FileExistsError('Existing checkpoints require --skip-training or an explicit individual --resume; use a new run directory')
    configs = {}
    for modality, path in [("visual", args.visual_config), ("audio", args.audio_config)]:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        root = Path(config["data"]["root_dir"])
        if not root.is_absolute():
            root = Path(args.data_root).resolve() / root.name if args.data_root else (ML_ROOT / root).resolve()
        config["data"]["root_dir"] = str(root)
        required = ("train", "val", "calibration", "test", "stacking", "fusion_val") if modality == "visual" else ("train", "val", "calibration", "test")
        # Preflight both corpora before expensive training starts.
        validate_dataset_manifest(root, required)
        configs[modality] = config
    output.mkdir(parents=True, exist_ok=True)
    effective = output / "effective_configs"
    effective.mkdir(exist_ok=True)
    checkpoints = output / "checkpoints"
    for modality, config in configs.items():
        path = effective / (modality + ".yaml")
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        if not args.skip_training:
            script = "train_srm_cnn.py" if modality == "visual" else "train_lcnn_audio.py"
            subprocess.run([sys.executable, str(ML_ROOT / "scripts" / script), "--config", str(path), "--device", args.device,
                            "--output-dir", str(checkpoints)], check=True)
    from datetime import datetime, timezone
    release_id = args.release_id or datetime.now(timezone.utc).strftime("onn-%Y%m%dT%H%M%SZ")
    bundle, fusion = output / "bundle", output / "fusion"
    visual_checkpoint = checkpoints / "srm_cnn_genimage_best.pth"
    audio_checkpoint = checkpoints / "lcnn_audio_asvspoof_best.pth"
    def call(script, *arguments):
        subprocess.run([sys.executable, str(ML_ROOT / "scripts" / script), *map(str, arguments)], check=True)
    call("export_all_models.py", "--visual-checkpoint", visual_checkpoint, "--audio-checkpoint", audio_checkpoint,
         "--output-dir", bundle, "--release-id", release_id)
    call("train_xgboost_meta.py", "--config", args.fusion_config, "--visual-checkpoint", visual_checkpoint,
         "--visual-tflite", bundle / "models/srm_forensic_cnn_int8.tflite", "--output-dir", fusion, "--release-id", release_id)
    call("generate_android_constants.py", "--bundle", bundle, "--fusion-dir", fusion)
    call("release_bundle.py", "--bundle", bundle, "--validate-only")
    print(f"Complete staged release {bundle}; validated=false until independently measured device report is attached.")

if __name__ == "__main__":
    main()
