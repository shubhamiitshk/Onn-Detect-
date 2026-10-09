"""Validate a complete content-addressed release and install only validated bundles.

The default is inspection. --install explicitly replaces an assets directory
after staging a complete copy, retaining a recoverable previous directory.
An independent device report is required to promote desktop exports.
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data.contracts import sha256_file
from data.image_frontend import frontend_id

MODEL_NAMES = {'srm_forensic_cnn_int8.tflite', 'lcnn_audio_int8.tflite'}
REQUIRED_ARTIFACTS = {'constants/lcnn_audio_threshold.json', 'constants/meta_classifier.json',
                      'constants/natural_diff_histogram.json', 'constants/model_hashes.json',
                      'fusion/XGBoostMetaClassifier.kt', 'fusion/xgboost_meta.json',
                      'fusion/meta_results.json', 'evaluation_report.json', 'source_snapshot.json'}


def read_object(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value

def validate_bundle(bundle, require_validated=False):
    bundle = Path(bundle).resolve()
    # Check every entry before opening the manifest: even its own path or an
    # unlisted empty directory must not redirect reads/copies outside a bundle.
    if any(path.is_symlink() for path in bundle.rglob('*')):
        raise ValueError('Symlinked release entries are unsupported')
    manifests = [p for p in (bundle / 'release_manifest.json', bundle / 'constants/release_manifest.json') if p.is_file()]
    if len(manifests) != 1:
        raise ValueError('Bundle must have exactly one release manifest')
    manifest_path = manifests[0]
    manifest_relative = manifest_path.relative_to(bundle).as_posix()
    manifest = read_object(manifest_path)
    if not isinstance(manifest, dict) or type(manifest.get('validated')) is not bool:
        raise ValueError('validated must be an explicit Boolean')
    if type(manifest.get('schema_version')) is not int or manifest.get('schema_version') != 1 or manifest.get('score_convention') != 'real=0_fake=1':
        raise ValueError('Unsupported release schema/polarity')
    if not isinstance(manifest.get('release_id'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', manifest['release_id']):
        raise ValueError('Release ID missing')
    models = manifest.get('models', {})
    if not isinstance(models, dict) or set(models) != MODEL_NAMES:
        raise ValueError('Release must contain exactly the required neural models')
    declared = set()
    for name, model in models.items():
        if not isinstance(model, dict):
            raise ValueError('Model entries must be JSON objects')
        if '/' in name or '\\' in name:
            raise ValueError('Model name must be an asset basename')
        path = bundle / 'models' / name
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != bundle and parent.is_relative_to(bundle)):
            raise ValueError('Symlinked model paths are unsupported')
        with path.open('rb') as stream:
            header = stream.read(8)
        if len(header) < 8 or header[4:8] != b'TFL3':
            raise ValueError(f'Invalid TFLite header: {name}')
        if model.get('sha256') != sha256_file(path):
            raise ValueError(f'Model checksum mismatch: {name}')
        shape = model.get('input_shape')
        if model.get('dtype') != 'FLOAT32' or model.get('output_shape') != [1, 1] or any(type(n) is not int for n in model.get('output_shape', [])) or not isinstance(shape, list) or len(shape) != 4 or not all(type(n) is int and n > 0 for n in shape):
            raise ValueError(f'Invalid model tensor contract: {name}')
        if name.startswith('lcnn'):
            expected_frontend = 'lfcc-v3-sr16000-f128-c40-nfft512-hop160-topdbnone-resampler-quadratic_window_sinc_v1-float64stft-float32'
            if shape != [1, 401, 40, 1]:
                raise ValueError('Unsupported audio shape')
        else:
            if shape[0] != 1 or shape[1] != shape[2] or shape[3] != 3 or not 32 <= shape[1] <= 1024:
                raise ValueError('Unsupported visual shape')
            expected_frontend = frontend_id(shape[1])
        if model.get('frontend_id') != expected_frontend:
            raise ValueError(f'Frontend mismatch: {name}')
        declared.add('models/' + name)
    artifacts = manifest.get('artifacts', {})
    if not isinstance(artifacts, dict):
        raise ValueError('artifacts must be a JSON object')
    if not REQUIRED_ARTIFACTS <= set(artifacts):
        raise ValueError(f'Incomplete bundle: missing {sorted(REQUIRED_ARTIFACTS - set(artifacts))}')
    for relative, digest in artifacts.items():
        from pathlib import PurePosixPath
        if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise ValueError('Invalid artifact path')
        raw_path = bundle / relative
        if raw_path.is_symlink() or any(parent.is_symlink() for parent in raw_path.parents if parent != bundle and parent.is_relative_to(bundle)):
            raise ValueError('Symlinked artifact paths are unsupported')
        path = raw_path.resolve()
        if not path.is_relative_to(bundle) or relative in ('release_manifest.json', 'constants/release_manifest.json'):
            raise ValueError('Invalid artifact path')
        if not re.fullmatch('[0-9a-f]{64}', str(digest)) or sha256_file(path) != digest:
            raise ValueError(f'Artifact checksum mismatch: {relative}')
        declared.add(relative)
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
    if actual != declared | {manifest_relative}:
        raise ValueError(f'Unmanifested/missing release files: {actual ^ (declared | {manifest_relative})}')
    if sha256_file(bundle / 'evaluation_report.json') != manifest.get('evaluation_report_hash'):
        raise ValueError('Evaluation report linkage mismatch')
    report = read_object(bundle / 'evaluation_report.json')
    if report.get('release_id') != manifest['release_id']:
        raise ValueError('Evaluation release ID mismatch')
    snapshot_path = bundle / 'source_snapshot.json'
    if sha256_file(snapshot_path) != manifest.get('source_snapshot_sha256') or report.get('source_snapshot_sha256') != manifest.get('source_snapshot_sha256'):
        raise ValueError('Actual source snapshot linkage is missing')
    snapshot = read_object(snapshot_path)
    if not isinstance(snapshot.get('files'), dict) or not snapshot['files'] or snapshot.get('source_commit') != manifest.get('source_commit') or type(snapshot.get('source_dirty')) is not bool:
        raise ValueError('Source snapshot is missing actual file hashes/dirty status')
    if any(not re.fullmatch('[0-9a-f]{64}', str(digest)) for digest in snapshot['files'].values()):
        raise ValueError('Invalid source snapshot file hash')
    report_models = report.get('models', {})
    if not isinstance(report_models, dict):
        raise ValueError('Evaluation models must be a JSON object')
    for name in MODEL_NAMES:
        entry = report_models.get(name)
        parity = entry.get('parity') if isinstance(entry, dict) else None
        if not isinstance(parity, dict) or parity.get('passed') is not True:
            raise ValueError(f'Missing enforced export validation: {name}')
        calibration = entry.get('calibration')
        if not isinstance(calibration, dict) or calibration.get('model_hash') != models[name]['sha256'] or calibration.get('frontend_id') != models[name]['frontend_id']:
            raise ValueError(f'Evaluation report model/frontend lineage differs: {name}')
    if not re.fullmatch('[0-9a-f]{40}', str(manifest.get('source_commit', ''))):
        raise ValueError('Source commit unavailable')
    data_hashes = manifest.get('data_manifest_hashes')
    if not isinstance(data_hashes, dict) or not data_hashes or any(not re.fullmatch('[0-9a-f]{64}', str(v)) for v in data_hashes.values()):
        raise ValueError('Dataset lineage unavailable')
    calibrations = manifest.get('calibration')
    if not isinstance(calibrations, dict):
        raise ValueError('Calibration must be a JSON object')
    for name, model in models.items():
        cal = calibrations.get(name, {})
        if not isinstance(cal, dict):
            raise ValueError('Model calibration must be a JSON object')
        if cal.get('model_hash') != model['sha256'] or cal.get('frontend_id') != model['frontend_id']:
            raise ValueError(f'Calibration/model lineage mismatch: {name}')
        if type(cal.get('threshold')) not in (int, float) or not 0 <= cal['threshold'] <= 1 or type(cal.get('auc')) not in (int, float) or not .5 < cal['auc'] <= 1 or type(cal.get('n_calibration')) is not int or cal['n_calibration'] < 100:
            raise ValueError(f'Calibration is missing, insufficient or at/below chance: {name}')
        if cal.get('split_manifest_hash') not in manifest['data_manifest_hashes'].values():
            raise ValueError('Calibration split differs from recorded dataset')
    meta = read_object(bundle / 'constants/meta_classifier.json')
    if meta.get('decision_threshold') != manifest['calibration']['srm_forensic_cnn_int8.tflite']['threshold']:
        raise ValueError('Fusion operating threshold mismatch')
    audio = read_object(bundle / 'constants/lcnn_audio_threshold.json')
    if audio.get('threshold_at_eer') != manifest['calibration']['lcnn_audio_int8.tflite']['threshold']:
        raise ValueError('Audio operating threshold mismatch')
    kotlin = (bundle / 'fusion/XGBoostMetaClassifier.kt').read_text(encoding='utf-8')
    revision = manifest.get('fusion_revision')
    if revision != manifest['release_id'] or f'const val RELEASE_ID = {json.dumps(revision)}' not in kotlin:
        raise ValueError('Compiled fusion source release ID mismatch')
    if require_validated or manifest.get('validated') is True:
        device = manifest.get('device_validation', {})
        if manifest.get('validated') is not True or not isinstance(device, dict) or device.get('passed') is not True:
            raise ValueError('Independent device evaluation has not passed; bundle cannot be installed/distributed')
        device_path = bundle / 'device_evaluation.json'
        if not device_path.is_file() or device.get('report_hash') != sha256_file(device_path):
            raise ValueError('Device evaluation report linkage missing')
        device_report = read_object(device_path)
        if device_report.get('release_id') != manifest['release_id'] or device_report.get('passed') is not True:
            raise ValueError('Device report belongs to another/failed release')
        if device_report.get('source_snapshot_sha256') != manifest.get('source_snapshot_sha256'):
            raise ValueError('Device report source snapshot differs from this release')
        if device_report.get('fusion_kotlin_sha256') != artifacts['fusion/XGBoostMetaClassifier.kt']:
            raise ValueError('Device report compiled fusion source differs from this release')
        device_models = device_report.get('models')
        if not isinstance(device_models, dict):
            raise ValueError('Device models must be a JSON object')
        for name, model in models.items():
            entry = device_models.get(name, {})
            if not isinstance(entry, dict) or entry.get('sha256') != model['sha256'] or entry.get('frontend_id') != model['frontend_id'] or entry.get('parity_passed') is not True:
                raise ValueError(f'Device model/preprocessing parity missing: {name}')
        if device_report.get('fusion_revision') != revision or device_report.get('kotlin_parity_passed') is not True or device_report.get('offline_verified') is not True:
            raise ValueError('Device fusion/offline verification missing')
    return manifest


def attach_device_report(bundle, report_path):
    """Promote a complete candidate, preserving the original on any failure.

    A staged sibling keeps directory renames on the same filesystem. The old
    complete bundle remains recoverable beside the promoted version.
    """
    bundle = Path(bundle).resolve()
    manifest = validate_bundle(bundle)
    if manifest['validated']:
        raise ValueError('This release already has a device report; use a new release version')
    report = read_object(report_path)
    manifest_relative = 'release_manifest.json' if (bundle / 'release_manifest.json').is_file() else 'constants/release_manifest.json'
    backup = bundle.with_name(bundle.name + '.previous-device-' + manifest['release_id'])
    if backup.exists():
        raise FileExistsError('Previous device-promotion bundle exists; choose a new release version')
    with tempfile.TemporaryDirectory(prefix=bundle.name + '.device-', dir=bundle.parent) as temp:
        candidate = Path(temp) / 'bundle'
        shutil.copytree(bundle, candidate)
        target = candidate / 'device_evaluation.json'
        target.write_text(json.dumps(report, indent=2), encoding='utf-8')
        digest = sha256_file(target)
        manifest['device_validation'] = {'passed': True, 'report_hash': digest}
        manifest['artifacts']['device_evaluation.json'] = digest
        manifest['validated'] = True
        manifest['validation_status'] = 'desktop_and_independent_device_validation_passed'
        (candidate / manifest_relative).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        validate_bundle(candidate, True)
        os.rename(bundle, backup)
        try:
            os.rename(candidate, bundle)
        except Exception:
            os.rename(backup, bundle)
            raise
    return backup

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--require-validated', action='store_true')
    parser.add_argument('--device-report', help='Attach independently measured phone report; then validate')
    parser.add_argument('--install', help='Destination assets directory; a validated bundle is required')
    parser.add_argument('--kotlin-destination', help='Explicit compiled source path paired with --install')
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    if args.device_report:
        backup = attach_device_report(bundle, args.device_report)
        print(f'Device report attached; previous complete bundle retained at {backup}')
    manifest = validate_bundle(bundle, args.require_validated or bool(args.install))
    if args.install:
        if not args.kotlin_destination:
            raise ValueError('Installing image fusion also requires --kotlin-destination')
        destination = Path(args.install).resolve()
        if destination == bundle or destination.is_relative_to(bundle) or bundle.is_relative_to(destination):
            raise ValueError('Install destination and bundle must not overlap')
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = Path(tempfile.mkdtemp(prefix=destination.name + '.install-', dir=destination.parent)) / 'assets'
        shutil.copytree(bundle, staged)
        if (staged / 'release_manifest.json').exists():
            os.rename(staged / 'release_manifest.json', staged / 'constants/release_manifest.json')
        backup = destination.with_name(destination.name + '.previous-' + manifest['release_id'])
        failed = destination.with_name(destination.name + '.failed-' + manifest['release_id'])
        if backup.exists() or failed.exists():
            raise FileExistsError('Previous/failed release already exists; choose a new release ID')
        source = Path(args.kotlin_destination).resolve()
        if source.name != 'XGBoostMetaClassifier.kt' or source.is_relative_to(bundle) or source.is_relative_to(destination):
            raise ValueError('Compiled fusion destination must be an explicit separate XGBoostMetaClassifier.kt source file')
        old_source = source.read_bytes() if source.exists() else None
        source.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=source.name + '.install-', suffix='.kt', dir=source.parent, delete=False) as stream:
            source_temp = Path(stream.name)
            stream.write((bundle / 'fusion/XGBoostMetaClassifier.kt').read_bytes())
        moved_previous = False
        try:
            if destination.exists():
                os.rename(destination, backup)
                moved_previous = True
            os.rename(staged, destination)
            os.replace(source_temp, source)
        except Exception:
            if destination.exists() and not staged.exists():
                os.rename(destination, failed)
            if moved_previous and backup.exists():
                os.rename(backup, destination)
            if old_source is not None:
                source.write_bytes(old_source)
            elif source.exists():
                os.rename(source, source.with_name(source.name + '.failed-' + manifest['release_id']))
            raise
        print(f'Installed validated release {manifest["release_id"]}; previous assets retained at {backup}')
    else:
        print(f'Release {manifest["release_id"]}: complete hashes/lineage checked; validated={manifest["validated"]}')

if __name__ == '__main__':
    main()
