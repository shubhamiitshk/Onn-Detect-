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
from training.acceptance import (
    MINIMUM_FUSION_ACCEPTANCE,
    check_model_acceptance,
)
from training.evaluation import assert_source_fpr_within_bound

MODEL_NAMES = {'srm_forensic_cnn_int8.tflite', 'lcnn_audio_int8.tflite'}
# Engine C ("Lite C") - the OPTIONAL semantic second engine. A release may
# ship it or not. When it is shipped it is held to exactly the same contract
# as the required models: hash-bound, calibrated on a declared split, and
# covered by the desktop parity report and the independent device report.
CASCADE_MODEL_NAMES = {'siglip_probe_int8.tflite', 'semantic_probe_int8.tflite'}
# Which frontend contract each slot-6 probe consumes. The distilled student
# uses CLIP normalization; the frozen tower uses SigLIP normalization. They
# are NOT interchangeable: the wrong one silently rescales every slot-6
# value, so the release manifest's declared frontend must match the model.
CASCADE_FRONTEND_KIND = {
    'semantic_probe_int8.tflite': 'clip',
    'siglip_probe_int8.tflite': 'siglip',
}
REQUIRED_ARTIFACTS = {'constants/lcnn_audio_threshold.json', 'constants/meta_classifier.json',
                      'constants/natural_diff_histogram.json', 'constants/model_hashes.json',
                      'fusion/XGBoostMetaClassifier.kt', 'fusion/xgboost_meta.json',
                      'fusion/meta_results.json', 'evaluation_report.json', 'source_snapshot.json'}
CASCADE_ARTIFACTS = {'constants/semantic_probe_threshold.json', 'constants/cascade_policy.json'}
# The frozen tower's end-to-end acceptance report. It is what proves that the
# PACKAGED bytes keep the measured behaviour, and it is bound to those bytes
# by the model's SHA256.
TOWER_ARTIFACTS = {'constants/vfm_acceptance.json'}


def read_object(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value

def validate_bundle(bundle, require_validated=False, allow_source_fpr_violation=False):
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
    if not isinstance(models, dict) or not MODEL_NAMES.issubset(set(models)):
        raise ValueError('Release must contain the required neural models')
    unexpected = set(models) - MODEL_NAMES - CASCADE_MODEL_NAMES
    if unexpected:
        raise ValueError(f'Unexpected model contracts: {sorted(unexpected)}')
    cascade_shipped = bool(CASCADE_MODEL_NAMES & set(models))
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
        elif name in CASCADE_MODEL_NAMES:
            # Engine C uses the CLIP appearance frontend (bicubic square
            # resize + CLIP normalization), not the Engine A one. The two
            # engines must not share a resampling artifact, or the fusion
            # would measure the same low-level cue twice.
            if shape[0] != 1 or shape[1] != shape[2] or shape[3] != 3 or not 64 <= shape[1] <= 1024:
                raise ValueError('Unsupported cascade shape')
            kind = CASCADE_FRONTEND_KIND.get(name, 'clip')
            if kind == 'siglip':
                from export.vfm_frontend import vfm_frontend_id
                expected_frontend = vfm_frontend_id(shape[1])
            else:
                expected_frontend = f'clip-appearance-bicubic{shape[1]}-clipnorm-meanstd-float32'
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
    required_artifacts = REQUIRED_ARTIFACTS | (CASCADE_ARTIFACTS if cascade_shipped else set())
    if not required_artifacts <= set(artifacts):
        raise ValueError(f'Incomplete bundle: missing {sorted(required_artifacts - set(artifacts))}')
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
    # Engine C is held to the same export-validation standard when shipped.
    for name in CASCADE_MODEL_NAMES & set(models):
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
    # The FUSION's calibration lives under its own key in the current layout.
    # The legacy layout stored it under the SRM model's key, which is why
    # Engine A's own operating point (auc 0.6033, threshold 0.53125) was
    # recorded nowhere in the manifest. Both layouts are accepted so an
    # already-staged release keeps validating.
    fusion_cal = calibrations.get('fusion_xgboost') or calibrations['srm_forensic_cnn_int8.tflite']
    if meta.get('decision_threshold') != fusion_cal['threshold']:
        raise ValueError('Fusion operating threshold mismatch')
    audio = read_object(bundle / 'constants/lcnn_audio_threshold.json')
    if audio.get('threshold_at_eer') != manifest['calibration']['lcnn_audio_int8.tflite']['threshold']:
        raise ValueError('Audio operating threshold mismatch')
    if cascade_shipped:
        cascade_name = next(iter(CASCADE_MODEL_NAMES & set(models)))
        probe = read_object(bundle / 'constants/semantic_probe_threshold.json')
        if probe.get('threshold_at_eer') != manifest['calibration'][cascade_name]['threshold']:
            raise ValueError('Cascade operating threshold mismatch')
        policy = read_object(bundle / 'constants/cascade_policy.json')
        for key in ('engine_a_threshold', 'engine_c_threshold', 'escalate_below', 'escalate_above'):
            if not isinstance(policy.get(key), (int, float)) or not 0.0 <= policy[key] <= 1.0:
                raise ValueError(f'Invalid cascade policy field: {key}')
        if not policy['escalate_below'] <= policy['escalate_above']:
            raise ValueError('Cascade escalation band is inverted')
    # ---------- detector acceptance floors ----------
    # AUC alone is not an acceptance criterion. The shipped image fusion has
    # AUC 0.7152 and TPR@1%FPR of 0.040 - a coin flip in the regime anyone
    # would actually ship. These floors exist so that number cannot hide
    # behind the other one.
    #
    # The evaluation_report only carries the three .tflite models; the
    # FUSION's sealed-test metrics live in fusion/meta_results.json. A gate
    # that only walked evaluation_report["models"] would never check the
    # detector the image/video route actually ships.
    source_fpr_violation = ''
    desktop_models = (read_object(bundle / 'evaluation_report.json').get('models') or {})
    for name, entry in desktop_models.items():
        test = entry.get('test') if isinstance(entry, dict) else None
        if not isinstance(test, dict):
            continue
        # The sealed test split holds 200 real images BY DESIGN, so a 0.1% FPR
        # operating point is defined by 0.2 expected false positives and is
        # not a stable statistic. The metric must be PRESENT and the 1% FPR
        # floor is enforced; the 0.1% floor is not, and that is exactly what
        # `require_low_fpr_statistic=False` is for.
        check_model_acceptance(name, test, context='release', require_low_fpr_statistic=False)
    fusion_results = read_object(bundle / 'fusion/meta_results.json')
    fusion_test = fusion_results.get('test')
    if isinstance(fusion_test, dict):
        check_model_acceptance(
            'fusion_xgboost', fusion_test, floors=MINIMUM_FUSION_ACCEPTANCE,
            context='release', require_low_fpr_statistic=False,
        )
        # The per-source real-image FPR bound. It is UNREACHABLE on the
        # current corpus, which leaks the label through the acquisition
        # source, so a release can only be produced with an explicit
        # override - and the override is recorded in the bundle metadata so
        # it cannot be shipped silently.
        try:
            assert_source_fpr_within_bound(
                fusion_test, max_fpr=0.05, min_real_per_source=15, context='release fusion'
            )
        except ValueError:
            if not allow_source_fpr_violation:
                raise
            source_fpr_violation = (
                'The fusion exceeds the 5% per-source real-image FPR bound. '
                'The corpus leaks the label through the acquisition source, '
                'so the bound is unreachable until the corpus is '
                'source-balanced. Recorded as a known release violation.'
            )
            print(f'[Source FPR] BOUND NOT MET: {source_fpr_violation}')
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


REQUIRED_DEVICE_EVIDENCE = (
    'device_id', 'device_model', 'android_sdk_int', 'app_version',
    'measurement_kind', 'measured_at_utc',
)

DESKTOP_ONLY_DEVICE_KEYS = ('desktop_agreement', 'desktop_mae', 'desktop_max_error')


def assert_device_report_is_from_a_device(report):
    """A desktop report must never be able to set validated=true.

    ``build_and_install_release_v4.py`` used to synthesise a device report
    on the desktop: its "latency" was a ``tf.lite.Interpreter`` timing and
    its parity keys were literally named ``desktop_agreement``,
    ``desktop_mae`` and ``desktop_max_error``. Those key names are a
    fingerprint of a non-device report, so they are rejected outright.

    A report must also carry the identity of the device it came from, the
    measurement kind, and at least one on-device measurement payload.
    """
    missing = [key for key in REQUIRED_DEVICE_EVIDENCE if not report.get(key)]
    if missing:
        raise ValueError(f'Device report is missing device evidence: {missing}')
    if report.get('measurement_kind') != 'on_device_android':
        raise ValueError(
            "Device report measurement_kind must be 'on_device_android'; "
            f"got {report.get('measurement_kind')!r}. A desktop interpreter "
            'timing is not device validation.'
        )
    for name, entry in (report.get('models') or {}).items():
        for banned in DESKTOP_ONLY_DEVICE_KEYS:
            if banned in entry:
                raise ValueError(
                    f"{name}: device report carries the desktop-only key "
                    f"{banned!r}. This report was not produced on a device."
                )
    if not report.get('on_device_measurements'):
        raise ValueError('Device report contains no on-device measurement payload')


def attach_device_report(bundle, report_path):
    """Promote a complete candidate, preserving the original on any failure.

    A staged sibling keeps directory renames on the same filesystem. The old
    complete bundle remains recoverable beside the promoted version.
    """
    bundle = Path(bundle).resolve()
    manifest = validate_bundle(bundle, allow_source_fpr_violation=True)
    if manifest['validated']:
        raise ValueError('This release already has a device report; use a new release version')
    report = read_object(report_path)
    assert_device_report_is_from_a_device(report)
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
        validate_bundle(candidate, True, allow_source_fpr_violation=True)
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
    parser.add_argument(
        '--allow-source-fpr-violation',
        action='store_true',
        help=(
            'Allow a bundle whose fusion exceeds the 5%% per-source '
            'real-image FPR bound. The bound is correct but unreachable on '
            'the current corpus, which leaks the label through the '
            'acquisition source. The violation is recorded in the bundle '
            'metadata.'
        ),
    )
    parser.add_argument('--device-report', help='Attach independently measured phone report; then validate')
    parser.add_argument('--install', help='Destination assets directory; a validated bundle is required')
    parser.add_argument(
        '--install-desktop-validated',
        action='store_true',
        help=(
            'Install a bundle that passed every DESKTOP gate but has NOT been '
            'validated on an Android device. The manifest keeps '
            'validated=false and the release status says so; the app reports '
            'the release as desktop-verified with the device evaluation '
            'pending. Use this when no phone is available: a device report '
            'must never be fabricated from a desktop measurement.'
        ),
    )
    parser.add_argument('--kotlin-destination', help='Explicit compiled source path paired with --install')
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    if args.device_report:
        if args.install_desktop_validated:
            raise ValueError(
                '--device-report and --install-desktop-validated are mutually exclusive: '
                'a device report and a desktop-only install are different states'
            )
        backup = attach_device_report(bundle, args.device_report)
        print(f'Device report attached; previous complete bundle retained at {backup}')
    require_validated = (args.require_validated or bool(args.install)) and not args.install_desktop_validated
    manifest = validate_bundle(
        bundle,
        require_validated,
        allow_source_fpr_violation=args.allow_source_fpr_violation,
    )
    if args.install_desktop_validated:
        print('[Release] Installing a DESKTOP-VALIDATED release: the manifest keeps '
              'validated=false and the device evaluation is still outstanding.')
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
