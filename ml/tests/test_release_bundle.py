"""Release boundaries: malformed contracts and failed install transactions."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import release_bundle
from data.contracts import sha256_file

def fixture(root):
    """Non-distributable fixture; no assertion of learned model accuracy."""
    root.mkdir(parents=True)
    for directory in ('models', 'constants', 'fusion'):
        (root / directory).mkdir()
    models = {}
    calibrations = {}
    frontends = ['rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-size32-float32',
                 'lfcc-v3-sr16000-f128-c40-nfft512-hop160-topdbnone-resampler-quadratic_window_sinc_v1-float64stft-float32']
    for index, name in enumerate(sorted(release_bundle.MODEL_NAMES, reverse=True)):
        path = root / 'models' / name
        path.write_bytes(b'\0\0\0\0TFL3' + bytes(range(32)))
        shape = [1, 32, 32, 3] if name.startswith('srm') else [1, 401, 40, 1]
        frontend = frontends[0] if name.startswith('srm') else frontends[1]
        models[name] = {'sha256': sha256_file(path), 'input_shape': shape, 'output_shape': [1, 1],
                        'dtype': 'FLOAT32', 'frontend_id': frontend}
        calibrations[name] = {'model_hash': sha256_file(path), 'frontend_id': frontend, 'threshold': .4,
                              'auc': .8, 'n_calibration': 100, 'split_manifest_hash': 'a' * 64}
    values = {'constants/meta_classifier.json': {'decision_threshold': .4},
              'constants/lcnn_audio_threshold.json': {'threshold_at_eer': .4},
              'constants/natural_diff_histogram.json': {'bins': [.1]},
              'constants/model_hashes.json': {}, 'fusion/xgboost_meta.json': {}, 'fusion/meta_results.json': {},
              'evaluation_report.json': {'release_id': 'test-release', 'models': {name: {'parity': {'passed': True},
                  'calibration': {'model_hash': value['sha256'], 'frontend_id': value['frontend_id']}}
                  for name, value in models.items()}}}
    for relative, value in values.items():
        (root / relative).write_text(json.dumps(value), encoding='utf-8')
    (root / 'fusion/XGBoostMetaClassifier.kt').write_text('companion object { const val RELEASE_ID = "test-release" }', encoding='utf-8')
    snapshot = root / 'source_snapshot.json'
    snapshot.write_text(json.dumps({'files': {'ml/src/model.py': 'c' * 64}, 'source_commit': 'b' * 40, 'source_dirty': True}), encoding='utf-8')
    report = values['evaluation_report.json']
    report['source_snapshot_sha256'] = sha256_file(snapshot)
    (root / 'evaluation_report.json').write_text(json.dumps(report), encoding='utf-8')
    manifest = {'source_snapshot_sha256': sha256_file(snapshot), 'schema_version': 1, 'release_id': 'test-release', 'validated': False,
                'score_convention': 'real=0_fake=1', 'models': models, 'calibration': calibrations,
                'source_commit': 'b' * 40, 'data_manifest_hashes': {'visual': 'a' * 64},
                'evaluation_report_hash': sha256_file(root / 'evaluation_report.json'), 'fusion_revision': 'test-release',
                'artifacts': {path.relative_to(root).as_posix(): sha256_file(path) for path in root.rglob('*') if path.is_file()}}
    (root / 'release_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return manifest


def device_report(path, manifest):
    path.write_text(json.dumps({'release_id': manifest['release_id'], 'passed': True,
                               'source_snapshot_sha256': manifest['source_snapshot_sha256'],
                               'fusion_kotlin_sha256': manifest['artifacts']['fusion/XGBoostMetaClassifier.kt'],
                               'fusion_revision': manifest['fusion_revision'], 'kotlin_parity_passed': True,
                               'offline_verified': True,
                               'models': {name: {'sha256': value['sha256'], 'frontend_id': value['frontend_id'],
                                                 'parity_passed': True} for name, value in manifest['models'].items()}}), encoding='utf-8')

class ReleaseTests(unittest.TestCase):
    def test_report_cannot_claim_parity_for_different_model_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'bundle'
            manifest = fixture(root)
            report_path = root / 'evaluation_report.json'
            report = json.loads(report_path.read_text(encoding='utf-8'))
            report['models']['srm_forensic_cnn_int8.tflite']['calibration']['model_hash'] = '0' * 64
            report_path.write_text(json.dumps(report), encoding='utf-8')
            manifest['evaluation_report_hash'] = sha256_file(report_path)
            manifest['artifacts']['evaluation_report.json'] = sha256_file(report_path)
            (root / 'release_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                release_bundle.validate_bundle(root)

    def test_same_release_id_cannot_reuse_other_source_or_fusion_evidence(self):
        for field in ('source_snapshot_sha256', 'fusion_kotlin_sha256'):
            with tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                root = parent / 'bundle'
                manifest = fixture(root)
                report_path = parent / 'phone-report.json'
                device_report(report_path, manifest)
                report = json.loads(report_path.read_text(encoding='utf-8'))
                report[field] = '0' * 64
                report_path.write_text(json.dumps(report), encoding='utf-8')
                with self.assertRaises(ValueError):
                    release_bundle.attach_device_report(root, report_path)
                self.assertFalse(release_bundle.validate_bundle(root)['validated'])
                self.assertFalse((root / 'device_evaluation.json').exists())

    def test_device_promotion_supports_android_manifest_location(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / 'bundle'
            manifest = fixture(root)
            (root / 'release_manifest.json').rename(root / 'constants/release_manifest.json')
            report = parent / 'phone-report.json'
            device_report(report, manifest)
            backup = release_bundle.attach_device_report(root, report)
            self.assertTrue(release_bundle.validate_bundle(root, True)['validated'])
            self.assertFalse(release_bundle.validate_bundle(backup)['validated'])
            self.assertTrue((root / 'constants/release_manifest.json').is_file())
            self.assertFalse((backup / 'device_evaluation.json').exists())

    def test_failed_device_promotion_preserves_original_complete_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / 'bundle'
            manifest = fixture(root)
            original = (root / 'release_manifest.json').read_bytes()
            report = parent / 'phone-report.json'
            device_report(report, manifest)
            rename = release_bundle.os.rename
            def fail_candidate(origin, target):
                if Path(target) == root and Path(origin).parent != parent:
                    raise OSError('simulated candidate directory replacement failure')
                return rename(origin, target)
            with patch.object(release_bundle.os, 'rename', side_effect=fail_candidate):
                with self.assertRaises(OSError):
                    release_bundle.attach_device_report(root, report)
            self.assertEqual((root / 'release_manifest.json').read_bytes(), original)
            self.assertFalse((root / 'device_evaluation.json').exists())
            self.assertFalse(release_bundle.validate_bundle(root)['validated'])

    def test_external_manifest_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / 'bundle'
            fixture(root)
            manifest = root / 'release_manifest.json'
            outside = parent / 'outside.json'
            manifest.rename(outside)
            try:
                manifest.symlink_to(outside)
            except OSError as error:
                self.skipTest(f'This Windows account cannot create symlinks: {error}')
            with self.assertRaises(ValueError):
                release_bundle.validate_bundle(root)

    def test_incomplete_and_unvalidated_release_cannot_distribute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'bundle'
            fixture(root)
            self.assertFalse(release_bundle.validate_bundle(root)['validated'])
            with self.assertRaises(ValueError):
                release_bundle.validate_bundle(root, True)
            (root / 'constants/meta_classifier.json').unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                release_bundle.validate_bundle(root)

    def test_string_boolean_float_shape_nan_and_path_escape_rejected(self):
        mutations = [lambda m: m.update(validated='false'),
                     lambda m: m['models']['srm_forensic_cnn_int8.tflite'].update(input_shape=[True, 32, 32, 3]),
                     lambda m: m['calibration']['lcnn_audio_int8.tflite'].update(threshold=float('nan')),
                     lambda m: m['artifacts'].update({'../outside': 'a' * 64})]
        for mutate in mutations:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'bundle'
                manifest = fixture(root)
                mutate(manifest)
                (root / 'release_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaises(ValueError):
                    release_bundle.validate_bundle(root)

    def test_successful_install_puts_manifest_in_android_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            bundle, destination = parent / 'bundle', parent / 'assets'
            manifest = fixture(bundle)
            manifest['validated'] = True
            source = parent / 'XGBoostMetaClassifier.kt'
            arguments = ['release_bundle', '--bundle', str(bundle), '--install', str(destination), '--kotlin-destination', str(source)]
            with patch.object(release_bundle, 'validate_bundle', return_value=manifest), patch.object(sys, 'argv', arguments):
                release_bundle.main()
            self.assertTrue((destination / 'constants/release_manifest.json').is_file())
            self.assertFalse((destination / 'release_manifest.json').exists())
            self.assertEqual(source.read_bytes(), (bundle / 'fusion/XGBoostMetaClassifier.kt').read_bytes())

    def test_source_install_failure_restores_previous_assets_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            bundle, destination = parent / 'bundle', parent / 'assets'
            manifest = fixture(bundle)
            manifest['validated'] = True
            destination.mkdir()
            (destination / 'old.txt').write_text('original assets')
            source = parent / 'XGBoostMetaClassifier.kt'
            source.write_text('original source')
            arguments = ['release_bundle', '--bundle', str(bundle), '--install', str(destination), '--kotlin-destination', str(source)]
            actual_replace = release_bundle.os.replace
            def fail_source(origin, target):
                if Path(target) == source:
                    raise OSError('simulated source replacement failure')
                return actual_replace(origin, target)
            with patch.object(release_bundle, 'validate_bundle', return_value=manifest), patch.object(sys, 'argv', arguments), patch.object(release_bundle.os, 'replace', side_effect=fail_source):
                with self.assertRaises(OSError):
                    release_bundle.main()
            self.assertEqual((destination / 'old.txt').read_text(), 'original assets')
            self.assertEqual(source.read_text(), 'original source')
            self.assertTrue((parent / 'assets.failed-test-release/constants/release_manifest.json').is_file())

    def test_failed_first_install_retains_candidate_and_leaves_no_active_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            bundle, destination = parent / 'bundle', parent / 'assets'
            manifest = fixture(bundle)
            manifest['validated'] = True
            source = parent / 'XGBoostMetaClassifier.kt'
            source.write_text('original source')
            arguments = ['release_bundle', '--bundle', str(bundle), '--install', str(destination), '--kotlin-destination', str(source)]
            with patch.object(release_bundle, 'validate_bundle', return_value=manifest), patch.object(sys, 'argv', arguments), patch.object(release_bundle.os, 'replace', side_effect=OSError('source replace fails')):
                with self.assertRaises(OSError):
                    release_bundle.main()
            self.assertFalse(destination.exists())
            self.assertEqual(source.read_text(), 'original source')
            self.assertTrue((parent / 'assets.failed-test-release/constants/release_manifest.json').is_file())

    def test_existing_failed_backup_blocks_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            bundle, destination = parent / 'bundle', parent / 'assets'
            manifest = fixture(bundle)
            manifest['validated'] = True
            destination.mkdir()
            (destination / 'old.txt').write_text('old')
            (parent / 'assets.failed-test-release').mkdir()
            source = parent / 'XGBoostMetaClassifier.kt'
            source.write_text('old code')
            arguments = ['release_bundle', '--bundle', str(bundle), '--install', str(destination), '--kotlin-destination', str(source)]
            with patch.object(release_bundle, 'validate_bundle', return_value=manifest), patch.object(sys, 'argv', arguments):
                with self.assertRaises(FileExistsError):
                    release_bundle.main()
            self.assertEqual((destination / 'old.txt').read_text(), 'old')
            self.assertEqual(source.read_text(), 'old code')

class CascadeModelContractTests(unittest.TestCase):
    """Engine C ("Lite C") is optional, but never half-verified.

    A release may ship the second engine or not. When it does, it must clear
    the same gates as the required models; when it does not, nothing about
    the required models may change.
    """

    def _add_cascade(self, root, manifest, frontend=None, shape=(1, 224, 224, 3)):
        name = next(iter(release_bundle.CASCADE_MODEL_NAMES))
        path = root / 'models' / name
        path.write_bytes(b'\0\0\0\0TFL3' + bytes(range(64)))
        frontend_id = frontend or f'clip-appearance-bicubic{shape[1]}-clipnorm-meanstd-float32'
        manifest['models'][name] = {'sha256': sha256_file(path), 'input_shape': list(shape),
                                   'output_shape': [1, 1], 'dtype': 'FLOAT32', 'frontend_id': frontend_id}
        manifest['calibration'][name] = {'model_hash': sha256_file(path), 'frontend_id': frontend_id,
                                         'threshold': .5, 'auc': .85, 'n_calibration': 400,
                                         'split_manifest_hash': 'a' * 64}
        report_path = root / 'evaluation_report.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        report['models'][name] = {'parity': {'passed': True},
                                  'calibration': {'model_hash': sha256_file(path), 'frontend_id': frontend_id}}
        report_path.write_text(json.dumps(report), encoding='utf-8')
        manifest['evaluation_report_hash'] = sha256_file(report_path)
        manifest['artifacts']['evaluation_report.json'] = sha256_file(report_path)
        (root / 'constants/semantic_probe_threshold.json').write_text(
            json.dumps({'threshold_at_eer': .5, 'auc': .85, 'n_calibration': 400,
                        'split_manifest_hash': 'a' * 64, 'model_hash': sha256_file(path),
                        'frontend_id': frontend_id}), encoding='utf-8')
        (root / 'constants/cascade_policy.json').write_text(
            json.dumps({'engine_a_threshold': .5, 'engine_c_threshold': .5,
                        'escalate_below': .4, 'escalate_above': .6}), encoding='utf-8')
        manifest['artifacts'] = {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in root.rglob('*') if path.is_file() and path.name != 'release_manifest.json'
        }
        (root / 'release_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return manifest

    def test_a_release_without_the_cascade_model_still_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'bundle'
            fixture(root)
            manifest = release_bundle.validate_bundle(root)
            self.assertNotIn(next(iter(release_bundle.CASCADE_MODEL_NAMES)), manifest['models'])

    def test_a_complete_cascade_model_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'bundle'
            manifest = fixture(root)
            self._add_cascade(root, manifest)
            validated = release_bundle.validate_bundle(root)
            self.assertIn(next(iter(release_bundle.CASCADE_MODEL_NAMES)), validated['models'])

    def test_a_cascade_model_with_the_wrong_frontend_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'bundle'
            manifest = fixture(root)
            self._add_cascade(root, manifest, frontend='rgb-exif-srgb-whitealpha-pow2cap1024-bilinear-square-v2-size224-float32')
            with self.assertRaises(ValueError):
                release_bundle.validate_bundle(root)

    def test_a_cascade_model_without_its_artifacts_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'bundle'
            manifest = fixture(root)
            self._add_cascade(root, manifest)
            (root / 'constants/cascade_policy.json').unlink()
            manifest['artifacts'].pop('constants/cascade_policy.json', None)
            (root / 'release_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                release_bundle.validate_bundle(root)

    def test_a_cascade_model_at_or_below_chance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'bundle'
            manifest = fixture(root)
            self._add_cascade(root, manifest)
            name = next(iter(release_bundle.CASCADE_MODEL_NAMES))
            manifest['calibration'][name]['auc'] = 0.5
            (root / 'release_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                release_bundle.validate_bundle(root)

    def test_an_unexpected_model_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'bundle'
            manifest = fixture(root)
            manifest['models']['not_a_real_model.tflite'] = {
                'sha256': 'a' * 64, 'input_shape': [1, 32, 32, 3], 'output_shape': [1, 1],
                'dtype': 'FLOAT32', 'frontend_id': 'x'}
            (root / 'release_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                release_bundle.validate_bundle(root)


if __name__ == '__main__':
    unittest.main()

