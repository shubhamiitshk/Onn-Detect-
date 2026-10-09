"""Content lineage, split leakage and atomic manifest preparation regressions."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from data.contracts import SPLITS, sha256_file, validate_dataset_manifest
import prepare_dataset_manifest


def prepared_dataset(root):
    records = []
    for index, split in enumerate(SPLITS):
        for label in (0, 1):
            relative = f'{split}/{"real" if label == 0 else "fake"}/sample.png'
            path = root / relative
            path.parent.mkdir(parents=True)
            Image.new('RGB', (8, 8), (index * 40, label * 100, 37)).save(path)
            records.append({'path': relative, 'split': split, 'label': label,
                            'sha256': sha256_file(path), 'source_id': relative, 'group_id': relative})
    manifest = {'schema_version': 1, 'label_convention': 'real=0_fake=1',
                'source': 'regression-fixture', 'revision': 'fixture-v1',
                'grouping_verified': True, 'records': records}
    (root / 'dataset_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return manifest


class DatasetManifestTests(unittest.TestCase):
    def test_content_change_and_group_leakage_are_rejected(self):
        for changed_bytes in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = prepared_dataset(root)
                validate_dataset_manifest(root)
                if changed_bytes:
                    (root / manifest['records'][0]['path']).write_bytes(b'changed source bytes')
                else:
                    manifest['records'][2]['group_id'] = manifest['records'][0]['group_id']
                    (root / 'dataset_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaises(ValueError):
                    validate_dataset_manifest(root)

    def test_identical_media_across_different_group_ids_still_leaks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = prepared_dataset(root)
            first, second = manifest['records'][0], manifest['records'][2]
            shutil.copy2(root / first['path'], root / second['path'])
            second['sha256'] = first['sha256']
            (root / 'dataset_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                validate_dataset_manifest(root)

    def test_schema_boolean_and_missing_fusion_splits_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = prepared_dataset(root)
            with self.assertRaises(ValueError):
                validate_dataset_manifest(root, (*SPLITS, 'stacking', 'fusion_val'))
            manifest['schema_version'] = True
            (root / 'dataset_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                validate_dataset_manifest(root)

    def test_preparation_validation_failure_preserves_prior_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = prepared_dataset(root)
            original = (root / 'dataset_manifest.json').read_bytes()
            group_map = root / 'groups.json'
            group_map.write_text(json.dumps({record['path']: {'source_id': record['source_id'], 'group_id': record['group_id']}
                                            for record in manifest['records']}), encoding='utf-8')
            args = ['prepare_dataset_manifest', '--root', str(root), '--source', 'fixture',
                    '--revision', 'v2', '--group-map', str(group_map)]
            with patch.object(sys, 'argv', args), patch.object(prepare_dataset_manifest, 'validate_dataset_manifest', side_effect=ValueError('simulated final validation failure')):
                with self.assertRaises(ValueError):
                    prepare_dataset_manifest.main()
            self.assertEqual((root / 'dataset_manifest.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
