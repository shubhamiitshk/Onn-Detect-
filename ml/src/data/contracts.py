"""Strict dataset discovery and content-addressed split contracts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.opus'}
SPLITS = ('train', 'val', 'calibration', 'test')
CLASSES = {'real': 0, 'authentic': 0, '0_real': 0,
           'fake': 1, 'synthetic': 1, '1_fake': 1}

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def semantic_label(names, source_label):
    """Translate semantic class names, refusing ambiguous numeric labels."""
    if names is None:
        raise ValueError('Source has no semantic class names; supply a verified label map')
    name = str(names[int(source_label)]).lower().strip()
    if name in ('real', 'authentic', 'bonafide', 'bona-fide', 'human', 'natural'):
        return 0
    if name in ('fake', 'synthetic', 'spoof', 'ai', 'ai-generated'):
        return 1
    raise ValueError(f'Unknown source class {name!r}')

def discover_samples(root, split, extensions):
    if split not in (*SPLITS, 'stacking', 'fusion_val'):
        raise ValueError(f'Unsupported explicit split {split!r}')
    split_root = Path(root) / split
    if not split_root.is_dir():
        raise FileNotFoundError(f'Missing split directory: {split_root}')
    samples = []
    for path in sorted(split_root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        parts = path.relative_to(split_root).parts
        label = CLASSES.get(parts[0].lower()) if len(parts) > 1 else None
        if label is None:
            raise ValueError(f'No unambiguous class directory for {path}')
        samples.append((path, label))
    if not samples or {label for _, label in samples} != {0, 1}:
        raise ValueError(f'{split_root} must contain both real and fake samples')
    return samples

def validate_dataset_manifest(root, required_splits=SPLITS, *, manifest_path=None):
    """Verify identity, bytes and original-photo/recording group disjointness.

    Unknown ancestry is not a verified independent split. Each record needs
    path, sha256, label, split, source_id and an authoritative group_id.
    """
    root = Path(root).resolve()
    manifest_path = Path(manifest_path) if manifest_path is not None else root / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int or manifest.get('schema_version') != 1 or manifest.get('label_convention') != 'real=0_fake=1':
        raise ValueError('Unsupported dataset manifest or label convention')
    if not manifest.get('source') or not manifest.get('revision'):
        raise ValueError('Dataset source and immutable revision are required')
    if manifest.get('grouping_verified') is not True:
        raise ValueError('Verify original-photo/recording group IDs before training')
    if not isinstance(manifest.get('records'), list):
        raise ValueError('Dataset records must be a JSON list')
    membership, byte_membership, paths = {}, {}, set()
    counts = {split: [0, 0] for split in required_splits}
    for record in manifest.get('records', []):
        if not isinstance(record, dict):
            raise ValueError('Dataset records must be JSON objects')
        split, label = record['split'], record['label']
        parts = Path(record['path']).parts
        if len(parts) < 3 or parts[0] != split or CLASSES.get(parts[1].lower()) != label or type(label) is not int:
            raise ValueError('Record path/split/semantic label disagree')
        if label not in (0, 1) or not record.get('source_id') or not record.get('group_id'):
            raise ValueError('Every record needs a binary label, source ID and group ID')
        path = (root / record['path']).resolve()
        if not path.is_relative_to(root) or path in paths:
            raise ValueError(f'Invalid/duplicate record path {path}')
        paths.add(path)
        if sha256_file(path) != record['sha256']:
            raise ValueError(f'Dataset content changed: {path}')
        for identity, mapping in ((record['group_id'], membership), (record['sha256'], byte_membership)):
            previous = mapping.setdefault(identity, split)
            if previous != split:
                raise ValueError(f'Cross-split leakage for {identity}: {previous} / {split}')
        if split in counts:
            counts[split][label] += 1
    for split, count in counts.items():
        if not all(count):
            raise ValueError(f'Explicit {split} split needs both classes; got {count}')
        listed = {str((root / r['path']).resolve()) for r in manifest['records'] if r['split'] == split}
        actual = {str(p.resolve()) for p, _ in discover_samples(root, split, IMAGE_EXTS | AUDIO_EXTS)}
        if listed != actual:
            raise ValueError(f'Manifest does not cover exactly the {split} files')
    return {'manifest_sha256': sha256_file(manifest_path), 'source': manifest['source'],
            'revision': manifest['revision'], 'split_counts': counts}

def balanced_subset(samples, limit=None, seed=42):
    import random
    groups = [[item for item in samples if item[1] == label] for label in (0, 1)]
    rng = random.Random(seed)
    for group in groups:
        rng.shuffle(group)
    per_class = min(map(len, groups)) if limit is None else min(min(map(len, groups)), limit // 2)
    if per_class < 1:
        raise ValueError('Balanced sample pool requires both classes')
    return [item for pair in zip(groups[0][:per_class], groups[1][:per_class]) for item in pair]
