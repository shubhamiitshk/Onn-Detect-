"""Validate prepared split directories with an authoritative source/group map.

group-map JSON maps each root-relative media path to {source_id, group_id}.
The caller must obtain IDs from original photo/recording metadata, linking all
derived variants; per-file counters do not establish independent groups.
"""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data.contracts import SPLITS, IMAGE_EXTS, AUDIO_EXTS, discover_samples, sha256_file, validate_dataset_manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--group-map', required=True)
    parser.add_argument('--include-fusion', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    group_map = json.loads(Path(args.group_map).read_text(encoding='utf-8'))
    splits = (*SPLITS, 'stacking', 'fusion_val') if args.include_fusion else SPLITS
    records, group_splits, byte_splits = [], {}, {}
    for split in splits:
        for path, label in discover_samples(root, split, IMAGE_EXTS | AUDIO_EXTS):
            relative = path.relative_to(root).as_posix()
            ids = group_map[relative]
            if not ids.get('source_id') or not ids.get('group_id'):
                raise ValueError(f'Missing authoritative source/group IDs for {relative}')
            digest = sha256_file(path)
            for key, mapping in ((ids['group_id'], group_splits), (digest, byte_splits)):
                if mapping.setdefault(key, split) != split:
                    raise ValueError(f'Cross-split leakage for {relative}')
            # Full decode preflight; no corrupt-file substitution. Audio is
            # decoded in bounded blocks even for multi-hour source recordings.
            if path.suffix.lower() in IMAGE_EXTS:
                from PIL import Image
                with Image.open(path) as image:
                    image.verify()
            else:
                import soundfile as sf
                import numpy as np
                info = sf.info(path)
                if info.frames <= 0 or not 1 <= info.samplerate <= 384000:
                    raise ValueError(f'Unusable audio {relative}')
                decoded_frames = 0
                for waveform in sf.blocks(path, blocksize=65536, dtype='float32', always_2d=True):
                    if waveform.size == 0 or not np.isfinite(waveform).all():
                        raise ValueError(f'Unusable audio {relative}')
                    decoded_frames += waveform.shape[0]
                if decoded_frames != info.frames:
                    raise ValueError(f'Truncated audio {relative}')
            records.append({'path': relative, 'label': label, 'split': split,
                            'source_id': ids['source_id'], 'group_id': ids['group_id'], 'sha256': digest,
                            **{key: ids[key] for key in ('source', 'generator', 'speaker', 'language', 'codec', 'platform') if key in ids}})
    manifest = {'schema_version': 1, 'source': args.source, 'revision': args.revision,
                'label_convention': 'real=0_fake=1', 'grouping_verified': True,
                'group_map_sha256': sha256_file(args.group_map), 'records': records}
    temporary = root / 'dataset_manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    verified = validate_dataset_manifest(root, splits, manifest_path=temporary)
    os.replace(temporary, root / 'dataset_manifest.json')
    print(json.dumps(verified, indent=2))

if __name__ == '__main__':
    main()
