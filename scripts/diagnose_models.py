"""Compare a verified checkpoint and TFLite export on the same labeled inputs.

Requires an explicit dataset, threshold and model paths. This checks export
parity, not Android bitmap/codec parity or accuracy on unseen generators.
Legacy checkpoints with unknown label/frontend/data contracts are rejected.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ml' / 'src'))
sys.path.insert(0, str(ROOT / 'ml' / 'scripts'))


def main(default_modality=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modality', choices=('visual', 'audio'), required=default_modality is None, default=default_modality)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tflite', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split', choices=('calibration', 'test'), default='test')
    parser.add_argument('--threshold', type=float, required=True, help='Frozen operational threshold from calibration, never fit to test')
    parser.add_argument('--limit', type=int, default=200)
    args = parser.parse_args()
    if args.limit < 100 or not 0 <= args.threshold <= 1:
        parser.error('At least 100 balanced samples and a finite threshold in [0,1] are required')
    from data.contracts import validate_dataset_manifest
    from data.lfcc_frontend import FRONTEND_ID
    from models.srm_forensic_cnn import SRMForensicCNN
    from models.lcnn_mfm_audio import LCNNMFMAudio
    from export_all_models import collect, load_model
    from validate_export import validate_models_real

    lineage = validate_dataset_manifest(args.data_root)
    cls = SRMForensicCNN if args.modality == 'visual' else LCNNMFMAudio
    model, state = load_model(args.checkpoint, cls, FRONTEND_ID if args.modality == 'audio' else None)
    if state['contract']['dataset_manifest_hash'] != lineage['manifest_sha256']:
        raise ValueError('Dataset manifest differs from the checkpoint; this is not a matched export check')
    size = int(state['contract']['input_shape'][1])
    samples, values = collect(args.data_root, args.split, args.modality, size, args.limit)
    if len(samples) < 100:
        raise ValueError('Insufficient labeled samples for the export parity gate')
    metrics = validate_models_real(model, args.tflite, args.modality,
        real_images=values if args.modality == 'visual' else None,
        real_specs_pt=values if args.modality == 'audio' else None,
        num_samples=args.limit, labels=[label for _, label in samples], threshold=args.threshold)
    print(json.dumps(dict(metrics, dataset_manifest_sha256=lineage['manifest_sha256'], split=args.split,
        evidence='desktop export parity; device and unseen-generator validation are separate'), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
