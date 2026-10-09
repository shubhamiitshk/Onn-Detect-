"""Authoritative config entrypoint shared by image and speech training."""
import argparse
import json
import random
from pathlib import Path
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from data.contracts import validate_dataset_manifest, sha256_file
from data.image_dataset import StrictImageDataset, image_tensor
from data.image_frontend import frontend_id
from data.audio_dataset import StrictAudioDataset
from data.lfcc_frontend import FRONTEND_ID
from data.augmentations import PhonePipelineAugmentation
from models.srm_forensic_cnn import SRMForensicCNN
from models.lcnn_mfm_audio import LCNNMFMAudio
from training.trainer import Trainer
from training.checkpoint import load_verified_checkpoint
from training.provenance import training_code_hash
from training.metrics import compute_auc, compute_eer, compute_accuracy
from utils.device import get_device
from utils.seed import set_global_seed

ML_ROOT = Path(__file__).resolve().parents[2]

class ImageTrainingTransform:
    def __init__(self, size):
        self.size = size
        self.phone = PhonePipelineAugmentation(p_jpeg=0.7, p_downscale=0.5, p_blur=0.15)
    def __call__(self, image):
        image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        if random.random() < .5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < .5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return image_tensor(self.phone(image), self.size)

def run(modality):
    parser = argparse.ArgumentParser(description=f'Train {modality} with four explicit data splits')
    parser.add_argument('--config', default=str(ML_ROOT / 'configs' / ('srm_cnn_genimage.yaml' if modality == 'visual' else 'lcnn_asvspoof.yaml')))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir', default=str(ML_ROOT / 'checkpoints'))
    for flag, kind in [('epochs', int), ('batch-size', int), ('lr', float), ('num-workers', int)]:
        parser.add_argument('--' + flag, type=kind)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--resume')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    for flag in ('epochs', 'batch_size', 'num_workers', 'amp'):
        if getattr(args, flag) is not None:
            config['training'][flag] = getattr(args, flag)
    if args.lr is not None:
        config['optimizer']['lr'] = args.lr
    data = config['data']
    root = Path(data['root_dir'])
    root = root if root.is_absolute() else (ML_ROOT / root).resolve()
    provenance = validate_dataset_manifest(root)
    config['data']['root_dir'] = str(root)
    config['checkpoint_dir'] = str(Path(args.output_dir).resolve())
    name = 'srm_cnn_genimage' if modality == 'visual' else 'lcnn_audio_asvspoof'
    config['model_name'] = name
    if not args.resume and not args.eval_only and any((Path(config['checkpoint_dir']) / f'{name}_{kind}.pth').exists() for kind in ('best', 'last')):
        raise FileExistsError('Existing model checkpoints are preserved; choose a new --output-dir or explicitly --resume')
    device = get_device() if args.device == 'auto' else torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; choose --device cpu')
    set_global_seed(config.get('seed', 42))
    if modality == 'visual':
        size = int(data['img_size'])
        if config['model']['input_size'] != size or size < 32:
            raise ValueError('Model/data input size mismatch or unsupported size')
        model = SRMForensicCNN()
        datasets = {split: StrictImageDataset(root, split, size, ImageTrainingTransform(size) if split == 'train' else None,
                                             cache_in_memory=data.get('cache_in_memory', False), cache_bytes=data.get('cache_bytes', 32 * 1024 * 1024))
                    for split in ('train', 'val', 'test')}
        frontend = frontend_id(size)
        shape = [1, size, size, 3]
    else:
        model = LCNNMFMAudio()
        datasets = {split: StrictAudioDataset(root, split, data['max_duration'], data['sample_rate']) for split in ('train', 'val', 'test')}
        frontend, shape = FRONTEND_ID, [1, 401, 40, 1]
    config['contract'] = {'frontend_id': frontend, 'input_shape': shape, 'architecture': type(model).__name__,
                          'label_convention': 'real=0_fake=1', 'dataset_manifest_hash': provenance['manifest_sha256'],
                          'source': provenance['source'], 'revision': provenance['revision'],
                          'training_code_hash': training_code_hash(),
                          'audio_crop_policy': 'source_window_then_resample_center_crop_v1' if modality == 'audio' else None}
    tc = config['training']
    workers = int(tc['num_workers'])
    if workers < 0 or int(tc['batch_size']) <= 0:
        raise ValueError('Invalid worker/batch configuration')
    kwargs = {'batch_size': int(tc['batch_size']), 'num_workers': workers, 'pin_memory': device.type == 'cuda'}
    if workers:
        kwargs['prefetch_factor'] = int(tc.get('prefetch_factor', 2))
        # Restart workers each epoch so checkpointed RNG determines worker seeds.
        kwargs['persistent_workers'] = False
    loaders = {split: DataLoader(ds, shuffle=split == 'train', **kwargs) for split, ds in datasets.items()}
    trainer = Trainer(model, loaders['train'], loaders['val'], config, device=device)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    if not args.eval_only:
        trainer.train()
    path = Path(config['checkpoint_dir']) / (name + '_best.pth')
    state = load_verified_checkpoint(path, config['contract'])
    model.load_state_dict(state['state_dict'])
    model.to(device).eval()
    labels, scores = [], []
    with torch.no_grad():
        for inputs, targets in loaders['test']:
            scores.extend(model(inputs.to(device, dtype=torch.float32)).cpu().flatten().tolist())
            labels.extend(targets.tolist())
    report = {'test_auc': float(compute_auc(labels, scores)), 'test_eer': float(compute_eer(labels, scores)),
              'test_accuracy_at_0_5': float(compute_accuracy(labels, scores)), 'eval_split': 'test',
              'eval_samples': len(labels), 'contract': config['contract'], 'checkpoint_sha256': sha256_file(path),
              'status': 'pytorch_only_not_mobile_release', 'input_size': shape}
    output = Path(config['checkpoint_dir']) / (name + '_results.json')
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
