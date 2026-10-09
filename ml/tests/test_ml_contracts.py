"""CPU regressions for previously reproduced validity failures."""
import json
import os
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from data.contracts import semantic_label, discover_samples, IMAGE_EXTS, balanced_subset, sha256_file, validate_dataset_manifest
from data.image_dataset import StrictImageDataset
from data.audio_dataset import StrictAudioDataset
from data.lfcc_extractor import LFCCExtractor
from data.lfcc_frontend import FRONTEND_ID
from data.lfcc_reference import resample_sinc
from models.lcnn_mfm_audio import LCNNMFMAudio
from models.srm_forensic_cnn import SRMForensicCNN
from training.trainer import Trainer
from training.checkpoint import atomic_torch_save, load_verified_checkpoint
from training.metrics import compute_auc, compute_eer
from validate_export import compare_scores

torch.set_num_threads(1)

class ContractTests(unittest.TestCase):
    def test_source_semantics_cifake_and_audio(self):
        self.assertEqual([semantic_label(['FAKE', 'REAL'], i) for i in (0, 1)], [1, 0])
        self.assertEqual([semantic_label(['real', 'fake'], i) for i in (0, 1)], [0, 1])
        with self.assertRaises(ValueError):
            semantic_label(['0', '1'], 0)

    def test_missing_split_never_falls_back_and_nested_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ('real', 'fake'):
                path = root / 'train' / label / 'speaker'
                path.mkdir(parents=True)
                Image.new('RGB', (64, 80), label == 'fake' and 'white' or 'black').save(path / 'sample.png')
            self.assertEqual({label for _, label in discover_samples(root, 'train', IMAGE_EXTS)}, {0, 1})
            with self.assertRaises(FileNotFoundError):
                StrictImageDataset(root, 'test')
            samples = balanced_subset(discover_samples(root, 'train', IMAGE_EXTS), 2)
            self.assertEqual([label for _, label in samples], [0, 1])

    def test_lazy_bounded_cache_and_corrupt_samples_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ('real', 'fake'):
                path = root / 'train' / label
                path.mkdir(parents=True)
                for i in range(2):
                    Image.new('RGB', (64, 64), (i * 40, 20, 30)).save(path / f'{i}.png')
            dataset = StrictImageDataset(root, 'train', 64, cache_in_memory=True, cache_bytes=64 * 64 * 3)
            self.assertFalse(dataset.image_cache)
            for index in range(len(dataset)):
                tensor, _ = dataset[index]
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertEqual(tuple(tensor.shape), (3, 64, 64))
                self.assertLessEqual(dataset.cache_size, dataset.cache_limit)
            path = root / 'train/fake/corrupt.png'
            path.write_bytes(b'not an image')
            dataset = StrictImageDataset(root, 'train')
            index = next(i for i, (p, _) in enumerate(dataset.samples) if p == path)
            with self.assertRaises(RuntimeError):
                dataset[index]

    def test_float64_lfcc_to_real_lcnn_is_float32(self):
        features = LFCCExtractor().extract(torch.randn(64000, dtype=torch.float64))
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(list(features.shape), [1, 1, 401, 40])
        model = LCNNMFMAudio().eval()
        with torch.no_grad():
            result = model(features)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(list(result.shape), [1, 1])

    def test_audio_loader_to_model_uses_shared_resampling(self):
        # SoundFile is optional in this checkout; a decoding stub isolates the
        # actual dataset->resampler->LFCC->network integration, not accuracy.
        fake_sf = types.SimpleNamespace(info=lambda path: types.SimpleNamespace(samplerate=48000, frames=4800), read=lambda *args, **kwargs: (np.sin(np.arange(4800)[:, None] / 11).astype(np.float32), 48000))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'soundfile': fake_sf}):
            root = Path(directory)
            for label in ('real', 'fake'):
                path = root / 'val' / label / 'recording'
                path.mkdir(parents=True)
                (path / 'sample.wav').write_bytes(b'decoder stub')
            dataset = StrictAudioDataset(root, 'val')
            batch, labels = next(iter(DataLoader(dataset, batch_size=2)))
            self.assertEqual(batch.dtype, torch.float32)
            with torch.no_grad():
                output = LCNNMFMAudio().eval()(batch)
            self.assertEqual(list(output.shape), [2, 1])
            with patch.object(fake_sf, 'read', side_effect=ValueError('broken file')):
                with self.assertRaises(RuntimeError):
                    dataset[0]

    def test_real_soundfile_decode_through_lfcc_lcnn_and_backward(self):
        try:
            import soundfile as sf
        except ImportError:
            if os.environ.get('ONN_REQUIRE_REAL_AUDIO_IO') == '1':
                raise
            self.skipTest('Install pinned SoundFile for real audio I/O; CI requires ONN_REQUIRE_REAL_AUDIO_IO=1')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Synthetic signal fixtures verify PCM decoding, resampling,
            # tensor precision and training plumbing, not detector accuracy.
            time = np.arange(12000) / 48000
            for label, frequency in (('real', 440), ('fake', 1100)):
                path = root / 'train' / label
                path.mkdir(parents=True)
                waveform = (.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
                sf.write(path / 'fixture.wav', waveform, 48000, subtype='PCM_16')
            dataset = StrictAudioDataset(root, 'train')
            inputs, labels = next(iter(DataLoader(dataset, batch_size=2)))
            self.assertEqual(inputs.dtype, torch.float32)
            self.assertEqual(list(inputs.shape), [2, 1, 401, 40])
            model = LCNNMFMAudio()
            logits = model.forward_logits(inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float().reshape(-1, 1))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            self.assertTrue(gradients and all(value is not None and torch.isfinite(value).all() for value in gradients))

    def test_custom_resampler_identity_and_rate_validation(self):
        wave = np.ones(1000)
        result = resample_sinc(wave, 48000, 16000)
        self.assertEqual(len(result), 334)
        np.testing.assert_allclose(result, 1, atol=1e-10)
        with self.assertRaises(ValueError):
            resample_sinc(wave, 0, 16000)

    def test_vectorized_resampler_matches_independent_scalar_oracle(self):
        # Scalar historical formula provides an independent regression oracle
        # for fractional rates and both short/long reflective-edge policies.
        def oracle(signal, rate):
            ratio = 16000 / rate
            half = int(np.ceil(6 * max(1, 1 / ratio)))
            padded = np.pad(signal, (half, half), mode='reflect' if len(signal) > half else 'constant')
            output = []
            for index in range(int(np.ceil(len(signal) * ratio))):
                position = index / ratio
                base = int(np.floor(position))
                total = weight_sum = 0.0
                for k in range(-half, half + 1):
                    distance = np.pi * .99 * min(1, ratio) * (base + k - position)
                    sinc = 1.0 if abs(distance) < 1e-12 else np.sin(distance) / distance
                    weight = .99 * min(1, ratio) * sinc * max(0, 1 - (k / (half + 1.0)) ** 2)
                    total += weight * padded[base + k + half]
                    weight_sum += weight
                output.append(total / weight_sum if weight_sum else 0)
            return np.asarray(output)
        for rate in (44100, 48000, 8000):
            for count in (7, 127):
                signal = np.sin(np.arange(count) * .73) + np.cos(np.arange(count) * .17)
                np.testing.assert_allclose(resample_sinc(signal, rate, 16000), oracle(signal, rate), atol=2e-14, rtol=2e-14)

    def test_bounded_audio_decode_never_loads_whole_long_recording(self):
        from data.audio_dataset import load_analysis_waveform
        calls = []
        def read(path, **kwargs):
            calls.append(kwargs)
            self.assertLessEqual(kwargs['stop'] - kwargs['start'], 192000)
            return np.ones((kwargs['stop'] - kwargs['start'], 1), dtype=np.float32), 48000
        backend = types.SimpleNamespace(info=lambda path: types.SimpleNamespace(samplerate=48000, frames=48000 * 3600), read=read)
        with patch.dict(sys.modules, {'soundfile': backend}):
            waveform = load_analysis_waveform('one_hour.wav', 'test')
        self.assertEqual(len(waveform), 64000)
        self.assertGreater(calls[0]['start'], 0)

    def test_export_audio_features_require_an_explicit_split(self):
        """make_lfcc_fn must never fall back to a default split.

        load_analysis_waveform randomizes the 4 s window only for
        split == 'train' and centres it otherwise, so an omitted argument
        silently feeds centred crops to the INT8 representative set while the
        checkpoint was trained on random crops.
        """
        import export_all_models
        with self.assertRaises(TypeError):
            export_all_models.make_lfcc_fn()
        for unsupported in ('stacking', 'fusion_val', '', 'TRAIN', None):
            with self.assertRaises(ValueError):
                export_all_models.make_lfcc_fn(unsupported)

    def test_export_train_and_calibration_use_different_window_positions(self):
        """The train representative set must use random crops, not centred ones."""
        import export_all_models
        starts = []

        def read(path, **kwargs):
            starts.append(kwargs['start'])
            count = kwargs['stop'] - kwargs['start']
            return np.ones((count, 1), dtype=np.float32), 48000

        backend = types.SimpleNamespace(
            info=lambda path: types.SimpleNamespace(samplerate=48000, frames=48000 * 3600), read=read)
        with patch.dict(sys.modules, {'soundfile': backend}):
            for split in ('calibration', 'calibration', 'test', 'test'):
                export_all_models.make_lfcc_fn(split)('long.wav')
        self.assertEqual(starts[0], starts[1], 'calibration windows must be deterministic')
        self.assertEqual(starts[2], starts[3], 'test windows must be deterministic')

        # Random crop happens inside load_analysis_waveform; sample it directly
        # rather than reimplementing the RNG here.
        from data.audio_dataset import load_analysis_waveform
        train_starts, eval_starts = set(), set()
        with patch.dict(sys.modules, {'soundfile': backend}):
            for _ in range(24):
                starts.clear()
                load_analysis_waveform('long.wav', 'train')
                train_starts.add(starts[0])
                starts.clear()
                load_analysis_waveform('long.wav', 'calibration')
                eval_starts.add(starts[0])
        self.assertEqual(len(eval_starts), 1, 'non-train splits must be centred')
        self.assertGreater(len(train_starts), 1, 'train windows must actually vary')

    def test_sealed_test_default_evaluates_every_file_not_parity_pool(self):
        import export_all_models
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ('real', 'fake'):
                path = root / 'test' / label
                path.mkdir(parents=True)
                for index in range(3):
                    Image.new('RGB', (32, 32), (index * 40, 0, 0)).save(path / f'{index}.png')
            with patch.object(export_all_models, '_run_tflite', return_value=.2) as run:
                samples, scores = export_all_models.sealed_test_scores(root, 'visual', 32, 'model', limit=None)
            self.assertEqual(len(samples), 6)
            self.assertEqual(run.call_count, 6)
            self.assertEqual(len(scores), 6)

    def test_one_class_metrics_fail_below_chance_remains_visible(self):
        with self.assertRaises(ValueError):
            compute_auc([0, 0], [.1, .2])
        with self.assertRaises(ValueError):
            compute_eer([0, 0], [.1, .2])
        self.assertEqual(compute_auc([0, 1], [.9, .1]), 0)

    def test_export_gates_detect_confidence_damage_above_same_decision(self):
        # Old .5 agreement gate passed this 100%; calibrated threshold differs.
        with self.assertRaises(ValueError):
            compare_scores([0, 0, 1, 1], [.1, .2, .8, .9], [.4, .45, .55, .6], .225)
        metrics = compare_scores([0, 0, 1, 1], [.1, .2, .8, .9], [.101, .201, .801, .901], .225)
        self.assertTrue(metrics['passed'])

    def test_tflite_interpreter_allocated_once_for_sequential_scores(self):
        from validate_export import _interpreter_for, _run_tflite
        allocations = []
        class Interpreter:
            def __init__(self, **kwargs):
                allocations.append(kwargs)
            def allocate_tensors(self):
                pass
            def get_input_details(self):
                return [{'shape': np.array([1, 2, 2, 3]), 'dtype': np.float32, 'index': 0}]
            def get_output_details(self):
                return [{'shape': np.array([1, 1]), 'dtype': np.float32, 'index': 1}]
            def set_tensor(self, index, value):
                self.value = value
            def invoke(self):
                pass
            def get_tensor(self, index):
                return np.array([[self.value.mean()]], dtype=np.float32)
        fake_tf = types.SimpleNamespace(lite=types.SimpleNamespace(Interpreter=Interpreter))
        _interpreter_for.cache_clear()
        try:
            with patch.dict(sys.modules, {'tensorflow': fake_tf}):
                for score in (.1, .2, .8):
                    self.assertAlmostEqual(_run_tflite('immutable-model.tflite', np.full((1, 2, 2, 3), score, np.float32)), score)
                self.assertEqual(len(allocations), 1)
                with self.assertRaises(ValueError):
                    _run_tflite('immutable-model.tflite', np.full((1, 2, 2, 3), np.nan, np.float32))
        finally:
            _interpreter_for.cache_clear()

    def test_srm_logits_preserve_probability_api(self):
        model = SRMForensicCNN().eval()
        for size in (32, 64):
            x = torch.rand(2, 3, size, size)
            with torch.no_grad():
                torch.testing.assert_close(torch.sigmoid(model.forward_logits(x)), model(x))

    def test_checkpoint_requires_lineage_and_restores_rng_optimizer(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)
            def forward_logits(self, x):
                return self.linear(x)
        x = torch.randn(6, 2, dtype=torch.float64)
        y = torch.tensor([0, 1, 0, 1, 0, 1])
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            contract = {'frontend_id': 'test', 'dataset_manifest_hash': 'a' * 64,
                        'input_shape': [1, 2], 'label_convention': 'real=0_fake=1'}
            config = {'training': {'epochs': 2, 'amp': True, 'gradient_accumulation_steps': 2},
                      'contract': contract, 'checkpoint_dir': directory}
            trainer = Trainer(Tiny(), loader, loader, config, device=torch.device('cpu'))
            self.assertFalse(trainer.amp_enabled)
            self.assertTrue(np.isfinite(trainer.train_epoch()))
            state = trainer._state(0)
            path = Path(directory) / 'last.pth'
            atomic_torch_save(state, path)
            expected = (torch.rand(4), np.random.rand(4), random.random())
            resumed = Trainer(Tiny(), loader, loader, config, device=torch.device('cpu'))
            resumed.load_checkpoint(path)
            torch.testing.assert_close(torch.rand(4), expected[0])
            np.testing.assert_allclose(np.random.rand(4), expected[1])
            self.assertEqual(random.random(), expected[2])
            self.assertEqual(resumed.start_epoch, 1)
            atomic_torch_save({'state_dict': {}}, path)
            with self.assertRaises(ValueError):
                load_verified_checkpoint(path)

    def test_accumulated_unequal_batches_match_single_batch_optimizer_update(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)
            def forward_logits(self, value):
                return self.linear(value)
        dataset = TensorDataset(torch.tensor([[.3, .5], [-1, .4], [.8, -.7]]), torch.tensor([0, 1, 1]))
        whole = DataLoader(dataset, batch_size=3)
        micro = DataLoader(dataset, batch_size=2)
        model_whole, model_micro = Tiny(), Tiny()
        model_micro.load_state_dict(model_whole.state_dict())
        contract = {'frontend_id': 'test', 'dataset_manifest_hash': 'a' * 64,
                    'input_shape': [1, 2], 'label_convention': 'real=0_fake=1'}
        def config(accumulation):
            return {'training': {'epochs': 1, 'amp': False, 'gradient_clip': 100,
                                 'gradient_accumulation_steps': accumulation}, 'contract': contract}
        reference = Trainer(model_whole, whole, whole, config(1), device='cpu')
        candidate = Trainer(model_micro, micro, micro, config(2), device='cpu')
        reference.train_epoch()
        candidate.train_epoch()
        for expected, actual in zip(reference.model.parameters(), candidate.model.parameters()):
            torch.testing.assert_close(expected, actual)
            # Adam's first and second moments expose differently weighted
            # gradients even when first-step parameter signs happen to agree.
            for moment in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(reference.optimizer.state[expected][moment], candidate.optimizer.state[actual][moment])

if __name__ == '__main__':
    unittest.main()
