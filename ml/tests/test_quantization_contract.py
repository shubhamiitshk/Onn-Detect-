"""Unsupported integer conversions cannot silently become release models."""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class QuantizationTests(unittest.TestCase):
    def test_saved_model_integer_failure_requires_explicit_fallback(self):
        class Converter:
            def __init__(self):
                self.target_spec = types.SimpleNamespace(supported_ops=None)
                self.calls = 0
            def convert(self):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError('Unsupported integer operation')
                return b'\0\0\0\0TFL3diagnostic-only'
        converters = []
        def make_converter(path):
            converter = Converter()
            converters.append(converter)
            return converter
        fake_tf = types.SimpleNamespace(lite=types.SimpleNamespace(
            TFLiteConverter=types.SimpleNamespace(from_saved_model=make_converter),
            Optimize=types.SimpleNamespace(DEFAULT='default'),
            OpsSet=types.SimpleNamespace(TFLITE_BUILTINS_INT8='int8', TFLITE_BUILTINS='float')))
        spec = importlib.util.spec_from_file_location('quantize_contract_fixture', ROOT / 'src/export/quantize_int8.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'tensorflow': fake_tf}):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / 'saved'
            saved.mkdir()
            (saved / 'saved_model.pb').write_bytes(b'diagnostic fixture')
            destination = Path(directory) / 'model.tflite'
            with self.assertRaises(RuntimeError):
                module.quantize_saved_model(str(saved), lambda: iter(()), str(destination))
            self.assertEqual(converters[0].calls, 1)
            self.assertFalse(destination.exists())
            result = module.quantize_saved_model(str(saved), lambda: iter(()), str(destination), allow_dynamic_range=True)
            self.assertEqual(result, 'dynamic-range')
            self.assertEqual(converters[1].calls, 2)


if __name__ == '__main__':
    unittest.main()
