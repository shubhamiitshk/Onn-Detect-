import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('package_chat_model', ROOT / 'scripts/package_chat_model.py')
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


def write_fixture(path, architecture='qwen35', dimension=1024):
    def string(value):
        raw = value.encode()
        return struct.pack('<Q', len(raw)) + raw
    header = b'GGUF' + struct.pack('<IQQ', 3, 1, 2)
    header += string('general.architecture') + struct.pack('<I', 8) + string(architecture)
    header += string(architecture + '.embedding_length') + struct.pack('<II', 4, dimension)
    with path.open('wb') as stream:
        stream.write(header)
        stream.truncate(50 * 1024**2)


class ChatModelPackTest(unittest.TestCase):
    def test_reads_real_metadata_fields_not_filename_guesses(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'not-the-model-name.gguf'
            write_fixture(path)
            self.assertEqual(('qwen35', 1024), packager.gguf_identity(path))

    def test_pack_preserves_exact_source_bytes_and_has_no_forensic_head(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            language = root / 'source.gguf'
            write_fixture(language)
            original = packager.file_contract(language, 'unused')['sha256']
            output = root / 'conversation.zip'
            manifest = packager.build_pack(language, None, output, model_id='local/test-qwen', display_name='Test metadata fixture')
            self.assertEqual(original, manifest['language']['sha256'])
            self.assertEqual(original, packager.file_contract(language, 'unused')['sha256'])
            with zipfile.ZipFile(output) as archive:
                self.assertEqual({'chat_model.json', 'language.gguf'}, set(archive.namelist()))
                declared = json.loads(archive.read('chat_model.json'))
                self.assertEqual('conversation', declared['purpose'])
                self.assertNotIn('head_contract', declared)
                self.assertIsNone(declared['provenance']['upstream_revision'])

    def test_unsupported_architecture_does_not_create_a_pack(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_fixture(root / 'embedding.gguf', architecture='embeddinggemma')
            with self.assertRaises(ValueError):
                packager.build_pack(root / 'embedding.gguf', None, root / 'no.zip', model_id='bad', display_name='bad')
            self.assertFalse((root / 'no.zip').exists())

    def test_existing_output_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_fixture(root / 'language.gguf')
            output = root / 'existing.zip'
            output.write_bytes(b'preserved user artifact')
            with self.assertRaises(FileExistsError):
                packager.build_pack(root / 'language.gguf', None, output, model_id='test', display_name='test')
            self.assertEqual(b'preserved user artifact', output.read_bytes())

    def test_fake_projector_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_fixture(root / 'language.gguf')
            (root / 'projector.gguf').write_bytes(b'not a GGUF')
            with self.assertRaises(ValueError):
                packager.build_pack(root / 'language.gguf', root / 'projector.gguf', root / 'no.zip', model_id='test', display_name='test')
            self.assertFalse((root / 'no.zip').exists())


if __name__ == '__main__':
    unittest.main()
