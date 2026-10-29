#!/usr/bin/env python3
"""Package existing local Qwen3.5 GGUF weights for on-device conversation.

This does not train, download, validate a detector, sign a publisher identity,
or change the supplied weights. File hashes preserve the imported byte identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import zipfile

MAX_TOTAL_BYTES = 3 * 1024**3


def gguf_identity(path: Path) -> tuple[str, int]:
    with path.open('rb') as stream:
        def read(fmt: str) -> int:
            raw = stream.read(struct.calcsize('<' + fmt))
            if len(raw) != struct.calcsize('<' + fmt):
                raise ValueError('Truncated GGUF metadata')
            return struct.unpack('<' + fmt, raw)[0]
        def string(limit: int = 1_048_576) -> str:
            length = read('Q')
            if length > limit or stream.tell() + length > path.stat().st_size:
                raise ValueError('Invalid GGUF string')
            return stream.read(length).decode('utf-8')
        sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
        def skip(kind: int, depth: int = 0) -> None:
            if depth > 2:
                raise ValueError('Invalid metadata nesting')
            if kind == 8:
                size = read('Q')
                if size > path.stat().st_size - stream.tell(): raise ValueError('Truncated metadata')
                stream.seek(size, 1)
            elif kind == 9:
                sub, count = read('I'), read('Q')
                if count > 2_000_000: raise ValueError('Excessive metadata array')
                if sub in sizes: stream.seek(sizes[sub] * count, 1)
                else:
                    for _ in range(count): skip(sub, depth + 1)
            elif kind in sizes: stream.seek(sizes[kind], 1)
            else: raise ValueError('Unsupported GGUF metadata type')
        if stream.read(4) != b'GGUF' or read('I') not in (2,3) or read('Q') == 0:
            raise ValueError('Not a supported GGUF language model')
        count = read('Q')
        if not 0 < count <= 2_000_000: raise ValueError('Invalid metadata count')
        architecture = None
        dimension = None
        for _ in range(count):
            if stream.tell() > 64 * 1024**2: raise ValueError('GGUF identity header is too large')
            key, kind = string(1024), read('I')
            if key == 'general.architecture' and kind == 8: architecture = string(100)
            elif key.endswith('.embedding_length') and kind in (4,5): dimension = read('I')
            else: skip(kind)
            if architecture is not None and dimension is not None:
                return architecture, dimension
    raise ValueError('GGUF identity metadata is absent')


def file_contract(path: Path, name: str) -> dict:
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'file': name, 'bytes': path.stat().st_size, 'sha256': sha}


def build_pack(language: Path, projector: Path | None, destination: Path, *, model_id: str, display_name: str) -> dict:
    language = language.resolve(strict=True)
    projector = projector.resolve(strict=True) if projector else None
    destination = destination.resolve()
    if destination.exists(): raise FileExistsError(f'Refusing to overwrite {destination}')
    if language.stat().st_size < 50 * 1024**2: raise ValueError('Language GGUF is too small')
    architecture, dimension = gguf_identity(language)
    if architecture != 'qwen35' or dimension not in (1024, 2048, 2560):
        raise ValueError(f'This runtime currently supports tested Qwen3.5 models, not {architecture}/{dimension}')
    if projector:
        with projector.open('rb') as stream:
            if stream.read(4) != b'GGUF': raise ValueError('Vision projector is not GGUF')
        if not 1024**2 <= projector.stat().st_size <= 768 * 1024**2: raise ValueError('Invalid projector size')
    if language.stat().st_size + (projector.stat().st_size if projector else 0) > MAX_TOTAL_BYTES:
        raise ValueError('Pack exceeds the 3 GB runtime limit')
    manifest = {
        'schema_version': 1, 'purpose': 'conversation', 'model_id': model_id,
        'display_name': display_name, 'architecture': architecture, 'hidden_dim': dimension,
        'language': file_contract(language, 'language.gguf'),
        'projector': file_contract(projector, 'vision.gguf') if projector else None,
        'provenance': {'source_kind': 'existing_user_provided_local_files',
                       'identity': 'sha256_of_exact_input_bytes',
                       'upstream_revision': None,
                       'note': 'Upstream revision is not established by a local file hash; this pack does not grant forensic validation or publisher authentication.'},
        'runtime': {'backend': 'llama.cpp', 'execution': 'CPU', 'vision_resolution_cap_px': 256,
                    'note': 'RAM and token speed must be measured on the target phone. The supplied language/projector pairing requires runtime smoke and task-quality evaluation; it does not grant forensic calibration.'},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix='chat-pack-', suffix='.zip', dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.writestr('chat_model.json', json.dumps(manifest, indent=2) + '\n')
            archive.write(language, 'language.gguf')
            if projector: archive.write(projector, 'vision.gguf')
        # Rename on the same volume only after the complete archive exists.
        temporary.rename(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', type=Path, required=True)
    parser.add_argument('--projector', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-id', default='Qwen/Qwen3.5-0.8B')
    parser.add_argument('--display-name', default='Qwen 3.5 0.8B · 4-bit')
    args = parser.parse_args()
    manifest = build_pack(args.language, args.projector, args.output, model_id=args.model_id, display_name=args.display_name)
    print(json.dumps({'output': str(args.output.resolve()), 'manifest': manifest}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
