"""Capture actual source bytes and runtime dependencies, including dirty edits."""
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from data.contracts import sha256_file

def training_code_hash():
    root = Path(__file__).resolve().parents[1]
    files = {p.relative_to(root).as_posix(): sha256_file(p)
             for area in ('data', 'models', 'training', 'utils')
             for p in sorted((root / area).rglob('*.py'))}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()

def source_snapshot(repository):
    repository = Path(repository).resolve()
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=repository, text=True).splitlines()
    files = {}
    for name in sorted(set(names)):
        path = repository / name
        if not path.is_file():
            continue
        if path.suffix in ('.py', '.kt', '.kts', '.yaml', '.yml', '.toml', '.xml') or path.name.startswith('requirements') or path.name in ('Dockerfile', 'Makefile', 'gradle-wrapper.properties'):
            files[name.replace('\\', '/')] = sha256_file(path)
    if not files:
        raise ValueError('Source snapshot is empty')
    versions = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions() if dist.metadata.get('Name')}
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=repository, text=True).strip())
    return {'schema_version': 1, 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
            'source_dirty': dirty, 'files': files, 'environment_packages': dict(sorted(versions.items()))}
