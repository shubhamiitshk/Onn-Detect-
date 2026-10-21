"""Capture actual source bytes and runtime dependencies, including dirty edits."""
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from data.contracts import sha256_file

# The source areas whose bytes can change what a trained model IS.
TRAINING_SOURCE_AREAS = ('data', 'models', 'training', 'utils')


def training_source_files():
    """Every training-source file and its hash."""
    root = Path(__file__).resolve().parents[1]
    return {p.relative_to(root).as_posix(): sha256_file(p)
            for area in TRAINING_SOURCE_AREAS
            for p in sorted((root / area).rglob('*.py'))}


def training_code_hash():
    return hashlib.sha256(json.dumps(training_source_files(), sort_keys=True).encode()).hexdigest()


def _git_has_object(repository, revision) -> bool:
    try:
        subprocess.check_output(['git', 'cat-file', '-e', f'{revision}^{{commit}}'],
                                cwd=repository, text=True,
                                stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def training_source_drift(repository, revision):
    """The training-source files that differ from the checkpoint's baseline.

    The aggregate hash can say THAT something changed but not WHAT. A
    blanket "retrain every base model" answer is wrong when the change is,
    say, a new release-acceptance module that no model depends on. This
    lists the actual drift so a release can record exactly which bytes
    moved since the checkpoint was trained.

    Returns ``(files, basis)``. ``files`` is ``None`` when the drift could
    NOT be computed - an unknown drift is reported as unknown, never as an
    empty list, because "nothing changed" and "I could not tell" are very
    different statements.

    The baseline is the checkpoint's own recorded revision when that commit
    exists in this repository, and the repository HEAD otherwise (some
    checkpoints were trained in a different lineage). The basis that was
    actually used is returned so the release can record it.
    """
    repository = Path(repository).resolve()
    areas = [f"ml/src/{area}" for area in TRAINING_SOURCE_AREAS]
    basis = revision if (revision and _git_has_object(repository, revision)) else "HEAD"
    try:
        changed = subprocess.check_output(
            ['git', 'diff', '--name-only', basis, '--', *areas],
            cwd=repository, text=True).splitlines()
        untracked = subprocess.check_output(
            ['git', 'ls-files', '--others', '--exclude-standard', '--', *areas],
            cwd=repository, text=True).splitlines()
    except (subprocess.CalledProcessError, OSError):
        return None, basis
    files = sorted({name.replace('\\', '/') for name in changed + untracked if name.strip()})
    return files, basis

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
