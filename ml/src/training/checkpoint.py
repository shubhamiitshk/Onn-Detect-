"""Versioned checkpoint contract and atomic saves."""
import os
from pathlib import Path
import torch

def atomic_torch_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)

def load_verified_checkpoint(path, expected_contract=None):
    state = torch.load(path, map_location='cpu', weights_only=True)
    contract = state.get('contract')
    if state.get('schema_version') != 1 or not contract:
        raise ValueError('Legacy checkpoint has no validated data/frontend lineage; retrain explicitly')
    for field in ('frontend_id', 'dataset_manifest_hash', 'label_convention', 'input_shape'):
        if not contract.get(field):
            raise ValueError(f'Checkpoint contract missing {field}')
    if contract['label_convention'] != 'real=0_fake=1':
        raise ValueError('Checkpoint label convention mismatch')
    if expected_contract is not None and contract != expected_contract:
        raise ValueError('Checkpoint frontend, architecture or dataset contract differs from this run')
    return state
