"""Invariants for the Fisher between-source nuisance subspace projector.

The NumPy implementation lives in ml/scripts/train_siglip_tap.py and the
PyTorch implementation lives in ml/src/models/orthogonal_subspace_probe.py.
They serve different roles (offline analysis vs. in-graph training) and are
NOT yet numerically reconciled; this test pins the NumPy projector's
documented invariants so regressions are caught.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_siglip_tap import FisherSourceSubspaceProjector


def test_projector_identity_with_single_source():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 8))
    sources = ["a"] * 20
    p = FisherSourceSubspaceProjector(embed_dim=8, max_rank=3)
    p.fit_real_sources(X, sources)
    out = p.project(X)
    assert np.allclose(out, X)


def test_projector_removes_between_source_basis():
    rng = np.random.default_rng(1)
    # Two sources with a clear mean shift along axis 0.
    Xa = rng.normal(loc=0.0, scale=0.1, size=(20, 4))
    Xb = rng.normal(loc=5.0, scale=0.1, size=(20, 4))
    X = np.vstack([Xa, Xb])
    sources = ["a"] * 20 + ["b"] * 20
    p = FisherSourceSubspaceProjector(embed_dim=4, max_rank=3)
    p.fit_real_sources(X, sources)
    proj = p.project(X)
    # Between-source centroid difference should be (near) zero after projection.
    ma = proj[:20].mean(axis=0)
    mb = proj[20:].mean(axis=0)
    assert np.linalg.norm(ma - mb) < 1e-6


def test_projector_preserves_dimension():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 6))
    sources = (["a"] * 10) + (["b"] * 10) + (["c"] * 10)
    p = FisherSourceSubspaceProjector(embed_dim=6, max_rank=2)
    p.fit_real_sources(X, sources)
    out = p.project(X)
    assert out.shape == X.shape
    assert p.proj_matrix.shape == (6, 6)