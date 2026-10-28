"""Shared test fixtures, synthetic media generators, and verification helpers for E2E tests."""
from __future__ import annotations

import base64
import datetime
import hashlib
import io
import json
import math
import os
import shutil
import struct
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

# Repository root and key directory paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "models"
CONSTANTS_DIR = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "constants"
SCRIPTS_DIR = REPO_ROOT / "scripts"
DOCS_DIR = REPO_ROOT / "docs"
ANDROID_DIR = REPO_ROOT / "android"

# Feature schema order invariant per CLAUDE.md
FEATURE_SCHEMA = [
    "srm_score",
    "fft_score",
    "kl_score",
    "flatness_score",
    "resolution_norm",
    "is_compressed",
    "semantic_probe_score",
    "c2pa_present",
    "lip_sync_corr",
    "landmark_jitter",
]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def temp_workspace() -> Path:
    tmp = tempfile.mkdtemp(prefix="onn_e2e_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


def generate_synthetic_pcm_wav(
    sample_rate: int = 16000,
    duration_sec: float = 1.0,
    freq_hz: float = 440.0,
    amplitude: float = 0.5,
    noise_level: float = 0.0,
) -> bytes:
    """Generates a 16-bit mono PCM WAV buffer."""
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * freq_hz * t)
    if noise_level > 0.0:
        signal += np.random.normal(0, noise_level, num_samples)
    signal = np.clip(signal, -1.0, 1.0)
    int16_samples = (signal * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_samples.tobytes())
    return buf.getvalue()


def generate_synthetic_rgb_image(
    width: int = 128,
    height: int = 128,
    pattern: str = "gradient",
) -> bytes:
    """Generates an uncompressed or simple PNG image buffer."""
    from PIL import Image

    if pattern == "gradient":
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                arr[y, x, 0] = int((x / max(1, width - 1)) * 255)
                arr[y, x, 1] = int((y / max(1, height - 1)) * 255)
                arr[y, x, 2] = 128
    elif pattern == "noise":
        arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    elif pattern == "flat":
        arr = np.full((height, width, 3), 128, dtype=np.uint8)
    elif pattern == "checkerboard":
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        block = 16
        for y in range(height):
            for x in range(width):
                if ((x // block) + (y // block)) % 2 == 0:
                    arr[y, x, :] = 255
    else:
        arr = np.zeros((height, width, 3), dtype=np.uint8)

    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_canonical_receipt_payload(
    media_sha256: Optional[str] = None,
    overall_score: float = 0.8500,
    confidence: float = 0.9500,
    verdict: str = "LIKELY_AI",
    security_tier: str = "STRONGBOX",
    nonce: str = "e2e-test-challenge-nonce-001",
    timestamp_epoch_ms: int = 1792963200000,
    signals: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Generates a valid dictionary matching CanonicalReceiptPayload schema."""
    if media_sha256 is None:
        media_sha256 = hashlib.sha256(b"sample-e2e-media-content").hexdigest()

    if signals is None:
        signals = {
            "srm": 0.8200,
            "semantic_probe": 0.8800,
            "fft": 0.4500,
            "kl": 0.5100,
        }

    return {
        "challenge_nonce": nonce,
        "confidence": confidence,
        "device_security_tier": security_tier,
        "media_sha256": media_sha256,
        "model_hashes": {
            "srm_forensic_cnn_int8.tflite": "08397e56b53e506ef89b290f114f167babb2ca530a52320ec770b519bf307ca1",
            "siglip_probe_int8.tflite": "1b8c4f3de7822196c3c0ebf59f683d86f24d5fa2476fb1d9312a1e14b1519c7b",
        },
        "model_release": "onn-prod-v10",
        "overall_score": overall_score,
        "schema_version": "onn-receipt-v1",
        "signals": signals,
        "timestamp_epoch_ms": timestamp_epoch_ms,
        "verdict": verdict,
    }


def generate_mock_face_landmarks(
    num_frames: int = 10,
    has_face: bool = True,
    gapped_indices: Optional[List[int]] = None,
) -> List[Optional[List[Dict[str, float]]]]:
    """Generates synthetic 478 MediaPipe facial landmark dictionaries."""
    if not has_face:
        return [None] * num_frames

    gapped = set(gapped_indices or [])
    frames: List[Optional[List[Dict[str, float]]]] = []
    for f in range(num_frames):
        if f in gapped:
            frames.append(None)
            continue

        # Standard face layout: left eye around (0.35, 0.4), right eye around (0.65, 0.4), nose (0.5, 0.5)
        # Mouth around (0.5, 0.7) with small movement over time
        mouth_delta = 0.005 * math.sin(f * 0.5)
        landmarks = []
        for i in range(478):
            if i == 33:  # Left eye outer
                landmarks.append({"x": 0.35, "y": 0.40, "z": 0.0})
            elif i == 263:  # Right eye outer
                landmarks.append({"x": 0.65, "y": 0.40, "z": 0.0})
            elif i == 1:  # Nose tip
                landmarks.append({"x": 0.50, "y": 0.50, "z": 0.0})
            elif i == 13:  # Upper lip
                landmarks.append({"x": 0.50, "y": 0.68 - mouth_delta, "z": 0.0})
            elif i == 14:  # Lower lip
                landmarks.append({"x": 0.50, "y": 0.72 + mouth_delta, "z": 0.0})
            elif i == 61:  # Left mouth corner
                landmarks.append({"x": 0.42, "y": 0.70, "z": 0.0})
            elif i == 291:  # Right mouth corner
                landmarks.append({"x": 0.58, "y": 0.70, "z": 0.0})
            else:
                landmarks.append({"x": 0.50 + 0.001 * (i % 10), "y": 0.50 + 0.001 * (i // 10), "z": 0.0})
        frames.append(landmarks)
    return frames
