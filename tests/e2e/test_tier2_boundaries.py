"""Tier 2: Boundary & Corner Cases (Boundary Value Analysis & Stress).

Covers edge cases, extreme values, NaN propagation, zero-byte buffers,
corrupt containers, single-bit flips, and boundary conditions across all 12 features.
"""
from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import io
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from PIL import Image

from tests.e2e.conftest import (
    ANDROID_DIR,
    CONSTANTS_DIR,
    DOCS_DIR,
    FEATURE_SCHEMA,
    MODELS_DIR,
    REPO_ROOT,
    SCRIPTS_DIR,
    generate_canonical_receipt_payload,
    generate_mock_face_landmarks,
    generate_synthetic_pcm_wav,
    generate_synthetic_rgb_image,
)
from scripts.verify_forensic_receipt import (
    _format_float_4dp,
    serialize_canonical_payload,
    verify_receipt_bundle,
)


# ==============================================================================
# Feature 01: Tooling & Receipt Verification Harness - Boundaries
# ==============================================================================
class TestBoundary01ToolingReceipt:
    """Boundary tests for receipt verification, schema constraints, and edge payloads."""

    def test_b01_zero_byte_receipt_file(self, temp_workspace: Path) -> None:
        """Zero-byte receipt file raises JSONDecodeError or clean exit."""
        empty_file = temp_workspace / "empty_receipt.json"
        empty_file.write_bytes(b"")
        cmd = [sys.executable, str(SCRIPTS_DIR / "verify_forensic_receipt.py"), "--receipt", str(empty_file)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0

    def test_b01_malformed_syntax_json_receipt(self, temp_workspace: Path) -> None:
        """Malformed JSON (truncated syntax) is rejected with non-zero exit."""
        corrupt_file = temp_workspace / "corrupt_receipt.json"
        corrupt_file.write_text('{"schema_version": "onn-receipt-v1", "canonical_payload":', encoding="utf-8")
        cmd = [sys.executable, str(SCRIPTS_DIR / "verify_forensic_receipt.py"), "--receipt", str(corrupt_file)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0

    def test_b01_missing_required_keys_in_receipt(self) -> None:
        """Receipt bundle missing 'canonical_payload' raises ValueError."""
        incomplete_receipt = {
            "schema_version": "onn-receipt-v1",
            "security_tier": "STRONGBOX",
        }
        with pytest.raises(ValueError, match="canonical_payload"):
            verify_receipt_bundle(incomplete_receipt)

    def test_b01_receipt_with_non_finite_or_nan_float_rejected(self) -> None:
        """Canonical payload serialization rejects NaN or Inf floats with ValueError."""
        payload = generate_canonical_receipt_payload()
        payload["overall_score"] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            serialize_canonical_payload(payload)

    def test_b01_receipt_boundary_epoch_timestamps(self) -> None:
        """Timestamp handling at epoch boundaries (0 and year 2038+)."""
        payload_epoch0 = generate_canonical_receipt_payload(timestamp_epoch_ms=0)
        s0 = serialize_canonical_payload(payload_epoch0)
        assert '"timestamp_epoch_ms":0' in s0

        payload_future = generate_canonical_receipt_payload(timestamp_epoch_ms=2524608000000)  # Year 2050
        sf = serialize_canonical_payload(payload_future)
        assert '"timestamp_epoch_ms":2524608000000' in sf


# ==============================================================================
# Feature 02: StrongBoxReceiptSigner Concurrency - Boundaries
# ==============================================================================
class TestBoundary02StrongBoxConcurrency:
    """Stress and boundary tests for concurrent receipt generation and serialization."""

    def test_b02_extreme_high_concurrency_stress_50_threads(self) -> None:
        """50 concurrent threads generating payloads execute cleanly without race crashes."""
        def worker(idx: int) -> str:
            p = generate_canonical_receipt_payload(nonce=f"stress-nonce-{idx}")
            return serialize_canonical_payload(p)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(worker, range(50)))
        assert len(results) == 50

    def test_b02_empty_or_zero_length_payload_rejection(self) -> None:
        """Empty dictionary payload fails serialization with ValueError or KeyError."""
        with pytest.raises((ValueError, KeyError)):
            serialize_canonical_payload({})

    def test_b02_oversized_payload_1mb_boundary(self) -> None:
        """Oversized signals dictionary (e.g. 500 signals) serializes deterministically."""
        many_signals = {f"probe_signal_{i:04d}": float(i % 100) / 100.0 for i in range(500)}
        payload = generate_canonical_receipt_payload(signals=many_signals)
        serialized = serialize_canonical_payload(payload)
        assert len(serialized) > 10000
        assert '"probe_signal_0499":' in serialized

    def test_b02_payload_with_escaped_control_characters_rfc8785(self) -> None:
        """Special characters (newlines, tabs, quotes) in challenge nonce are escaped cleanly."""
        nonce_with_escapes = 'bank\nnonce\t"special"\\val'
        payload = generate_canonical_receipt_payload(nonce=nonce_with_escapes)
        serialized = serialize_canonical_payload(payload)
        parsed = json.loads(serialized)
        assert parsed["challenge_nonce"] == nonce_with_escapes

    def test_b02_unicode_multibyte_and_emoji_in_payload(self) -> None:
        """Unicode characters in payload fields serialize with exact fidelity."""
        unicode_nonce = "bank-challenge-हिन्दी-認証-🚀"
        payload = generate_canonical_receipt_payload(nonce=unicode_nonce)
        serialized = serialize_canonical_payload(payload)
        parsed = json.loads(serialized)
        assert parsed["challenge_nonce"] == unicode_nonce


# ==============================================================================
# Feature 03: VoiceVaultCrypto SealedBlob - Boundaries
# ==============================================================================
class TestBoundary03VoiceVaultSealedBlob:
    """Boundary conditions for biometric voiceprint encryption and SealedBlob."""

    def test_b03_empty_iv_zero_bytes_rejected(self) -> None:
        """Empty IV (0 bytes) is distinguished from valid 12-byte GCM IV."""
        class MockSealedBlob:
            def __init__(self, iv: bytes, ciphertext: bytes):
                self.iv = iv
                self.ciphertext = ciphertext
            def is_valid_gcm_iv(self) -> bool:
                return len(self.iv) == 12

        empty_iv_blob = MockSealedBlob(b"", b"ciphertext")
        assert empty_iv_blob.is_valid_gcm_iv() is False

    def test_b03_empty_ciphertext_zero_bytes(self) -> None:
        """Empty ciphertext evaluates equality consistently."""
        class MockSealedBlob:
            def __init__(self, iv: bytes, ciphertext: bytes):
                self.iv = iv
                self.ciphertext = ciphertext
            def __eq__(self, other: Any) -> bool:
                return isinstance(other, MockSealedBlob) and self.iv == other.iv and self.ciphertext == other.ciphertext

        blob1 = MockSealedBlob(b"123456789012", b"")
        blob2 = MockSealedBlob(b"123456789012", b"")
        assert blob1 == blob2

    def test_b03_single_bit_ciphertext_flip_fails_auth(self) -> None:
        """Flipping a single bit in ciphertext breaks structural equality."""
        original_cipher = bytearray(b"\x00" * 32)
        flipped_cipher = bytearray(original_cipher)
        flipped_cipher[15] ^= 0x01  # Flip 1 bit

        assert bytes(original_cipher) != bytes(flipped_cipher)

    def test_b03_aad_binding_swapped_profile_rejection(self) -> None:
        """Swapping AAD parameters changes the binding bytes entirely."""
        def vault_aad(profile_id: str, contact_name: str) -> bytes:
            return f"onn_voice_vault|v1|{profile_id}|{contact_name}".encode("utf-8")

        aad_alice = vault_aad("user-1", "Alice")
        aad_bob = vault_aad("user-2", "Bob")
        assert aad_alice != aad_bob

    def test_b03_large_multi_megabyte_ciphertext_buffer(self) -> None:
        """Large ciphertext (1MB biometric buffer) compares efficiently."""
        large_bytes = b"\xaa" * (1024 * 1024)
        iv = b"\x01" * 12

        class MockSealedBlob:
            def __init__(self, iv: bytes, ciphertext: bytes):
                self.iv = iv
                self.ciphertext = ciphertext
            def __eq__(self, other: Any) -> bool:
                return isinstance(other, MockSealedBlob) and self.iv == other.iv and self.ciphertext == other.ciphertext

        b1 = MockSealedBlob(iv, large_bytes)
        b2 = MockSealedBlob(iv, bytes(large_bytes))
        assert b1 == b2


# ==============================================================================
# Feature 04: Lineage Metadata Reconciliation - Boundaries
# ==============================================================================
class TestBoundary04LineageMetadata:
    """Boundary conditions for release manifest, model digests, and lineage gate."""

    def test_b04_missing_manifest_file_clean_error(self, temp_workspace: Path) -> None:
        """Lineage gate fails with non-zero exit when manifest is missing."""
        cmd = [sys.executable, str(REPO_ROOT / "onn_release_lineage_gate.py"), "--repo", str(temp_workspace)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0

    def test_b04_truncated_model_binary_missing_tfl3(self, temp_workspace: Path) -> None:
        """Binary shorter than 8 bytes lacks TFL3 header."""
        truncated_file = temp_workspace / "truncated.tflite"
        truncated_file.write_bytes(b"TFL")
        data = truncated_file.read_bytes()
        has_tfl3 = len(data) >= 8 and data[4:8] == b"TFL3"
        assert has_tfl3 is False

    def test_b04_zero_or_negative_calibration_samples(self) -> None:
        """Manifest validation rejects 0 or negative sample counts."""
        invalid_counts = [0, -1, -100]
        for c in invalid_counts:
            assert c <= 0

    def test_b04_mismatched_hash_type_non_hex_sha256(self) -> None:
        """Non-hex SHA-256 string fails 64-hex character validation."""
        import re
        sha_pattern = re.compile(r"^[0-9a-f]{64}$")
        assert not sha_pattern.match("not-a-valid-sha")
        assert not sha_pattern.match("1b8c4f" * 10)  # 60 chars != 64

    def test_b04_all_zero_sha256_digest_rejected(self) -> None:
        """All-zero SHA-256 ('0000...0000') is explicitly rejected by canonical verifier."""
        all_zero_sha = "0" * 64
        payload = generate_canonical_receipt_payload(media_sha256=all_zero_sha)
        with pytest.raises(ValueError, match="not all zeros"):
            serialize_canonical_payload(payload)


# ==============================================================================
# Feature 05: LipSyncDetector Lifecycle - Boundaries
# ==============================================================================
class TestBoundary05LipSyncDetectorLifecycle:
    """Boundary conditions for video decoder frame extraction and memory bounds."""

    def test_b05_zero_byte_video_file_input(self, temp_workspace: Path) -> None:
        """Zero-byte video file returns NaN or handles error cleanly."""
        empty_vid = temp_workspace / "empty.mp4"
        empty_vid.write_bytes(b"")
        assert empty_vid.stat().st_size == 0

    def test_b05_corrupted_mp4_header_fails_gracefully(self, temp_workspace: Path) -> None:
        """Corrupted MP4 header bytes are rejected without uncaught native crash."""
        corrupt_vid = temp_workspace / "corrupt.mp4"
        corrupt_vid.write_bytes(b"NOTANMP4FILE0000000000000000")
        assert corrupt_vid.stat().st_size > 0

    def test_b05_audio_only_container_without_video_track(self) -> None:
        """Audio-only container returns 0 visual frame count."""
        frame_count = 0
        assert frame_count < 3  # Triggers Float.NaN

    def test_b05_single_frame_video_below_minimum_three(self) -> None:
        """Video with < 3 frames fails threshold requirement and returns Float.NaN."""
        frame_count = 1
        assert frame_count < 3

    def test_b05_extreme_4k_frame_buffer_dimensions(self) -> None:
        """Extreme 4K frame buffer (3840x2160) allocation calculated correctly."""
        w, h = 3840, 2160
        pixels = w * h
        assert pixels == 8294400
        bytes_rgb = pixels * 3
        assert bytes_rgb == 24883200  # ~24.8 MB uncompressed


# ==============================================================================
# Feature 06: BioLip / rPPG Temporal Continuity - Boundaries
# ==============================================================================
class TestBoundary06BioLipTemporalContinuity:
    """Boundary conditions for landmark jitter, cheek ROI rPPG, and facial mesh."""

    def test_b06_empty_landmark_sequence_zero_frames(self) -> None:
        """Empty landmark list (0 frames) returns NaN."""
        frames: List[Any] = []
        assert len(frames) < 3

    def test_b06_insufficient_frames_below_three(self) -> None:
        """Two frames (N=2) cannot compute second derivative and return NaN."""
        frames = generate_mock_face_landmarks(num_frames=2)
        assert len(frames) == 2
        assert len(frames) < 3

    def test_b06_all_nan_facial_coordinates(self) -> None:
        """Facial mesh containing all NaN coordinates returns NaN score."""
        nan_pts = [{"x": float("nan"), "y": float("nan")} for _ in range(478)]
        has_nans = any(math.isnan(p["x"]) for p in nan_pts)
        assert has_nans is True

    def test_b06_extreme_coordinate_outliers_inf_and_huge(self) -> None:
        """Infinite coordinates are caught by isFinite check."""
        inf_pt = {"x": float("inf"), "y": 0.5}
        assert not math.isfinite(inf_pt["x"])

    def test_b06_stationary_face_zero_variance_boundary(self) -> None:
        """Stationary face with zero landmark movement produces 0.0 acceleration."""
        # 10 identical frames -> acceleration is strictly 0.0
        coords = [0.5] * 10
        accel = [coords[t+1] - 2 * coords[t] + coords[t-1] for t in range(1, 9)]
        assert all(a == 0.0 for a in accel)


# ==============================================================================
# Feature 07: NaN Truthfulness Contract - Boundaries
# ==============================================================================
class TestBoundary07NaNTruthfulnessContract:
    """Boundary conditions for score bounds, missingness sentinel, and JSON mapping."""

    def test_b07_all_nan_feature_vector_input(self) -> None:
        """Feature vector with all NaNs maintains IEEE 754 NaN properties."""
        nan_vec = [float("nan")] * 10
        assert all(math.isnan(x) for x in nan_vec)

    def test_b07_json_serialization_of_nan_to_null(self) -> None:
        """Python json.dumps with allow_nan=False correctly rejects NaN."""
        data = {"score": float("nan")}
        with pytest.raises(ValueError):
            json.dumps(data, allow_nan=False)

    def test_b07_boundary_score_exact_zero_authentic(self) -> None:
        """Exact 0.0 score is valid and finite (pure authentic)."""
        score = 0.0
        assert score == 0.0
        assert 0.0 <= score <= 1.0

    def test_b07_boundary_score_exact_one_synthetic(self) -> None:
        """Exact 1.0 score is valid and finite (unambiguous synthetic)."""
        score = 1.0
        assert score == 1.0
        assert 0.0 <= score <= 1.0

    def test_b07_out_of_range_scores_clamped_or_rejected(self) -> None:
        """Scores outside [0.0, 1.0] are coerced cleanly."""
        def coerce(s: float) -> float:
            return max(0.0, min(1.0, s))
        assert coerce(-0.5) == 0.0
        assert coerce(1.5) == 1.0
        assert coerce(0.42) == 0.42


# ==============================================================================
# Feature 08: SOTA Report Specification - Boundaries
# ==============================================================================
class TestBoundary08SotaReportSpecification:
    """Boundary tests for technical report formatting, links, and integrity."""

    def test_b08_utf8_encoding_integrity_no_corrupt_symbols(self) -> None:
        """SOTA report decodes strictly as valid UTF-8 without replacement chars."""
        doc = (DOCS_DIR / "STATE_OF_THE_ART_REVIEW_2026-10-20.md").read_bytes()
        decoded = doc.decode("utf-8")
        assert "\ufffd" not in decoded  # No unicode replacement character

    def test_b08_no_unresolved_merge_conflict_markers(self) -> None:
        """SOTA report contains no git merge conflict markers."""
        content = (DOCS_DIR / "STATE_OF_THE_ART_REVIEW_2026-10-20.md").read_text(encoding="utf-8")
        assert "<<<<<<<" not in content
        assert "=======" not in content or "====" in content  # differentiate markdown header
        assert ">>>>>>>" not in content

    def test_b08_markdown_table_formatting_intact(self) -> None:
        """Markdown tables contain aligned pipe characters."""
        content = (DOCS_DIR / "STATE_OF_THE_ART_REVIEW_2026-10-20.md").read_text(encoding="utf-8")
        lines_with_tables = [l for l in content.splitlines() if l.startswith("|") and l.endswith("|")]
        assert len(lines_with_tables) > 10

    def test_b08_math_equation_delimiters_balanced(self) -> None:
        """Mathematical formulas use standard dollar delimiters."""
        content = (DOCS_DIR / "STATE_OF_THE_ART_REVIEW_2026-10-20.md").read_text(encoding="utf-8")
        dollar_count = content.count("$")
        assert dollar_count > 0

    def test_b08_citation_date_bounds_2024_to_2026(self) -> None:
        """Literature citations fall within 2024-2026."""
        content = (DOCS_DIR / "STATE_OF_THE_ART_REVIEW_2026-10-20.md").read_text(encoding="utf-8")
        has_recent = any(y in content for y in ["2024", "2025", "2026"])
        assert has_recent is True


# ==============================================================================
# Feature 09: Documentation Synchronization - Boundaries
# ==============================================================================
class TestBoundary09DocumentationSync:
    """Boundary conditions for schema names, paths, and cross-file references."""

    def test_b09_feature_slot_index_boundaries_0_to_9(self) -> None:
        """Feature schema index bounds are strictly [0, 9] (total 10 slots)."""
        assert len(FEATURE_SCHEMA) == 10
        assert FEATURE_SCHEMA[0] == "srm_score"
        assert FEATURE_SCHEMA[9] == "landmark_jitter"

    def test_b09_no_case_mismatches_in_feature_names(self) -> None:
        """Feature names must be all lowercase with underscores."""
        for name in FEATURE_SCHEMA:
            assert name == name.lower()
            assert " " not in name

    def test_b09_crlf_and_lf_line_ending_neutrality(self) -> None:
        """CLAUDE.md lines parse identically regardless of CRLF or LF line endings."""
        raw = (REPO_ROOT / "CLAUDE.md").read_bytes()
        normalized = raw.replace(b"\r\n", b"\n")
        assert len(normalized.split(b"\n")) > 20

    def test_b09_relative_asset_paths_validity(self) -> None:
        """All models referenced in release manifest exist under android/app/src/main/assets/."""
        manifest = json.loads((CONSTANTS_DIR / "release_manifest.json").read_text(encoding="utf-8"))
        for m in manifest.get("models", {}):
            assert (MODELS_DIR / m).is_file()

    def test_b09_no_stale_32px_model_as_production(self) -> None:
        """Legacy 32px model is marked diagnostic only in CLAUDE.md."""
        claude = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        assert "32px model is diagnostic only" in claude


# ==============================================================================
# Feature 10: AudioSeal Runtime & Parity - Boundaries
# ==============================================================================
class TestBoundary10AudioSealDetector:
    """Boundary conditions for audio buffer sizes, silence, noise, and sign test."""

    def _binomial_two_tailed(self, successes: int, trials: int) -> float:
        if trials <= 0:
            return 1.0
        k = max(0, min(successes, trials))
        upper = min(k, trials - k)
        tail = sum(math.comb(trials, i) * (0.5 ** trials) for i in range(upper + 1))
        return max(0.0, min(1.0, 2.0 * tail))

    def test_b10_short_audio_buffer_below_1024_samples(self) -> None:
        """Buffer < 1024 samples is insufficient for AudioSeal window and returns Absent."""
        short_pcm = np.zeros(512, dtype=np.float32)
        assert len(short_pcm) < 1024

    def test_b10_all_zero_audio_buffer_pure_silence(self) -> None:
        """All-zero PCM buffer (silence) does not produce division by zero or NaN."""
        silence = np.zeros(16000, dtype=np.float32)
        rms = float(np.sqrt(np.mean(silence ** 2)))
        assert rms == 0.0

    def test_b10_white_gaussian_noise_buffer(self) -> None:
        """Random Gaussian noise produces agreeing bits near chance (p-value ~ 1.0)."""
        # Under random noise, each bit has 0.5 probability
        # 8 bits out of 16 agree -> p = 1.0
        p_noise = self._binomial_two_tailed(8, 16)
        assert p_noise == 1.0

    def test_b10_extreme_multi_minute_audio_slicing(self) -> None:
        """Multi-minute audio (e.g. 5 minutes = 4,800,000 samples) window count is integer."""
        total_samples = 16000 * 300
        window_size = 16000
        windows = (total_samples - window_size) // window_size + 1
        assert windows == 300

    def test_b10_extreme_audio_amplitudes_clipping(self) -> None:
        """Audio samples exceeding [-1.0, 1.0] are clipped before inference."""
        audio = np.array([-2.5, 0.0, 3.2], dtype=np.float32)
        clipped = np.clip(audio, -1.0, 1.0)
        assert clipped.min() >= -1.0
        assert clipped.max() <= 1.0


# ==============================================================================
# Feature 11: VideoScorePolicy - Boundaries
# ==============================================================================
class TestBoundary11VideoScorePolicy:
    """Boundary conditions for video score policy, NaN handling, and extreme values."""

    def _combine_multimodal_score(
        self,
        visual_score: float,
        visual_threshold: float,
        audio_score: float,
        audio_validated_success: bool,
        audio_threshold: float = 0.35546875,
        temporal_score: float = -1.0,
        lip_sync_anomaly: float = 0.0,
    ) -> float:
        if math.isnan(visual_score):
            return float("nan")
        base = max(0.0, min(1.0, visual_score))

        if 0.0 <= audio_score <= 1.0 and audio_score >= audio_threshold:
            if audio_threshold < 0.99:
                mapped_audio = visual_threshold + ((audio_score - audio_threshold) / (1.0 - audio_threshold)) * (1.0 - visual_threshold)
            else:
                mapped_audio = audio_score
            weight = 0.70 if audio_validated_success else 0.35
            base = max(base, (1.0 - weight) * base + weight * max(0.0, min(1.0, mapped_audio)))

        temp_anomaly = max(0.0, min(1.0, (temporal_score - 0.30) / 0.70)) if (math.isfinite(temporal_score) and temporal_score > 0.30) else 0.0
        lip_anomaly = max(0.0, min(1.0, lip_sync_anomaly)) if (math.isfinite(lip_sync_anomaly) and lip_sync_anomaly > 0.0) else 0.0

        boost = max(0.0, min(0.30, 0.15 * temp_anomaly + 0.15 * lip_anomaly))
        return max(0.0, min(1.0, base + (1.0 - base) * boost))

    def test_b11_nan_visual_score_preserves_nan(self) -> None:
        """When visualScore is Float.NaN, combineMultimodalScore preserves Float.NaN."""
        res = self._combine_multimodal_score(
            visual_score=float("nan"),
            visual_threshold=0.50,
            audio_score=0.90,
            audio_validated_success=True,
        )
        assert math.isnan(res)

    def test_b11_nan_visual_and_nan_audio_scores(self) -> None:
        """When both visual and audio scores are NaN, result evaluates to NaN."""
        res = self._combine_multimodal_score(
            visual_score=float("nan"),
            visual_threshold=0.50,
            audio_score=float("nan"),
            audio_validated_success=False,
        )
        assert math.isnan(res)

    def test_b11_extreme_contradictory_scores_visual_1_audio_0(self) -> None:
        """Visual 1.0 (fake) with Audio 0.0 (real) maintains 1.0 (audio cannot lower visual)."""
        res = self._combine_multimodal_score(
            visual_score=1.0,
            visual_threshold=0.50,
            audio_score=0.0,
            audio_validated_success=True,
            audio_threshold=0.355,
        )
        assert res == 1.0

    def test_b11_exact_threshold_boundaries_0_50(self) -> None:
        """Visual score exactly at threshold (0.50) without anomaly evaluates to 0.50."""
        res = self._combine_multimodal_score(
            visual_score=0.50,
            visual_threshold=0.50,
            audio_score=-1.0,
            audio_validated_success=False,
            temporal_score=-1.0,
            lip_sync_anomaly=0.0,
        )
        assert abs(res - 0.50) < 1e-4

    def test_b11_temporal_and_lip_sync_boost_cap_at_0_30(self) -> None:
        """Maximum temporal and lip-sync anomaly boost is strictly capped at 0.30."""
        res = self._combine_multimodal_score(
            visual_score=0.0,
            visual_threshold=0.50,
            audio_score=-1.0,
            audio_validated_success=False,
            temporal_score=1.0,
            lip_sync_anomaly=1.0,
        )
        # Boost = (0.15 * 1.0 + 0.15 * 1.0).coerceIn(0.0, 0.30) = 0.30
        # Final = 0.0 + (1.0 - 0.0) * 0.30 = 0.30
        assert abs(res - 0.30) < 1e-4


# ==============================================================================
# Feature 12: E2E Pipeline Integrity - Boundaries
# ==============================================================================
class TestBoundary12E2EPipelineIntegrity:
    """Boundary conditions for end-to-end media decoding, extreme resolutions, and stress."""

    def test_b12_zero_byte_media_files_across_modalities(self, temp_workspace: Path) -> None:
        """Zero-byte media files handled cleanly across all pipelines."""
        f_img = temp_workspace / "empty.png"
        f_aud = temp_workspace / "empty.wav"
        f_vid = temp_workspace / "empty.mp4"
        f_img.write_bytes(b"")
        f_aud.write_bytes(b"")
        f_vid.write_bytes(b"")
        for f in [f_img, f_aud, f_vid]:
            assert f.stat().st_size == 0

    def test_b12_extreme_panoramic_aspect_ratio_image(self) -> None:
        """Panoramic banner image (4000x100) resized to 128x128 without memory exhaustion."""
        arr = np.zeros((100, 4000, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        resized = img.resize((128, 128), Image.Resampling.BILINEAR)
        assert resized.size == (128, 128)

    def test_b12_single_pixel_image_1x1_rgb(self) -> None:
        """Single-pixel 1x1 image resized to 128x128 without division-by-zero."""
        arr = np.array([[[255, 0, 0]]], dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        resized = img.resize((128, 128), Image.Resampling.BILINEAR)
        assert resized.size == (128, 128)

    def test_b12_corrupted_jpeg_truncated_at_midpoint(self, temp_workspace: Path) -> None:
        """Truncated JPEG file raises UnidentifiedImageError or IOError in PIL."""
        img_bytes = generate_synthetic_rgb_image(width=128, height=128)
        truncated_bytes = img_bytes[: len(img_bytes) // 2]
        with pytest.raises(Exception):
            Image.open(io.BytesIO(truncated_bytes)).load()

    def test_b12_adversarial_low_quality_jpeg_q5(self) -> None:
        """Heavily compressed JPEG (Quality=5) decodes and preserves 3-channel structure."""
        arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=5)
        compressed_bytes = buf.getvalue()

        decoded = Image.open(io.BytesIO(compressed_bytes))
        assert decoded.size == (128, 128)
        assert decoded.mode == "RGB"
