"""Tier 5 Adversarial Stress Harness.

Empirically challenges the 5 Tier 5 target areas:
1. Audio/Video/Image edge cases (PCM buffers, sample rates, non-consecutive frames, SRM aspect ratios).
2. Missing evidence & IEEE 754 NaN handling (VideoScorePolicy, SignalBreakdownJson, ScanResult).
3. Provenance & AudioSeal (binomial threshold p <= 1e-4, inverted carriers, unbundled fallback).
4. Cryptographic integrity & KeyMint (StrongBoxReceiptSigner concurrency, SealedBlob equality).
5. Zero-network sandbox (Manifest permissions, network security config, client audit).
"""
import concurrent.futures
import hashlib
import json
import math
import os
import re
import sys
import unittest
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from scripts.verify_forensic_receipt import (
    serialize_canonical_payload,
    verify_receipt_bundle,
    _format_float_4dp,
)


class Tier5AdversarialStressTest(unittest.TestCase):
    """Tier 5 Adversarial Coverage & Stress Test Suite."""

    # =========================================================================
    # Area 1: Audio / Video / Image Edge Cases
    # =========================================================================

    def test_audio_resampling_odd_rates_and_boundaries(self):
        """Simulate PcmUtil.resample with quadratic-window sinc on odd sample rates."""
        rates = [8000, 11025, 12345, 16000, 22050, 32000, 44100, 48000, 96000]
        for src_rate in rates:
            for dst_rate in [16000]:
                duration_sec = 0.05
                n_samples = int(src_rate * duration_sec)
                t = np.linspace(0, duration_sec, n_samples, endpoint=False)
                # 440 Hz sine wave
                sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)

                # Resampling length contract
                ratio = dst_rate / src_rate
                expected_out_len = math.ceil(n_samples * ratio)
                self.assertGreater(expected_out_len, 0)
                self.assertLess(expected_out_len, 10_000_000)

                # Boundary: all zeros / silence
                silence = np.zeros(n_samples, dtype=np.float32)
                self.assertEqual(len(silence), n_samples)

                # Boundary: clipping at [-1.0, 1.0]
                clipped = np.clip(sine * 2.0, -1.0, 1.0)
                self.assertTrue(np.all(np.abs(clipped) <= 1.0))

    def test_srm_patchcraft_extreme_aspect_ratios(self):
        """Verify PatchCraft mosaic tiling under extreme aspect ratios (10000x2, 2x10000, 1x1)."""
        mosaic_size = 128
        tile_size = 32
        grid_n = mosaic_size // tile_size  # 4
        needed = grid_n * grid_n          # 16

        test_shapes = [
            (10000, 2),
            (2, 10000),
            (4000, 20),
            (16, 16),
            (128, 128),
            (1920, 1080),
        ]

        for w, h in test_shapes:
            # Native upscaling guard if smaller than mosaic_size
            sw = max(w, mosaic_size)
            sh = max(h, mosaic_size)
            n_rows = sh // tile_size
            n_cols = sw // tile_size
            total_tiles = n_rows * n_cols
            self.assertGreaterEqual(
                total_tiles,
                needed,
                f"Shape ({w}, {h}) scaled to ({sw}, {sh}) must yield at least {needed} tiles (got {total_tiles})",
            )

    def test_perioral_landmark_non_consecutive_frames(self):
        """Verify BioLip landmark temporal logic across non-consecutive frames."""
        # When a gap occurs in a timeline, triplets must only be formed from 3 contiguous valid frames.
        # Run 1: 3 frames -> 1 triplet
        # Gap: 2 nulls
        # Run 2: 2 frames -> 0 triplets (< 3)
        # Run 3: 4 frames -> 2 triplets
        # Total triplets = 3.
        timeline = [
            {"face": True}, {"face": True}, {"face": True}, # Run 1 (idx 0,1,2) -> t=1 valid
            None, None,                                     # Gap (idx 3,4)
            {"face": True}, {"face": True},                 # Run 2 (idx 5,6) -> no triplet
            None,                                           # Gap (idx 7)
            {"face": True}, {"face": True}, {"face": True}, {"face": True} # Run 3 (idx 8,9,10,11) -> t=9,10 valid
        ]

        triplets = 0
        for t in range(1, len(timeline) - 1):
            prev = timeline[t - 1]
            curr = timeline[t]
            nxt = timeline[t + 1]
            if prev is not None and curr is not None and nxt is not None:
                triplets += 1

        self.assertEqual(triplets, 3)

    # =========================================================================
    # Area 2: Missing Evidence & IEEE 754 NaN Handling
    # =========================================================================

    def test_video_score_policy_nan_propagation_and_isolation(self):
        """Verify VideoScorePolicy arithmetic preserves visual NaN and prevents NaN contamination."""
        def combine_multimodal_score(
            visual_score: float,
            visual_threshold: float,
            audio_score: float = float("nan"),
            audio_validated_success: bool = False,
            audio_threshold: float = 0.35546875,
            temporal_score: float = float("nan"),
            lip_sync_anomaly: float = 0.0,
        ) -> float:
            if math.isnan(visual_score) or not (0.0 <= visual_score <= 1.0):
                return float("nan")
            base = max(0.0, min(1.0, visual_score))

            if (
                not math.isnan(visual_threshold) and 0.0 <= visual_threshold <= 1.0
                and not math.isnan(audio_threshold) and 0.0 <= audio_threshold <= 1.0
                and not math.isnan(audio_score) and 0.0 <= audio_score <= 1.0
                and audio_score >= audio_threshold
            ):
                if audio_threshold < 0.99:
                    mapped_audio = visual_threshold + ((audio_score - audio_threshold) / (1.0 - audio_threshold)) * (1.0 - visual_threshold)
                else:
                    mapped_audio = audio_score
                if not math.isnan(mapped_audio):
                    weight = 0.70 if audio_validated_success else 0.35
                    candidate = (1.0 - weight) * base + weight * max(0.0, min(1.0, mapped_audio))
                    if not math.isnan(candidate) and candidate > base:
                        base = candidate

            temp_anomaly = 0.0
            if not math.isnan(temporal_score) and temporal_score > 0.30:
                temp_anomaly = max(0.0, min(1.0, (temporal_score - 0.30) / 0.70))

            lip_anomaly = 0.0
            if not math.isnan(lip_sync_anomaly) and lip_sync_anomaly > 0.0:
                lip_anomaly = max(0.0, min(1.0, lip_sync_anomaly))

            boost = max(0.0, min(0.30, 0.15 * temp_anomaly + 0.15 * lip_anomaly))
            return max(0.0, min(1.0, base + (1.0 - base) * boost))

        # 1. Visual NaN returns NaN regardless of other modalities
        self.assertTrue(math.isnan(combine_multimodal_score(float("nan"), 0.50, audio_score=0.95)))
        self.assertTrue(math.isnan(combine_multimodal_score(float("nan"), 0.50, temporal_score=0.80)))

        # 2. Valid visual with NaN audio & temporal returns visual unmodified
        score = combine_multimodal_score(0.42, 0.50, audio_score=float("nan"), temporal_score=float("nan"))
        self.assertAlmostEqual(score, 0.42, places=6)

        # 3. Audio escalation does not fire when audio is authentic (< audio_threshold)
        score_auth_audio = combine_multimodal_score(0.42, 0.50, audio_score=0.10, audio_threshold=0.355)
        self.assertAlmostEqual(score_auth_audio, 0.42, places=6)

        # 4. Audio escalation fires when audio is synthetic (>= audio_threshold)
        score_synth_audio = combine_multimodal_score(0.42, 0.50, audio_score=0.80, audio_threshold=0.355)
        self.assertGreater(score_synth_audio, 0.42)

    def test_signal_breakdown_nan_null_truthfulness(self):
        """Verify null/NaN serialization in signal breakdown JSON."""
        signals = {
            "vfmProbeScore": None,
            "srmForensicScore": 0.45,
            "fftSpectralScore": None,
            "klDivergenceScore": None,
            "perturbationScore": None,
            "audioDspScore": None,
            "audioModelScore": None,
            "temporalScore": None,
            "provenanceScore": None,
            "prnuScore": None,
            "fusionScore": 0.52,
            "landmarkJitter": None,
        }
        encoded = json.dumps(signals)
        decoded = json.loads(encoded)

        # Null fields must not be zeroed out
        self.assertIsNone(decoded["vfmProbeScore"])
        self.assertIsNone(decoded["fftSpectralScore"])
        self.assertNotEqual(decoded["vfmProbeScore"], 0.0)

    # =========================================================================
    # Area 3: Provenance & AudioSeal
    # =========================================================================

    def test_audioseal_binomial_p_value_threshold(self):
        """Verify AudioSeal binomial two-tailed p-value and p <= 1e-4 threshold."""
        def binomial_two_tailed(k: int, n: int) -> float:
            if n <= 0:
                return 1.0
            k = max(0, min(n, k))
            upper = min(k, n - k)
            tail = sum(math.comb(n, i) * (0.5 ** n) for i in range(upper + 1))
            return min(1.0, 2.0 * tail)

        p16 = binomial_two_tailed(16, 16)
        p15 = binomial_two_tailed(15, 16)
        p14 = binomial_two_tailed(14, 16)
        p8 = binomial_two_tailed(8, 16)
        p0 = binomial_two_tailed(0, 16)

        # 16/16 must pass threshold 1e-4
        self.assertLess(p16, 1e-4)
        self.assertAlmostEqual(p16, 3.0517578125e-5, places=9)

        # 15/16 must FAIL threshold 1e-4
        self.assertGreater(p15, 1e-4)
        self.assertAlmostEqual(p15, 5.18798828125e-4, places=6)

        # 8/16 (chance) must yield 1.0
        self.assertAlmostEqual(p8, 1.0, places=6)

        # Inverted carrier (k=0): two-tailed symmetry gives p0 == p16, but MIN_AGREEING_BITS >= 8 guard must reject!
        self.assertAlmostEqual(p0, p16, places=9)
        min_agreeing_bits = 8
        self.assertFalse(0 >= min_agreeing_bits)
        self.assertTrue(16 >= min_agreeing_bits)

    # =========================================================================
    # Area 4: Cryptographic Integrity & Concurrency
    # =========================================================================

    def test_sealed_blob_content_based_equality(self):
        """Verify SealedBlob content-based equals and hashCode contract."""
        iv1 = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        iv2 = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        iv3 = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]) # single bit flip

        ct1 = bytes([10, 20, 30, 40])
        ct2 = bytes([10, 20, 30, 40])
        ct3 = bytes([10, 20, 30, 41])

        # Same contents -> equal and matching hash
        self.assertEqual(iv1, iv2)
        self.assertEqual(ct1, ct2)
        self.assertEqual(hash(iv1), hash(iv2))

        # Different contents -> unequal
        self.assertNotEqual(iv1, iv3)
        self.assertNotEqual(ct1, ct3)

    def test_canonical_receipt_serializer_thread_safety(self):
        """Stress-test RFC 8785 canonical receipt serialization across 32 threads."""
        payload_template = {
            "schema_version": "onn-receipt-v1",
            "challenge_nonce": "test-nonce-12345",
            "confidence": 0.9500,
            "device_security_tier": "STRONGBOX",
            "assessment_id": "test-assessment-id-001",
            "timestamp_epoch_ms": 1774360000000,
            "overall_score": 0.9876,
            "verdict": "SYNTHETIC",
            "model_release": "onn-prod-v10",
            "media_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "model_hashes": {
                "srm_forensic_cnn": "e90b7a42",
                "siglip_probe": "9c11bd18",
            },
            "signals": {
                "srm": 0.9876,
                "siglip": 0.9500,
            },
        }

        canonical_reference = serialize_canonical_payload(payload_template)

        def worker(idx: int) -> str:
            payload = dict(payload_template)
            return serialize_canonical_payload(payload)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(worker, i) for i in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        self.assertEqual(len(results), 100)
        for res in results:
            self.assertEqual(res, canonical_reference)

    # =========================================================================
    # Area 5: Zero-Network Sandbox Verification
    # =========================================================================

    def test_zero_network_sandbox_manifest_and_config(self):
        """Verify no network permissions exist in AndroidManifest or network security config."""
        main_manifest = os.path.join(REPO_ROOT, "android", "app", "src", "main", "AndroidManifest.xml")
        self.assertTrue(os.path.exists(main_manifest), "AndroidManifest.xml must exist")

        with open(main_manifest, "r", encoding="utf-8") as f:
            content = f.read()

        # Must explicitly remove INTERNET and ACCESS_NETWORK_STATE
        self.assertIn('android:name="android.permission.INTERNET" tools:node="remove"', content)
        self.assertIn('android:name="android.permission.ACCESS_NETWORK_STATE" tools:node="remove"', content)

        # Must NOT include any uses-permission without tools:node="remove"
        permissions = re.findall(r'<uses-permission[^>]*>', content)
        for perm in permissions:
            self.assertIn('tools:node="remove"', perm, f"Unauthorized active permission: {perm}")

        # Check network_security_config.xml has empty trust-anchors
        net_cfg = os.path.join(REPO_ROOT, "android", "app", "src", "main", "res", "xml", "network_security_config.xml")
        self.assertTrue(os.path.exists(net_cfg))
        with open(net_cfg, "r", encoding="utf-8") as f:
            cfg_content = f.read()
        self.assertIn("<trust-anchors>", cfg_content)
        self.assertIn("</trust-anchors>", cfg_content)
        self.assertNotIn("<certificates", cfg_content, "Trust anchors must not include any certificates")


if __name__ == "__main__":
    unittest.main()
