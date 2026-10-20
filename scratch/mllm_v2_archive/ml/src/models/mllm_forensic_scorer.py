"""
MLLM-based forensic scorer for AI-generated media detection.

Uses a multimodal large language model (Gemma 3n / EmbeddingGemma-2)
to analyze images for signs of AI generation through semantic reasoning.

The MLLM provides:
  1. A calibrated probability score P(ai_generated) in [0, 1]
  2. Natural-language explanation of detected anomalies
  3. Self-confidence assessment (low/medium/high)
  4. List of specific anomalies detected

This scorer is designed to complement (not replace) the existing
SRM CNN and LCNN-MFM models by adding semantic reasoning that
pixel-level artifact detectors cannot provide.

Architecture:
    Image → MLLM (Gemma 3n E2B) → structured JSON response
         → parse P(ai) from token logprobs
         → calibrate via Platt scaling
         → return MllmScore dataclass

Research basis:
    - MoA-DF (ACM MM 2025): Multi-MLLM ensemble achieves SOTA
    - FakeVLM (NeurIPS 2025): MLLM explanations improve trust
    - Veritas (2025): Pattern-aware reasoning improves OOD
    - Cactus Hybrid (2025): Confidence probe from hidden states
    - DFBench: 540K image benchmark for deepfake detection
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class MllmScore:
    """Structured output from MLLM forensic analysis.

    Attributes:
        probability: Calibrated P(ai_generated) in [0, 1]
        raw_probability: Uncalibrated probability from MLLM
        confidence: Model self-confidence ("low", "medium", "high")
        anomalies: List of detected semantic anomalies
        reasoning: Natural-language explanation
        latency_ms: Inference latency in milliseconds
        model_name: Identifier for the MLLM variant used
        num_tokens: Number of tokens generated
        calibrated: Whether Platt scaling was applied
    """
    probability: float
    raw_probability: float
    confidence: str = "medium"
    anomalies: list[str] = field(default_factory=list)
    reasoning: str = ""
    latency_ms: float = 0.0
    model_name: str = ""
    num_tokens: int = 0
    calibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": self.probability,
            "raw_probability": self.raw_probability,
            "confidence": self.confidence,
            "anomalies": self.anomalies,
            "reasoning": self.reasoning,
            "latency_ms": self.latency_ms,
            "model_name": self.model_name,
            "num_tokens": self.num_tokens,
            "calibrated": self.calibrated,
        }


# ──────────────────────────────────────────────────────────────────────
# Prompt Engineering
# ──────────────────────────────────────────────────────────────────────

FORENSIC_IMAGE_PROMPT = """You are a forensic analyst specializing in AI-generated image detection.
Analyze the provided image for signs of artificial generation.

Examine these specific aspects:
1. ANATOMY: Check for impossible proportions, extra fingers, asymmetric features, unnatural body geometry
2. PHYSICS: Look for impossible lighting, incorrect shadows, violations of gravity, impossible reflections
3. TEXTURE: Identify repetitive patterns, unnatural smoothness, over-sharpened details, plastic skin
4. SEMANTICS: Assess contextual coherence, impossible scenarios, style inconsistencies, anachronisms
5. ARTIFACTS: Detect generative noise patterns, upscaling traces, color banding, edge discontinuities

Rate each aspect from 0.0 (natural) to 1.0 (clearly AI-generated).
Then provide your final assessment.

Respond in exactly this JSON format (no other text):
{
  "ai_probability": <float, 0.0 to 1.0>,
  "confidence": "<low|medium|high>",
  "aspect_scores": {
    "anatomy": <float>,
    "physics": <float>,
    "texture": <float>,
    "semantics": <float>,
    "artifacts": <float>
  },
  "anomalies": ["<specific anomaly 1>", "<specific anomaly 2>"],
  "reasoning": "<brief 1-2 sentence explanation>"
}"""

FORENSIC_AUDIO_PROMPT = """You are a forensic analyst specializing in AI-generated audio/speech detection.
The following is a transcription and spectral description of an audio clip.

Analyze for signs of synthetic speech:
1. PROSODY: Unnatural rhythm, monotone delivery, incorrect stress patterns
2. PHONETIC: Distorted consonants, unnatural formants, robotic timbre
3. CONTEXT: Semantic incoherence, irrelevant responses, canned phrases
4. ARTIFACTS: Digital noise, clipping, encoding anomalies, spectral holes

Audio features: {audio_features}

Respond in exactly this JSON format (no other text):
{{
  "ai_probability": <float, 0.0 to 1.0>,
  "confidence": "<low|medium|high>",
  "anomalies": ["<specific anomaly 1>", "<specific anomaly 2>"],
  "reasoning": "<brief 1-2 sentence explanation>"
}}"""

FORENSIC_VIDEO_PROMPT = """You are a forensic analyst specializing in AI-generated video detection.
Analyze the provided video frames for signs of artificial generation.

Examine these aspects:
1. TEMPORAL CONSISTENCY: Flickering, frame-to-frame inconsistency, unnatural motion
2. FACE STABILITY: Landmark jitter, identity drift, expression discontinuities
3. LIP SYNC: Audio-visual misalignment, unnatural mouth movements
4. PHYSICS: Impossible motion, gravity violations, incorrect momentum

Respond in exactly this JSON format (no other text):
{{
  "ai_probability": <float, 0.0 to 1.0>,
  "confidence": "<low|medium|high>",
  "aspect_scores": {{
    "temporal_consistency": <float>,
    "face_stability": <float>,
    "lip_sync": <float>,
    "physics": <float>
  }},
  "anomalies": ["<specific anomaly 1>", "<specific anomaly 2>"],
  "reasoning": "<brief 1-2 sentence explanation>"
}}"""


# ──────────────────────────────────────────────────────────────────────
# JSON Response Parser
# ──────────────────────────────────────────────────────────────────────

def parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON from MLLM response, handling common formatting issues."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in text
    json_match = re.search(r'\{[^{}]*"ai_probability"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: try to extract probability with regex
    prob_match = re.search(r'"ai_probability"\s*:\s*([0-9.]+)', text)
    if prob_match:
        return {"ai_probability": float(prob_match.group(1)), "confidence": "low"}

    raise ValueError(f"Could not parse MLLM response: {text[:200]}")


# ──────────────────────────────────────────────────────────────────────
# Platt Scaling Calibration
# ──────────────────────────────────────────────────────────────────────

class PlattScaler:
    """Platt scaling for calibrating MLLM probability outputs.

    Fits a logistic regression on the MLLM's raw scores to map them
    to calibrated probabilities. This is critical because MLLMs
    are typically overconfident in their assessments.

    Based on Platt (1999) and Guo et al. (2017) temperature scaling.
    """

    def __init__(self):
        self.a: float = 1.0  # slope
        self.b: float = 0.0  # intercept
        self.fitted: bool = False

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> "PlattScaler":
        """Fit Platt scaling parameters.

        Args:
            raw_scores: Raw MLLM probability outputs, shape (N,)
            labels: Ground truth labels (0=real, 1=AI), shape (N,)
        """
        from sklearn.linear_model import LogisticRegression

        # Use logit-transformed scores as feature
        logits = np.log(np.clip(raw_scores, 1e-7, 1 - 1e-7) /
                        (1 - np.clip(raw_scores, 1e-7, 1 - 1e-7)))
        X = logits.reshape(-1, 1)

        lr = LogisticRegression(C=1e10, solver="lbfgs")
        lr.fit(X, labels)

        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
        self.fitted = True
        return self

    def transform(self, raw_score: float) -> float:
        """Apply Platt scaling to a raw score."""
        if not self.fitted:
            return raw_score

        epsilon = 1e-7
        clipped = np.clip(raw_score, epsilon, 1 - epsilon)
        logit = math.log(clipped / (1 - clipped))
        z = self.a * logit + self.b
        return float(1.0 / (1.0 + math.exp(-z)))

    def to_dict(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "fitted": self.fitted}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlattScaler":
        scaler = cls()
        scaler.a = float(d["a"])
        scaler.b = float(d["b"])
        scaler.fitted = bool(d.get("fitted", True))
        return scaler


# ──────────────────────────────────────────────────────────────────────
# MLLM Forensic Scorer
# ──────────────────────────────────────────────────────────────────────

class MllmForensicScorer:
    """MLLM-based forensic scorer using Gemma 3n or similar models.

    This scorer uses a multimodal LLM to analyze images for signs of
    AI generation through semantic reasoning. It complements the
    existing SRM CNN (pixel artifacts) and LCNN-MFM (audio artifacts)
    by detecting semantic inconsistencies.

    Usage:
        scorer = MllmForensicScorer.from_pretrained("google/gemma-3n-E2B-it")
        score = scorer.analyze_image(image_path)
        print(f"P(AI): {score.probability:.3f}")
        print(f"Reasoning: {score.reasoning}")
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cpu",
        model_name: str = "gemma-3n-e2b",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        platt_scaler: Optional[PlattScaler] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.platt_scaler = platt_scaler or PlattScaler()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "google/gemma-3n-E2B-it",
        device: str = "cpu",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        **kwargs,
    ) -> "MllmForensicScorer":
        """Load a pretrained MLLM for forensic scoring.

        Args:
            model_id: HuggingFace model identifier
            device: Device to run on ("cpu", "cuda", "auto")
            dtype: Model dtype ("float32", "bfloat16", "float16")
            load_in_4bit: Use 4-bit quantization for memory efficiency
        """
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": device if device != "auto" else "auto",
        }

        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        model = AutoModelForImageTextToText.from_pretrained(
            model_id, **model_kwargs
        )
        model.eval()

        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_name=model_id,
            **kwargs,
        )

    def analyze_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
    ) -> MllmScore:
        """Analyze an image for AI generation signs.

        Args:
            image: PIL Image to analyze
            prompt: Custom prompt (uses default forensic prompt if None)

        Returns:
            MllmScore with calibrated probability and explanation
        """
        import time

        start_time = time.time()
        prompt = prompt or FORENSIC_IMAGE_PROMPT

        # Build conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True
        ).to(self.device)

        # Generate with logprobs for confidence extraction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy decoding for deterministic results
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=False,
            )

        # Extract generated text
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        # Extract logprobs for confidence scoring
        token_logprobs = self._extract_token_logprobs(outputs)

        # Parse response
        try:
            parsed = parse_json_response(generated_text)
            raw_prob = float(parsed.get("ai_probability", 0.5))
            confidence = parsed.get("confidence", "medium").lower()
            anomalies = parsed.get("anomalies", [])
            reasoning = parsed.get("reasoning", "")
        except ValueError:
            # Fallback: if parsing fails, use neutral score
            raw_prob = 0.5
            confidence = "low"
            anomalies = []
            reasoning = "Failed to parse model response"

        # Apply calibration
        calibrated_prob = self.platt_scaler.transform(raw_prob)

        elapsed_ms = (time.time() - start_time) * 1000

        return MllmScore(
            probability=calibrated_prob,
            raw_probability=raw_prob,
            confidence=confidence,
            anomalies=anomalies,
            reasoning=reasoning,
            latency_ms=elapsed_ms,
            model_name=self.model_name,
            num_tokens=len(generated_ids),
            calibrated=self.platt_scaler.fitted,
        )

    def _extract_token_logprobs(self, outputs: Any) -> list[float]:
        """Extract per-token logprobs from generation output."""
        if not hasattr(outputs, "scores") or outputs.scores is None:
            return []

        logprobs = []
        for score_tensor in outputs.scores:
            # score_tensor shape: (batch_size, vocab_size)
            log_probs = torch.log_softmax(score_tensor[0], dim=-1)
            # Get the logprob of the chosen token
            # Note: we need the actual token ids
            if hasattr(outputs, "sequences"):
                token_idx = len(logprobs)
                if token_idx < outputs.sequences.shape[1]:
                    token_id = outputs.sequences[0, token_idx].item()
                    logprobs.append(log_probs[token_id].item())
        return logprobs

    def calibrate(
        self,
        image_paths: list[Path],
        labels: np.ndarray,
        output_path: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Calibrate the MLLM scorer on a labeled dataset.

        Runs the MLLM on all images, collects raw scores, and fits
        Platt scaling to produce calibrated probabilities.

        Args:
            image_paths: List of image file paths
            labels: Ground truth (0=real, 1=AI)
            output_path: Where to save calibration parameters

        Returns:
            Calibration metrics dict
        """
        raw_scores = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            score = self.analyze_image(img)
            raw_scores.append(score.raw_probability)

        raw_scores = np.array(raw_scores)

        # Fit Platt scaling
        self.platt_scaler.fit(raw_scores, labels)

        # Compute calibration metrics
        calibrated = np.array([self.platt_scaler.transform(s) for s in raw_scores])

        metrics = {
            "n_samples": len(labels),
            "raw_auc": self._compute_auc(raw_scores, labels),
            "calibrated_auc": self._compute_auc(calibrated, labels),
            "raw_brier": float(np.mean((raw_scores - labels) ** 2)),
            "calibrated_brier": float(np.mean((calibrated - labels) ** 2)),
            "platt_a": self.platt_scaler.a,
            "platt_b": self.platt_scaler.b,
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump({
                    "metrics": metrics,
                    "platt_scaler": self.platt_scaler.to_dict(),
                    "model_name": self.model_name,
                }, f, indent=2)

        return metrics

    @staticmethod
    def _compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute ROC AUC."""
        from sklearn.metrics import roc_auc_score
        try:
            return float(roc_auc_score(labels, scores))
        except ValueError:
            return 0.5


# ──────────────────────────────────────────────────────────────────────
# Batch Processing for Training Pipeline
# ──────────────────────────────────────────────────────────────────────

def score_dataset_with_mllm(
    image_paths: list[Path],
    scorer: MllmForensicScorer,
    output_path: Optional[Path] = None,
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    """Score a dataset with MLLM for fusion training.

    This is the entry point for the training pipeline — it processes
    all images in a dataset and returns structured scores that can
    be used as additional features in the fusion meta-classifier.

    Args:
        image_paths: List of image file paths
        scorer: Initialized MllmForensicScorer
        output_path: Optional path to save results incrementally
        batch_size: Number of images to process before saving

    Returns:
        List of score dicts (one per image)
    """
    results = []

    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            score = scorer.analyze_image(img)
            results.append({
                "path": str(img_path),
                **score.to_dict(),
            })
        except Exception as e:
            # On failure, return neutral score (0.5) with low confidence
            results.append({
                "path": str(img_path),
                "probability": 0.5,
                "raw_probability": 0.5,
                "confidence": "low",
                "anomalies": [],
                "reasoning": f"Error: {str(e)}",
                "latency_ms": 0.0,
                "model_name": scorer.model_name,
                "num_tokens": 0,
                "calibrated": False,
            })

        # Incremental save
        if output_path and (i + 1) % batch_size == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results
