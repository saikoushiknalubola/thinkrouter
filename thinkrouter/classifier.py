"""
thinkrouter.classifier
~~~~~~~~~~~~~~~~~~~~~~
Two interchangeable backends:

  HeuristicClassifier  — zero dependencies, <1ms.
  DistilBertClassifier — fine-tuned DistilBERT, requires thinkrouter[classifier].
"""
from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from .constants import TIER_TOKEN_BUDGETS, Tier
from .exceptions import ClassifierError


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClassifierResult:
    """
    Outcome of a single difficulty classification call.

    Attributes
    ----------
    tier         : Predicted difficulty tier.
    confidence   : Max softmax probability in [0, 1].
    latency_ms   : Wall-clock time in milliseconds.
    backend      : Classifier backend identifier.
    token_budget : Thinking-token budget for this tier (auto-derived).
    """
    tier:         Tier
    confidence:   float
    latency_ms:   float
    backend:      str
    token_budget: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_budget", TIER_TOKEN_BUDGETS[self.tier])

    def __repr__(self) -> str:
        return (
            f"ClassifierResult("
            f"tier={self.tier.name}, "
            f"confidence={self.confidence:.3f}, "
            f"budget={self.token_budget} tokens, "
            f"latency={self.latency_ms:.2f}ms)"
        )


# ── Base ──────────────────────────────────────────────────────────────────────

class BaseClassifier(ABC):

    @abstractmethod
    def predict(self, query: str) -> ClassifierResult:
        """Classify difficulty of a single query string."""

    def predict_batch(self, queries: List[str]) -> List[ClassifierResult]:
        """Classify a list of queries. Override for batched GPU inference."""
        return [self.predict(q) for q in queries]


# ── Patterns ──────────────────────────────────────────────────────────────────

_NO_THINK: List[re.Pattern] = [
    re.compile(r"^\s*what\s+is\s+\d+[\s\+\-\*\/x×÷]\d+", re.I),
    re.compile(r"^\s*calculate\s+\d", re.I),
    re.compile(r"^\s*convert\s+\d+\s+\w+\s+to\s+\w+", re.I),
    re.compile(r"^\s*define\s+\w+", re.I),
    re.compile(r"^\s*what\s+(is|are)\s+(the\s+)?(capital|population|currency|flag|language)\s+of", re.I),
    re.compile(r"^\s*translate\s+.{1,60}\s+to\s+\w+", re.I),
    re.compile(r"^\s*(spell|capitalise|capitalize)\s+", re.I),
    re.compile(r"^\s*how\s+many\s+(days|hours|minutes|seconds)\s+(in|are\s+in)\s+a", re.I),
    re.compile(r"^\s*what\s+(year|date|day)\s+(was|is|did)", re.I),
    re.compile(r"^\s*who\s+(wrote|invented|discovered|founded|created)\s+\w+", re.I),
    re.compile(r"^\s*(yes|no|true|false)\s*\??\s*$", re.I),
]

# Checked BEFORE word-count so "Prove X" (5 words) routes correctly.
_FULL_THINK: List[re.Pattern] = [
    re.compile(r"\bprove\b", re.I),
    re.compile(r"\bderive\b", re.I),
    re.compile(r"\boptimis[ez]\b", re.I),
    re.compile(r"\bstep[\s\-]by[\s\-]step\b", re.I),
    re.compile(r"\bcomprehensive\b", re.I),
    re.compile(r"\banalyse\b|\banalyze\b", re.I),
    re.compile(r"\bdesign\b.{0,40}\b(system|database|architecture|pipeline|api|service)\b", re.I),
    re.compile(r"\bexplain\s+(in\s+detail|how|why|the\s+difference)\b", re.I),
    re.compile(r"\bwrite\s+(a\s+)?(\w+\s+)?(program|code|function|class|algorithm|implementation|script)\b", re.I),
    re.compile(r"\b(debug|fix|refactor)\b.{0,60}\b(code|bug|error|issue|function|deadlock|race\s+condition|memory\s+leak)\b", re.I),
    re.compile(r"\bimplement\b.{0,50}\b(algorithm|data\s+structure|tree|graph|sort|search)\b", re.I),
    re.compile(r"\bcompare\s+and\s+contrast\b", re.I),
    re.compile(r"\b(fault[\s\-]tolerant|highly\s+available|horizontally\s+scalable|microservice)\b", re.I),
    re.compile(r"\bcritically\s+(evaluate|analyse|analyze)\b", re.I),
    re.compile(r"\bwhat\s+are\s+the\s+(tradeoffs?|advantages|disadvantages|pros\s+and\s+cons)\b", re.I),
    re.compile(r"\bhow\s+does\b.{0,40}\bwork\b.{0,20}\b(in\s+detail|under\s+the\s+hood|internally)\b", re.I),
    re.compile(r"\brefactor\b.{0,50}\b(class|module|function|code|script)\b", re.I),
]

_NO_THINK_MAX   = 12
_FULL_THINK_MIN = 32


# ── Heuristic ─────────────────────────────────────────────────────────────────

class HeuristicClassifier(BaseClassifier):
    """
    Zero-dependency rule-based classifier.

    Evaluation order (stops at first match):
      1. NO_THINK patterns
      2. FULL_THINK patterns  ← before word count
      3. Short word count (<= 12)
      4. Long word count (>= 32)
      5. Default: SHORT
    """

    def predict(self, query: str) -> ClassifierResult:
        t0 = time.perf_counter()
        q  = query.strip()
        wc = len(q.split())

        for pat in _NO_THINK:
            if pat.search(q):
                return ClassifierResult(
                    tier=Tier.NO_THINK, confidence=0.88,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    backend="heuristic",
                )

        hits = sum(1 for pat in _FULL_THINK if pat.search(q))
        if hits >= 1:
            return ClassifierResult(
                tier=Tier.FULL,
                confidence=min(0.70 + hits * 0.05, 0.95),
                latency_ms=(time.perf_counter() - t0) * 1000,
                backend="heuristic",
            )

        if wc <= _NO_THINK_MAX:
            return ClassifierResult(
                tier=Tier.NO_THINK,
                confidence=max(0.55, 0.80 - (wc / _NO_THINK_MAX) * 0.25),
                latency_ms=(time.perf_counter() - t0) * 1000,
                backend="heuristic",
            )

        if wc >= _FULL_THINK_MIN:
            return ClassifierResult(
                tier=Tier.FULL,
                confidence=min(0.60 + (wc / 100) * 0.12, 0.88),
                latency_ms=(time.perf_counter() - t0) * 1000,
                backend="heuristic",
            )

        return ClassifierResult(
            tier=Tier.SHORT, confidence=0.62,
            latency_ms=(time.perf_counter() - t0) * 1000,
            backend="heuristic",
        )


# ── DistilBERT ────────────────────────────────────────────────────────────────

class DistilBertClassifier(BaseClassifier):
    """
    Fine-tuned DistilBERT classifier.
    Requires:  pip install thinkrouter[classifier]

    Parameters
    ----------
    model_name  : HuggingFace Hub model id.
    max_length  : Max WordPiece tokens (128 covers 98%+ of queries).
    threshold   : Min confidence to accept routing. Below → Tier.FULL.
    """

    DEFAULT_MODEL = "koushikmahaan/query-difficulty-classifier"

    def __init__(
        self,
        model_name: str   = DEFAULT_MODEL,
        max_length: int   = 128,
        threshold:  float = 0.75,
    ) -> None:
        self.model_name  = model_name
        self.max_length  = max_length
        self.threshold   = threshold
        self._tokenizer  = None
        self._model      = None
        self._device_str = "cpu"

    def _load(self) -> None:
        try:
            import torch
            from transformers import (
                DistilBertForSequenceClassification,
                DistilBertTokenizerFast,
            )
        except ImportError as exc:
            raise ClassifierError(
                "DistilBertClassifier requires the [classifier] extras.\n"
                "Install with:  pip install thinkrouter[classifier]"
            ) from exc

        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device_str = str(device)
        try:
            self._tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
            self._model = (
                DistilBertForSequenceClassification
                .from_pretrained(self.model_name)
                .to(device)
            )
            self._model.eval()
        except Exception as exc:
            raise ClassifierError(
                f"Failed to load model '{self.model_name}' from HuggingFace Hub.\n"
                f"Make sure you have run the training notebook and uploaded the model.\n"
                f"Original error: {exc}"
            ) from exc

    def predict(self, query: str) -> ClassifierResult:
        if self._model is None:
            self._load()

        import torch
        import torch.nn.functional as F

        t0  = time.perf_counter()
        enc = self._tokenizer(
            query, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        enc    = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            probs = F.softmax(self._model(**enc).logits, dim=-1).squeeze(0)

        max_conf   = float(probs.max().item())
        pred_class = int(probs.argmax().item())
        tier       = Tier(pred_class) if max_conf >= self.threshold else Tier.FULL

        return ClassifierResult(
            tier=tier, confidence=max_conf,
            latency_ms=(time.perf_counter() - t0) * 1000,
            backend=f"distilbert:{self._device_str}",
        )

    def predict_batch(self, queries: List[str]) -> List[ClassifierResult]:
        if self._model is None:
            self._load()

        import torch
        import torch.nn.functional as F

        t0  = time.perf_counter()
        enc = self._tokenizer(
            queries, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        enc    = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            probs = F.softmax(self._model(**enc).logits, dim=-1)

        per_ms = (time.perf_counter() - t0) * 1000 / max(len(queries), 1)
        results = []
        for i in range(len(queries)):
            max_conf   = float(probs[i].max().item())
            pred_class = int(probs[i].argmax().item())
            tier       = Tier(pred_class) if max_conf >= self.threshold else Tier.FULL
            results.append(ClassifierResult(
                tier=tier, confidence=max_conf,
                latency_ms=per_ms,
                backend=f"distilbert:{self._device_str}",
            ))
        return results


# ── Factory ───────────────────────────────────────────────────────────────────

def get_classifier(backend: str = "heuristic", **kwargs) -> BaseClassifier:
    """
    Instantiate a classifier by backend name.

    Parameters
    ----------
    backend : "heuristic" (default) or "distilbert"
    """
    if backend == "heuristic":
        return HeuristicClassifier(**kwargs)
    if backend == "distilbert":
        return DistilBertClassifier(**kwargs)
    raise ValueError(
        f"Unknown backend: {backend!r}. Choose 'heuristic' or 'distilbert'."
    )
