"""
thinkrouter.classifier
~~~~~~~~~~~~~~~~~~~~~~

Two interchangeable classifier backends:

  HeuristicClassifier  — zero dependencies, <1 ms, ships in the base package.
  DistilBertClassifier — fine-tuned DistilBERT, requires thinkrouter[classifier].

Both expose the same interface so the router can swap them transparently.

Usage::

    from thinkrouter.classifier import get_classifier

    clf = get_classifier("heuristic")
    result = clf.predict("What is 7 * 8?")
    # ClassifierResult(tier=NO_THINK, confidence=0.88, budget=50 tokens, latency=0.3ms)
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from .constants import Tier, TIER_TOKEN_BUDGETS


# --- Result container ---

@dataclass(frozen=True)
class ClassifierResult:
    """
    Outcome of a single difficulty classification call.

    Attributes
    ----------
    tier         : Predicted difficulty tier.
    confidence   : Classifier confidence in [0, 1] — max softmax probability.
    latency_ms   : Wall-clock classification time in milliseconds.
    backend      : Classifier backend identifier.
    token_budget : Thinking-token budget assigned by this tier (derived).
    """
    tier: Tier
    confidence: float
    latency_ms: float
    backend: str
    token_budget: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_budget", TIER_TOKEN_BUDGETS[self.tier])

    def __repr__(self) -> str:
        return (
            f"ClassifierResult("
            f"tier={self.tier.name}, "
            f"confidence={self.confidence:.3f}, "
            f"budget={self.token_budget} tokens, "
            f"latency={self.latency_ms:.1f}ms)"
        )


# --- Abstract base ---

class BaseClassifier(ABC):
    """Interface every ThinkRouter classifier backend must implement."""

    @abstractmethod
    def predict(self, query: str) -> ClassifierResult:
        """Classify difficulty of a single query string."""

    def predict_batch(self, queries: List[str]) -> List[ClassifierResult]:
        """Classify a list of queries. Override for batched GPU inference."""
        return [self.predict(q) for q in queries]


# --- Regex patterns ---

# Patterns that signal a query requires NO extended reasoning.
_NO_THINK_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*what\s+is\s+\d+[\s\+\-\*\/x×÷]\d+", re.I),
    re.compile(r"^\s*calculate\s+\d", re.I),
    re.compile(r"^\s*convert\s+\d+\s+\w+\s+to\s+\w+", re.I),
    re.compile(r"^\s*define\s+\w+", re.I),
    re.compile(
        r"^\s*what\s+(is|are)\s+(the\s+)?(capital|population|currency|flag"
        r"|language)\s+of",
        re.I,
    ),
    re.compile(r"^\s*translate\s+.{1,60}\s+to\s+\w+", re.I),
    re.compile(r"^\s*(spell|capitalise|capitalize)\s+", re.I),
    re.compile(
        r"^\s*how\s+many\s+(days|hours|minutes|seconds)\s+(in|are\s+in)\s+a",
        re.I,
    ),
    re.compile(r"^\s*what\s+(year|date|day)\s+(was|is|did)", re.I),
    re.compile(r"^\s*who\s+(wrote|invented|discovered|founded|created)\s+\w+", re.I),
]

# Patterns that signal full extended reasoning is needed.
# IMPORTANT: evaluated BEFORE word-count so short-but-hard queries
# like "Prove sqrt(2) is irrational." are not misfiled as NO_THINK.
_FULL_THINK_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bprove\b", re.I),
    re.compile(r"\bderive\b", re.I),
    re.compile(r"\boptimis[ez]\b", re.I),
    re.compile(r"\bstep[\s\-]by[\s\-]step\b", re.I),
    re.compile(r"\bcomprehensive\b", re.I),
    re.compile(r"\banalyse\b|\banalyze\b", re.I),
    re.compile(
        r"\bdesign\b.{0,40}\b(system|database|architecture|pipeline|api"
        r"|service)\b",
        re.I,
    ),
    re.compile(r"\bexplain\s+(in\s+detail|how|why|the\s+difference)\b", re.I),
    re.compile(
        r"\bwrite\s+(a\s+)?(\w+\s+)?(program|code|function|class|algorithm"
        r"|implementation|script)\b",
        re.I,
    ),
    re.compile(
        r"\b(debug|fix|refactor)\b.{0,60}\b(code|bug|error|issue|function|deadlock|race\s+condition|memory\s+leak)\b",
        re.I,
    ),
    re.compile(
        r"\bimplement\b.{0,50}\b(algorithm|data\s+structure|tree|graph|sort"
        r"|search)\b",
        re.I,
    ),
    re.compile(r"\bcompare\s+and\s+contrast\b", re.I),
    re.compile(
        r"\b(fault[\s\-]tolerant|highly\s+available|horizontally\s+scalable"
        r"|microservice)\b",
        re.I,
    ),
    re.compile(r"\bcritically\s+evaluate\b", re.I),
    re.compile(
        r"\bwhat\s+are\s+the\s+(tradeoffs?|advantages|disadvantages"
        r"|pros\s+and\s+cons)\b",
        re.I,
    ),
    re.compile(
        r"\bhow\s+does\b.{0,40}\bwork\b.{0,20}\b(in\s+detail|under\s+the\s+hood"
        r"|internally)\b",
        re.I,
    ),
]

_NO_THINK_MAX_WORDS = 12
_FULL_THINK_MIN_WORDS = 32


# --- Heuristic classifier ---

class HeuristicClassifier(BaseClassifier):
    """
    Zero-dependency rule-based difficulty classifier.

    Evaluation order (stops at the first match):
      1. NO_THINK regex patterns  — explicit simple-query signals
      2. FULL_THINK regex patterns — explicit hard-query signals (before word)
      3. Word count < 12          — short queries tend to be direct
      4. Word count >= 32         — long queries tend to need more reasoning
      5. Default                  — SHORT_THINK

    Confidence values are heuristic and intentionally conservative so the
    downstream confidence-threshold gate errs on the safe side (FULL fallback).
    """

    def predict(self, query: str) -> ClassifierResult:
        t0 = time.perf_counter()
        q = query.strip()
        wc = len(q.split())

        # 1. Explicit no-think patterns
        for pat in _NO_THINK_PATTERNS:
            if pat.search(q):
                ms = (time.perf_counter() - t0) * 1000
                return ClassifierResult(
                    tier=Tier.NO_THINK, confidence=0.88,
                    latency_ms=ms, backend="heuristic"
                )

        # 2. Explicit full-think patterns (must precede word-count check)
        hits = sum(1 for pat in _FULL_THINK_PATTERNS if pat.search(q))
        if hits >= 1:
            conf = min(0.70 + hits * 0.05, 0.95)
            ms = (time.perf_counter() - t0) * 1000
            return ClassifierResult(
                tier=Tier.FULL, confidence=conf,
                latency_ms=ms, backend="heuristic"
            )

        # 3. Short word count -> likely direct answer
        if wc <= _NO_THINK_MAX_WORDS:
            conf = max(0.55, 0.80 - (wc / _NO_THINK_MAX_WORDS) * 0.25)
            ms = (time.perf_counter() - t0) * 1000
            return ClassifierResult(
                tier=Tier.NO_THINK, confidence=conf,
                latency_ms=ms, backend="heuristic"
            )

        # 4. Long query -> likely full reasoning
        if wc >= _FULL_THINK_MIN_WORDS:
            conf = min(0.60 + (wc / 100) * 0.12, 0.88)
            ms = (time.perf_counter() - t0) * 1000
            return ClassifierResult(
                tier=Tier.FULL, confidence=conf,
                latency_ms=ms, backend="heuristic"
            )

        # 5. Default: short think
        ms = (time.perf_counter() - t0) * 1000
        return ClassifierResult(
            tier=Tier.SHORT, confidence=0.62,
            latency_ms=ms, backend="heuristic"
        )


# --- DistilBERT classifier ---

class DistilBertClassifier(BaseClassifier):
    """
    Fine-tuned DistilBERT difficulty classifier.

    Requires:  pip install thinkrouter[classifier]

    The model is loaded lazily on the first predict() call and cached for
    the lifetime of the instance. CUDA is used automatically when available.

    Parameters
    ----------
    model_name  : HuggingFace Hub model identifier.
    max_length  : Maximum WordPiece token length (128 covers 98%+ of queries).
    threshold   : Minimum softmax confidence to accept routing decision.
                  Queries below this threshold fall back to Tier.FULL.
    """

    # Default model: update after uploading to HuggingFace Hub.
    DEFAULT_MODEL = "koushikmahaan/query-difficulty-classifier"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 128,
        threshold: float = 0.75,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.threshold = threshold
        self._tokenizer = None
        self._model = None
        self._device_str = "cpu"

    def _load(self) -> None:
        try:
            import torch
            from transformers import (
                DistilBertTokenizerFast,
                DistilBertForSequenceClassification,
            )
        except ImportError as exc:
            raise ImportError(
                "DistilBertClassifier requires the [classifier] extras.\n"
                "Install with:  pip install thinkrouter[classifier]"
            ) from exc

        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device_str = str(device)

        self._tokenizer = DistilBertTokenizerFast.from_pretrained(
            self.model_name
        )
        self._model = (
            DistilBertForSequenceClassification.from_pretrained(
                self.model_name
            ).to(device)
        )
        self._model.eval()

    def predict(self, query: str) -> ClassifierResult:
        if self._model is None:
            self._load()

        import torch
        import torch.nn.functional as F

        t0 = time.perf_counter()
        enc = self._tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            probs = F.softmax(self._model(**enc).logits, dim=-1).squeeze(0)

        max_conf = float(probs.max().item())
        pred_class = int(probs.argmax().item())
        tier = Tier(pred_class) if max_conf >= self.threshold else Tier.FULL

        ms = (time.perf_counter() - t0) * 1000
        return ClassifierResult(
            tier=tier,
            confidence=max_conf,
            latency_ms=ms,
            backend=f"distilbert:{self._device_str}",
        )

    def predict_batch(self, queries: List[str]) -> List[ClassifierResult]:
        if self._model is None:
            self._load()

        import torch
        import torch.nn.functional as F

        t0 = time.perf_counter()
        enc = self._tokenizer(
            queries,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            probs = F.softmax(self._model(**enc).logits, dim=-1)

        per_ms = (time.perf_counter() - t0) * 1000 / len(queries)
        results = []
        for i in range(len(queries)):
            max_conf = float(probs[i].max().item())
            pred_class = int(probs[i].argmax().item())
            tier = (
                Tier(pred_class)
                if max_conf >= self.threshold
                else Tier.FULL
            )
            results.append(
                ClassifierResult(
                    tier=tier,
                    confidence=max_conf,
                    latency_ms=per_ms,
                    backend=f"distilbert:{self._device_str}",
                )
            )
        return results


# --- Factory ---

def get_classifier(backend: str = "heuristic", **kwargs) -> BaseClassifier:
    """
    Instantiate a classifier by backend name.

    Parameters
    ----------
    backend : "heuristic" (default) or "distilbert"
    **kwargs : passed to the classifier constructor.

    Examples
    --------
    >>> clf = get_classifier("heuristic")
    >>> clf = get_classifier("distilbert", threshold=0.80)
    """
    if backend == "heuristic":
        return HeuristicClassifier(**kwargs)
    if backend == "distilbert":
        return DistilBertClassifier(**kwargs)
    raise ValueError(
        f"Unknown backend: {backend!r}. Choose 'heuristic' or 'distilbert'."
    )
