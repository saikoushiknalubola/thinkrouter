"""
thinkrouter.classifier
~~~~~~~~~~~~~~~~~~~~~~
Two interchangeable backends with identical interfaces:

  HeuristicClassifier   — zero dependencies, <1 ms latency.
  DistilBertClassifier  — fine-tuned DistilBERT, requires thinkrouter[classifier].

Both return ClassifierResult and support single + batch prediction.
"""
from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from .constants import TIER_TOKEN_BUDGETS, Tier
from .exceptions import ClassifierError


# ── Result ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClassifierResult:
    """
    Output of a single difficulty classification call.

    Attributes
    ----------
    tier         : Predicted difficulty tier.
    confidence   : Max softmax probability in [0, 1].
    latency_ms   : Wall-clock classification time.
    backend      : Backend that produced this result.
    token_budget : Thinking-token budget for this tier (auto-set).
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


# ── Base ───────────────────────────────────────────────────────────────────

class BaseClassifier(ABC):

    @abstractmethod
    def predict(self, query: str) -> ClassifierResult:
        """Classify a single query string."""

    def predict_batch(self, queries: List[str]) -> List[ClassifierResult]:
        """Classify a list of queries. Override for true batched inference."""
        return [self.predict(q) for q in queries]


# ── Patterns ───────────────────────────────────────────────────────────────
# Two separate lists. Full-think patterns are checked BEFORE the word-count
# heuristic so short-but-hard queries ("Prove X") route correctly.

_NO_THINK: List[re.Pattern] = [
    re.compile(r"^\s*what\s+is\s+\d+[\s\+\-\*\/x×÷]\d+", re.I),
    re.compile(r"^\s*calculate\s+[\d\(]", re.I),
    re.compile(r"^\s*convert\s+\d+\s+\w+\s+to\s+\w+", re.I),
    re.compile(r"^\s*define\s+\w+", re.I),
    re.compile(r"^\s*what\s+(is|are)\s+(the\s+)?(capital|population|currency|language|flag)\s+of", re.I),
    re.compile(r"^\s*translate\s+.{1,60}\s+to\s+\w+", re.I),
    re.compile(r"^\s*(spell|capitalise|capitalize)\s+", re.I),
    re.compile(r"^\s*how\s+many\s+(days|hours|minutes|seconds)\s+(in|are\s+in)\s+a", re.I),
    re.compile(r"^\s*what\s+(year|date|day)\s+(was|is|did)\b", re.I),
    re.compile(r"^\s*who\s+(wrote|invented|discovered|founded|created|won)\b", re.I),
    re.compile(r"^\s*(yes|no|true|false)\s*\??\s*$", re.I),
    re.compile(r"^\s*is\s+\w+\s+(a|an|the)\b.{1,40}\??\s*$", re.I),
    re.compile(r"^\s*what\s+(color|colour|shape|size)\s+is\b", re.I),
    re.compile(r"^\s*(hello|hi|hey|greetings)\b", re.I),
]

_FULL_THINK: List[re.Pattern] = [
    # Mathematics & proofs
    re.compile(r"\bprove\b", re.I),
    re.compile(r"\bderive\b", re.I),
    re.compile(r"\bdemonstrate\s+(that|why|how)\b", re.I),
    re.compile(r"\bverify\s+(that|whether)\b", re.I),
    # System design
    re.compile(r"\bdesign\b.{0,50}\b(system|database|architecture|pipeline|api|service|platform)\b", re.I),
    re.compile(r"\barchitect\b.{0,50}\b(system|solution|platform|service)\b", re.I),
    re.compile(r"\b(fault[\s\-]tolerant|highly\s+available|horizontally\s+scalable|microservice)\b", re.I),
    re.compile(r"\bscalable\b.{0,30}\b(system|architecture|solution)\b", re.I),
    # Code generation
    re.compile(r"\bwrite\s+(a\s+)?(\w+\s+)?(program|code|function|class|algorithm|implementation|script|module)\b", re.I),
    re.compile(r"\bimplement\b.{0,60}\b(algorithm|data\s+structure|tree|graph|sort|search|queue|stack|heap)\b", re.I),
    re.compile(r"\bimplement\b.{0,30}\b(from\s+scratch|in\s+python|in\s+javascript|in\s+java|in\s+rust)\b", re.I),
    re.compile(r"\bcreate\s+(a\s+)?(\w+\s+)?(class|module|library|framework|api|sdk)\b", re.I),
    # Debugging & refactoring
    re.compile(
    r"\b(debug|fix|refactor)\b.{0,60}\b"
    r"(code|bug|error|issue|function|deadlock|race\s+condition|memory\s+leak|crash)\b",
    re.I
),
    re.compile(
    r"\boptimise?\b.{0,40}\b"
    r"(algorithm|code|query|performance|memory|time\s+complexity)\b",
    re.I
),
    # Explanation
    re.compile(r"\bexplain\s+(in\s+detail|how|why|the\s+difference|the\s+concept)\b", re.I),
    re.compile(r"\bwalk\s+(me\s+)?through\b", re.I),
    re.compile(r"\bhow\s+does\b.{0,50}\bwork\b.{0,20}\b(in\s+detail|under\s+the\s+hood|internally|step.by.step)\b", re.I),
    # Analysis & comparison
    re.compile(r"\bcompare\s+and\s+contrast\b", re.I),
    re.compile(r"\bwhat\s+are\s+the\s+(tradeoffs?|pros\s+and\s+cons|advantages\s+and\s+disadvantages)\b", re.I),
    re.compile(r"\bcritically\s+(evaluate|analyse|analyze|assess)\b", re.I),
    re.compile(r"\banalyse?\s+(the|this|how)\b", re.I),
    # Complexity indicators
    re.compile(r"\bstep[\s\-]by[\s\-]step\b", re.I),
    re.compile(r"\bcomprehensive\b", re.I),
    re.compile(r"\bin\s+depth\b", re.I),
    re.compile(r"\bdetailed\s+(explanation|analysis|breakdown|guide)\b", re.I),
]

_NO_THINK_MAX_WORDS  = 12
_FULL_THINK_MIN_WORDS = 32


# ── Heuristic classifier ───────────────────────────────────────────────────

class HeuristicClassifier(BaseClassifier):
    """
    Zero-dependency rule-based classifier.

    Evaluation order (stops at first match):
      1. NO_THINK regex patterns
      2. FULL_THINK regex patterns  ← before word count
      3. Short word count (<= 12)   → NO_THINK
      4. Long word count (>= 32)    → FULL
      5. Default                    → SHORT
    """

    def predict(self, query: str) -> ClassifierResult:
        t0 = time.perf_counter()
        q  = query.strip()
        wc = len(q.split())

        # 1. Explicit no-think signals
        for pat in _NO_THINK:
            if pat.search(q):
                return ClassifierResult(
                    tier=Tier.NO_THINK, confidence=0.88,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    backend="heuristic",
                )

        # 2. Explicit full-think signals (checked before word count)
        hits = sum(1 for pat in _FULL_THINK if pat.search(q))
        if hits >= 1:
            return ClassifierResult(
                tier=Tier.FULL,
                confidence=min(0.70 + hits * 0.04, 0.95),
                latency_ms=(time.perf_counter() - t0) * 1000,
                backend="heuristic",
            )

        # 3. Short query → likely direct answer
        if wc <= _NO_THINK_MAX_WORDS:
            conf = max(0.55, 0.80 - (wc / _NO_THINK_MAX_WORDS) * 0.25)
            return ClassifierResult(
                tier=Tier.NO_THINK, confidence=conf,
                latency_ms=(time.perf_counter() - t0) * 1000,
                backend="heuristic",
            )

        # 4. Long query → likely full reasoning
        if wc >= _FULL_THINK_MIN_WORDS:
            conf = min(0.60 + (wc / 100) * 0.12, 0.88)
            return ClassifierResult(
                tier=Tier.FULL, confidence=conf,
                latency_ms=(time.perf_counter() - t0) * 1000,
                backend="heuristic",
            )

        # 5. Default: moderate reasoning
        return ClassifierResult(
            tier=Tier.SHORT, confidence=0.62,
            latency_ms=(time.perf_counter() - t0) * 1000,
            backend="heuristic",
        )


# ── DistilBERT classifier ──────────────────────────────────────────────────

class DistilBertClassifier(BaseClassifier):
    """
    Fine-tuned DistilBERT classifier.
    Requires:  pip install thinkrouter[classifier]

    Parameters
    ----------
    model_name  : HuggingFace Hub model identifier.
    max_length  : Max WordPiece tokens (128 covers 98%+ of queries).
    threshold   : Min confidence to accept routing. Below → Tier.FULL (safe).
    """

    DEFAULT_MODEL = "Koushikmahaan/query-difficulty-classifier"

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
                "Install:  pip install thinkrouter[classifier]"
            ) from exc

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
                f"Failed to load '{self.model_name}' from HuggingFace Hub.\n"
                f"Run the training notebook first and upload the model.\n"
                f"Error: {exc}"
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


# ── Factory ────────────────────────────────────────────────────────────────

def get_classifier(backend: str = "heuristic", **kwargs) -> BaseClassifier:
    """
    Instantiate a classifier by name.

    Parameters
    ----------
    backend : "heuristic" (default, zero deps) or "distilbert"
    **kwargs : Forwarded to the classifier constructor.
    """
    if backend == "heuristic":
        return HeuristicClassifier(**kwargs)
    if backend == "distilbert":
        return DistilBertClassifier(**kwargs)
    raise ValueError(
        f"Unknown classifier backend: {backend!r}. Choose 'heuristic' or 'distilbert'."
    )
