"""
thinkrouter.confidence
~~~~~~~~~~~~~~~~~~~~~~
Phase 4 — Confidence Model.

Predicts hallucination risk for a (query, model) pair before inference.
High-risk queries are routed to RAG, escalation, or a safer model
rather than sent directly to a model that will likely fail.

Two backends:

  HeuristicConfidenceModel  — zero deps, instant, ships immediately.
                               Rule-based risk signals derived from
                               query structure and domain.

  AtlasConfidenceModel      — learns from accumulated Atlas quality
                               scores. Activates automatically when
                               the Atlas has enough labelled records.
                               Improves continuously with production traffic.

Usage::

    from thinkrouter.confidence import get_confidence_model, Recommendation

    model = get_confidence_model("heuristic")
    result = model.predict("What happened in last week's G7 summit?", "gpt-4o")
    print(result.risk_score)      # 0.82 — high risk
    print(result.recommendation)  # Recommendation.RAG
    print(result.reason)          # "Likely requires post-training knowledge"
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from .atlas import Atlas


# ── Recommendation enum ────────────────────────────────────────────────────

class Recommendation(str, Enum):
    PROCEED   = "proceed"    # Low risk — send to model as-is
    VERIFY    = "verify"     # Moderate risk — proceed but flag for review
    RAG       = "rag"        # High risk — augment with retrieval before sending
    ESCALATE  = "escalate"   # Very high risk — route to stronger model
    ABSTAIN   = "abstain"    # Extreme risk — do not send, return explanation


# ── Result ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConfidenceResult:
    """
    Hallucination risk prediction for a (query, model) pair.

    Attributes
    ----------
    risk_score      : Float in [0, 1]. 0 = confident, 1 = very likely to hallucinate.
    recommendation  : Routing recommendation based on risk_score.
    reason          : Human-readable explanation of the risk signal.
    signals         : List of specific risk signals that fired.
    latency_ms      : Time to produce this prediction.
    backend         : Backend that produced this result.
    """
    risk_score:     float
    recommendation: Recommendation
    reason:         str
    signals:        Tuple[str, ...]
    latency_ms:     float
    backend:        str

    @property
    def is_high_risk(self) -> bool:
        return self.risk_score >= 0.65

    @property
    def is_safe(self) -> bool:
        return self.risk_score < 0.35

    def __repr__(self) -> str:
        return (
            f"ConfidenceResult("
            f"risk={self.risk_score:.3f}, "
            f"rec={self.recommendation.value}, "
            f"backend={self.backend!r}, "
            f"latency={self.latency_ms:.2f}ms)"
        )


# ── Thresholds ─────────────────────────────────────────────────────────────

_THRESHOLDS: Dict[Recommendation, Tuple[float, float]] = {
    Recommendation.PROCEED:  (0.00, 0.35),
    Recommendation.VERIFY:   (0.35, 0.55),
    Recommendation.RAG:      (0.55, 0.75),
    Recommendation.ESCALATE: (0.75, 0.90),
    Recommendation.ABSTAIN:  (0.90, 1.01),
}

def _risk_to_recommendation(risk: float) -> Recommendation:
    for rec, (lo, hi) in _THRESHOLDS.items():
        if lo <= risk < hi:
            return rec
    return Recommendation.ABSTAIN


# ── Risk patterns ──────────────────────────────────────────────────────────

# (signal_name, pattern, risk_contribution, reason)
_TEMPORAL_RISKS: List[Tuple[str, re.Pattern, float, str]] = [
    ("recent_event",
     re.compile(r"\b(today|yesterday|this\s+week|last\s+week|this\s+month|last\s+month|"
                r"this\s+year|last\s+year|recently|just\s+announced|breaking|latest|"
                r"current\s+news|right\s+now|as\s+of\s+now)\b", re.I),
     0.65, "Likely requires post-training knowledge"),

    ("specific_date",
     re.compile(r"\b(in\s+)?(january|february|march|april|may|june|july|august|"
                r"september|october|november|december)\s+20(2[3-9]|[3-9]\d)\b", re.I),
     0.55, "References a specific recent date"),

    ("future_event",
     re.compile(r"\b(upcoming|next\s+week|next\s+month|next\s+year|will\s+happen|"
                r"going\s+to\s+happen|scheduled|announced\s+for)\b", re.I),
     0.60, "References a future or scheduled event"),
]

_SPECIFICITY_RISKS: List[Tuple[str, re.Pattern, float, str]] = [
    ("specific_person",
     re.compile(r"\b(ceo|cto|cfo|president|prime\s+minister|chancellor|senator|"
                r"director|manager)\s+of\b", re.I),
     0.50, "Asks about a specific person's role — may have changed"),

    ("specific_price",
     re.compile(r"\b(price|cost|salary|revenue|valuation|market\s+cap|"
                r"stock\s+price|share\s+price)\s+(of|for|is|was)\b", re.I),
     0.55, "Asks for a specific financial figure — changes frequently"),

    ("specific_statistic",
     re.compile(r"\b(how\s+many|what\s+percentage|what\s+number|"
                r"how\s+much\s+does|what\s+is\s+the\s+rate)\b", re.I),
     0.35, "Asks for a specific statistic that may be outdated"),

    ("named_version",
     re.compile(r"\b(version|v\d+\.\d+|release\s+\d+|update\s+\d+|"
                r"patch\s+\d+|model\s+\d+)\b", re.I),
     0.45, "References a specific software version — may be outdated"),
]

_HALLUCINATION_PRONE: List[Tuple[str, re.Pattern, float, str]] = [
    ("citation_request",
     re.compile(r"\b(cite|citation|reference|source|paper|study|research|"
                r"according\s+to|based\s+on\s+research|peer.reviewed)\b", re.I),
     0.60, "Requests citations — models frequently hallucinate references"),

    ("rare_entity",
     re.compile(r"\b(obscure|little.known|niche|rare|uncommon|"
                r"lesser.known|not\s+well.known)\b", re.I),
     0.55, "Asks about a rare or obscure entity"),

    ("medical_specific",
     re.compile(r"\b(dosage|drug\s+interaction|clinical\s+trial\s+result|"
                r"fda\s+approval|contraindication\s+for|side\s+effects\s+of\s+\w+\s+and)\b", re.I),
     0.70, "Specific medical claim — high hallucination risk"),

    ("legal_specific",
     re.compile(r"\b(case\s+law|precedent|ruling\s+in|decision\s+of|"
                r"statute\s+\d+|section\s+\d+|article\s+\d+)\b", re.I),
     0.65, "Specific legal reference — verify with authoritative source"),
]

_SAFE_SIGNALS: List[Tuple[str, re.Pattern, float]] = [
    ("conceptual",
     re.compile(r"\b(explain|what\s+is|define|how\s+does|concept\s+of|"
                r"difference\s+between|overview\s+of|introduction\s+to)\b", re.I),
     -0.20),

    ("historical_fact",
     re.compile(r"\b(in\s+the\s+\d{4}s?|historically|founded\s+in\s+1\d{3}|"
                r"born\s+in|died\s+in|ancient|medieval|19th\s+century|20th\s+century)\b", re.I),
     -0.15),

    ("mathematical",
     re.compile(r"\b(prove|theorem|calculate|derive|formula|equation)\b", re.I),
     -0.25),

    ("code_generation",
     re.compile(r"\b(write\s+(a\s+)?(function|class|program|script|algorithm)|"
                r"implement|debug|refactor)\b", re.I),
     -0.15),
]

# Models known to be weaker at certain risk types
_MODEL_RISK_MODIFIERS: Dict[str, float] = {
    "gpt-4o":            0.00,
    "gpt-4o-mini":       +0.10,
    "claude-sonnet-4-6": -0.05,
    "claude-opus-4-6":   -0.10,
    "llama3.1":          +0.08,
    "deepseek-coder-v2": +0.15,  # weaker on non-code topics
    "qwen2.5-math":      +0.20,  # weaker outside math
    "medllama2":         -0.10,  # better for medical
}


# ── Heuristic confidence model ─────────────────────────────────────────────

class HeuristicConfidenceModel:
    """
    Rule-based hallucination risk predictor.

    Zero dependencies. Activates immediately.
    Scores queries on temporal sensitivity, specificity, and
    known hallucination-prone patterns.
    """

    def predict(self, query: str, model: str = "gpt-4o") -> ConfidenceResult:
        t0     = time.perf_counter()
        q      = query.strip()
        risk   = 0.35   # base risk — models are reasonably reliable
        signals: List[str] = []
        reason = "No high-risk signals detected"

        # Positive risk signals
        for name, pat, contrib, rsn in (
            _TEMPORAL_RISKS + _SPECIFICITY_RISKS + _HALLUCINATION_PRONE
        ):
            if pat.search(q):
                risk += contrib
                signals.append(name)
                reason = rsn   # keep last (most specific)

        # Negative risk signals (reduce risk)
        for name, pat, reduction in _SAFE_SIGNALS:
            if pat.search(q):
                risk += reduction
                signals.append(f"safe:{name}")

        # Model-specific modifier
        model_key = model.split("/")[-1].split(":")[0].lower()
        for mk, mod in _MODEL_RISK_MODIFIERS.items():
            if mk in model_key or model_key in mk:
                risk += mod
                break

        # Clamp to [0, 1]
        risk = max(0.0, min(1.0, risk))
        rec  = _risk_to_recommendation(risk)

        if not signals:
            reason = "No specific risk signals — general knowledge query"

        return ConfidenceResult(
            risk_score=round(risk, 4),
            recommendation=rec,
            reason=reason,
            signals=tuple(signals),
            latency_ms=(time.perf_counter() - t0) * 1000,
            backend="heuristic",
        )

    def predict_batch(self, queries: List[str], model: str = "gpt-4o") -> List[ConfidenceResult]:
        return [self.predict(q, model) for q in queries]


# ── Atlas-backed confidence model ──────────────────────────────────────────

class AtlasConfidenceModel:
    """
    Confidence model trained on Atlas quality scores.

    Uses nearest-neighbor lookup in embedding space: finds similar
    past queries and averages their quality scores. Low average quality
    = high hallucination risk for this semantic neighbourhood.

    Requires Atlas with at least `min_atlas_size` labelled records
    (quality_score not None). Falls back to heuristic below threshold.

    Parameters
    ----------
    atlas           : Atlas instance.
    embedder        : Embedder matching the atlas.
    min_atlas_size  : Min labelled records before using atlas predictions.
    sim_threshold   : Min cosine similarity for a record to count.
    k               : Number of neighbours to average.
    """

    def __init__(
        self,
        atlas:          "Atlas",
        embedder,
        min_atlas_size: int   = 200,
        sim_threshold:  float = 0.80,
        k:              int   = 10,
    ) -> None:
        self._atlas          = atlas
        self._embedder       = embedder
        self._min_atlas_size = min_atlas_size
        self._sim_threshold  = sim_threshold
        self._k              = k
        self._heuristic      = HeuristicConfidenceModel()

    def _labelled_count(self) -> int:
        """Count records with a quality score."""
        try:
            cur = self._atlas._conn.execute(
                "SELECT COUNT(*) FROM records WHERE quality_score IS NOT NULL"
            )
            return cur.fetchone()[0]
        except Exception:
            return 0

    def predict(self, query: str, model: str = "gpt-4o") -> ConfidenceResult:
        t0 = time.perf_counter()

        if self._labelled_count() < self._min_atlas_size:
            # Not enough data — fall back to heuristic
            result = self._heuristic.predict(query, model)
            return ConfidenceResult(
                risk_score=result.risk_score,
                recommendation=result.recommendation,
                reason=f"[atlas-fallback] {result.reason}",
                signals=result.signals,
                latency_ms=(time.perf_counter() - t0) * 1000,
                backend="atlas-fallback",
            )

        vec     = self._embedder.embed(query)
        similar = self._atlas.find_similar(vec, k=self._k, min_score=self._sim_threshold)
        labelled = [
            r for r in similar
            if r.record.quality_score is not None
        ]

        if not labelled:
            result = self._heuristic.predict(query, model)
            return ConfidenceResult(
                risk_score=result.risk_score,
                recommendation=result.recommendation,
                reason="No labelled neighbours — using heuristic",
                signals=result.signals,
                latency_ms=(time.perf_counter() - t0) * 1000,
                backend="atlas-no-neighbours",
            )

        # Average quality of nearest labelled neighbours
        # Weighted by similarity
        total_weight = sum(r.similarity for r in labelled)
        avg_quality  = sum(
            r.record.quality_score * r.similarity for r in labelled
        ) / total_weight

        # risk = 1 - quality (quality 0.9 → risk 0.1)
        risk        = round(max(0.0, min(1.0, 1.0 - avg_quality)), 4)
        rec         = _risk_to_recommendation(risk)
        n_labelled  = len(labelled)
        avg_sim     = total_weight / n_labelled

        return ConfidenceResult(
            risk_score=risk,
            recommendation=rec,
            reason=f"Atlas: {n_labelled} similar labelled queries, avg quality {avg_quality:.2f}",
            signals=(f"atlas_neighbours:{n_labelled}", f"avg_sim:{avg_sim:.3f}"),
            latency_ms=(time.perf_counter() - t0) * 1000,
            backend="atlas",
        )

    def predict_batch(self, queries: List[str], model: str = "gpt-4o") -> List[ConfidenceResult]:
        return [self.predict(q, model) for q in queries]


# ── Factory ────────────────────────────────────────────────────────────────

def get_confidence_model(backend: str = "heuristic", **kwargs) -> HeuristicConfidenceModel:
    if backend == "heuristic":
        return HeuristicConfidenceModel(**kwargs)
    if backend == "atlas":
        return AtlasConfidenceModel(**kwargs)
    raise ValueError(
        f"Unknown confidence backend: {backend!r}. Choose 'heuristic' or 'atlas'."
    )
