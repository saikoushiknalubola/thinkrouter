"""
thinkrouter.cost
~~~~~~~~~~~~~~~~
Real-time cost tracker — v0.7.0.

Tracks actual API spend per routing decision with per-model pricing.
Shows exact ROI vs a GPT-4o-only baseline so you can quantify savings
in dollars, not just percentages.

Usage::

    from thinkrouter import ThinkRouter

    client = ThinkRouter(provider="openai", cost_tracking=True)
    r = client.chat("Write a binary search tree in Python.")

    print(r.cost_usd)                  # 0.000034  (deepseek-coder via Ollama = free)
    client.cost_tracker.print_summary()

    # Dashboard
    # ─────────────────────────────────────────────
    # Total spend           : $0.0023
    # Baseline (GPT-4o all) : $0.0187
    # Saved                 : $0.0164  (87.7%)
    # Most expensive domain : LEGAL
    # Cheapest domain       : CODE  (Ollama, free)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .constants import Tier
from .domain import Domain


# ── Per-model pricing (USD per 1M tokens, input / output) ─────────────────
# Sources: official pricing pages as of 2025-Q2
# Ollama local models = zero cost

MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o":               (2.50,  10.00),
    "gpt-4o-mini":          (0.15,   0.60),
    "gpt-4-turbo":          (10.00,  30.00),
    "o1":                   (15.00,  60.00),
    "o1-mini":              (3.00,  12.00),
    "o3":                   (10.00,  40.00),
    "o3-mini":              (1.10,   4.40),
    "o4-mini":              (1.10,   4.40),
    # Anthropic
    "claude-opus-4-6":      (15.00,  75.00),
    "claude-sonnet-4-6":    (3.00,  15.00),
    "claude-haiku-4-5":     (0.80,   4.00),
    # Ollama local — free
    "deepseek-coder-v2":    (0.00,   0.00),
    "qwen2.5-math":         (0.00,   0.00),
    "llama3.1":             (0.00,   0.00),
    "medllama2":            (0.00,   0.00),
    "codellama:13b":        (0.00,   0.00),
}

_BASELINE_MODEL = "gpt-4o"


def get_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a single API call."""
    key = model.split("/")[-1].split(":")[0].lower()
    pricing = None
    for mk, p in MODEL_PRICING.items():
        if mk in key or key in mk:
            pricing = p
            break
    if pricing is None:
        pricing = MODEL_PRICING[_BASELINE_MODEL]  # conservative fallback
    input_cost  = (input_tokens  / 1_000_000) * pricing[0]
    output_cost = (output_tokens / 1_000_000) * pricing[1]
    return round(input_cost + output_cost, 8)


def get_baseline_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """What this call would have cost on GPT-4o."""
    p = MODEL_PRICING[_BASELINE_MODEL]
    return round(
        (input_tokens  / 1_000_000) * p[0] +
        (output_tokens / 1_000_000) * p[1],
        8,
    )


# ── Record ─────────────────────────────────────────────────────────────────

@dataclass
class CostRecord:
    """A single cost-tracked routing event."""
    model:          str
    provider:       str
    domain:         Domain
    tier:           Tier
    input_tokens:   int
    output_tokens:  int
    cost_usd:       float
    baseline_usd:   float
    saved_usd:      float
    timestamp:      datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ── Summary ────────────────────────────────────────────────────────────────

@dataclass
class CostSummary:
    """Aggregate cost statistics."""
    total_calls:      int
    total_cost_usd:   float
    baseline_usd:     float
    saved_usd:        float
    savings_pct:      float
    cost_by_domain:   Dict[str, float]
    cost_by_model:    Dict[str, float]
    calls_by_domain:  Dict[str, int]
    free_calls:       int       # calls routed to Ollama (zero cost)
    since:            Optional[datetime]

    def __str__(self) -> str:
        lines = [
            "",
            "  ThinkRouter — Cost Dashboard",
            "  " + "─" * 48,
            f"  Total calls           : {self.total_calls:,}",
            f"  Actual spend          : ${self.total_cost_usd:.6f}",
            f"  GPT-4o baseline       : ${self.baseline_usd:.6f}",
            f"  Saved                 : ${self.saved_usd:.6f}  ({self.savings_pct:.1f}%)",
            f"  Free calls (Ollama)   : {self.free_calls:,}  ({self.free_calls/max(self.total_calls,1)*100:.1f}%)",
            "",
            "  Cost by domain:",
        ]
        for domain, cost in sorted(self.cost_by_domain.items(), key=lambda x: -x[1]):
            calls = self.calls_by_domain.get(domain, 0)
            lines.append(f"    {domain:<12} : ${cost:.6f}  ({calls} calls)")
        lines.append("")
        lines.append("  Cost by model:")
        for model, cost in sorted(self.cost_by_model.items(), key=lambda x: -x[1]):
            lines.append(f"    {model:<32} : ${cost:.6f}")
        if self.since:
            lines.append(
                f"\n  Tracking since : {self.since.strftime('%Y-%m-%d %H:%M UTC')}"
            )
        lines.append("")
        return "\n".join(lines)

    def daily_projection(self, calls_per_day: int = 10_000) -> str:
        if self.total_calls == 0:
            return "No data yet."
        avg_actual   = self.total_cost_usd  / self.total_calls
        avg_baseline = self.baseline_usd / self.total_calls
        proj_actual  = avg_actual   * calls_per_day
        proj_base    = avg_baseline * calls_per_day
        proj_saved   = proj_base - proj_actual
        return (
            f"  At {calls_per_day:,} calls/day:\n"
            f"    Actual spend    : ${proj_actual * 30:,.2f}/month\n"
            f"    Baseline spend  : ${proj_base   * 30:,.2f}/month\n"
            f"    Monthly savings : ${proj_saved  * 30:,.2f}/month"
        )


# ── Tracker ────────────────────────────────────────────────────────────────

class CostTracker:
    """
    Thread-safe cost accumulator.

    Tracks every routing decision with its actual cost vs
    what GPT-4o would have charged for the same call.

    Parameters
    ----------
    max_records : int
        Maximum CostRecord objects retained. 0 = aggregates only.
    """

    def __init__(self, max_records: int = 10_000) -> None:
        self._lock         = threading.Lock()
        self._records:     List[CostRecord] = []
        self._max          = max_records
        self._n            = 0
        self._total_cost   = 0.0
        self._total_base   = 0.0
        self._total_saved  = 0.0
        self._free_calls   = 0
        self._domain_cost: Dict[str, float] = {}
        self._domain_calls: Dict[str, int]  = {}
        self._model_cost:  Dict[str, float] = {}
        self._since:       Optional[datetime] = None

    def record(
        self,
        model:         str,
        provider:      str,
        domain:        Domain,
        tier:          Tier,
        input_tokens:  int,
        output_tokens: int,
    ) -> CostRecord:
        cost     = get_cost_usd(model, input_tokens, output_tokens)
        baseline = get_baseline_cost_usd(input_tokens, output_tokens)
        saved    = max(0.0, baseline - cost)
        rec      = CostRecord(
            model=model, provider=provider, domain=domain, tier=tier,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=cost, baseline_usd=baseline, saved_usd=saved,
        )
        with self._lock:
            if self._max > 0:
                if len(self._records) >= self._max:
                    self._records.pop(0)
                self._records.append(rec)
            self._n           += 1
            self._total_cost  += cost
            self._total_base  += baseline
            self._total_saved += saved
            if cost == 0.0:
                self._free_calls += 1
            dom = domain.value
            self._domain_cost[dom]  = self._domain_cost.get(dom, 0.0) + cost
            self._domain_calls[dom] = self._domain_calls.get(dom, 0)  + 1
            self._model_cost[model] = self._model_cost.get(model, 0.0) + cost
            if self._since is None:
                self._since = rec.timestamp
        return rec

    def summary(self) -> CostSummary:
        with self._lock:
            n      = self._n
            total  = self._total_cost
            base   = self._total_base
            saved  = self._total_saved
            pct    = (saved / base * 100) if base else 0.0
            return CostSummary(
                total_calls=n,
                total_cost_usd=total,
                baseline_usd=base,
                saved_usd=saved,
                savings_pct=pct,
                cost_by_domain=dict(self._domain_cost),
                cost_by_model=dict(self._model_cost),
                calls_by_domain=dict(self._domain_calls),
                free_calls=self._free_calls,
                since=self._since,
            )

    def print_summary(self) -> None:
        print(str(self.summary()))

    def recent(self, n: int = 20) -> List[CostRecord]:
        with self._lock:
            return list(self._records[-n:])

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._n = 0
            self._total_cost = self._total_base = self._total_saved = 0.0
            self._free_calls = 0
            self._domain_cost.clear()
            self._domain_calls.clear()
            self._model_cost.clear()
            self._since = None
