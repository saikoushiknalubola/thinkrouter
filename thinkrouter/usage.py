"""
thinkrouter.usage
~~~~~~~~~~~~~~~~~
Thread-safe usage tracker with aggregate savings dashboard.
Zero external dependencies.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .constants import (
    TIER_DESCRIPTIONS,
    TIER_LABELS,
    TIER_TOKEN_BUDGETS,
    Tier,
)


@dataclass
class CallRecord:
    """Single routing event record."""
    query_preview:   str
    tier:            Tier
    confidence:      float
    tokens_used:     int
    tokens_saved:    int
    latency_ms:      float
    model:           str
    provider:        str
    timestamp:       datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class UsageSummary:
    """Aggregate statistics over all recorded calls."""
    total_calls:        int
    total_tokens_used:  int
    total_tokens_saved: int
    savings_pct:        float
    avg_confidence:     float
    avg_latency_ms:     float
    tier_breakdown:     Dict[Tier, int]
    since:              Optional[datetime]

    def __str__(self) -> str:
        lines = [
            "",
            "  ThinkRouter — Usage Dashboard",
            "  " + "─" * 48,
            f"  Total calls          : {self.total_calls:,}",
            f"  Tokens used          : {self.total_tokens_used:,}",
            f"  Tokens saved         : {self.total_tokens_saved:,}",
            f"  Compute savings      : {self.savings_pct:.1f}%",
            f"  Avg confidence       : {self.avg_confidence:.3f}",
            f"  Avg classifier time  : {self.avg_latency_ms:.2f} ms",
            "",
            "  Routing breakdown:",
        ]
        for tier in Tier:
            count = self.tier_breakdown.get(tier, 0)
            pct   = (count / self.total_calls * 100) if self.total_calls else 0.0
            lines.append(
                f"    {TIER_LABELS[tier]:<15} : {count:>6,}  ({pct:.1f}%)"
                f"  — {TIER_DESCRIPTIONS[tier]}"
            )
        if self.since:
            lines.append(
                f"\n  Tracking since : {self.since.strftime('%Y-%m-%d %H:%M UTC')}"
            )
        lines.append("")
        return "\n".join(lines)


class UsageTracker:
    """
    Thread-safe in-memory usage accumulator.

    Parameters
    ----------
    max_records : int
        Maximum individual CallRecord objects retained (FIFO eviction).
        Set to 0 to track aggregates only.
    """

    def __init__(self, max_records: int = 10_000) -> None:
        self._lock         = threading.Lock()
        self._records:     List[CallRecord] = []
        self._max_records  = max_records
        self._n            = 0
        self._tokens_used  = 0
        self._tokens_saved = 0
        self._sum_conf     = 0.0
        self._sum_lat      = 0.0
        self._tier_counts: Dict[Tier, int] = {t: 0 for t in Tier}
        self._first_ts:    Optional[datetime] = None

    def record(
        self,
        query:      str,
        tier:       Tier,
        confidence: float,
        latency_ms: float,
        model:      str = "",
        provider:   str = "",
    ) -> CallRecord:
        """Record a routing decision. Thread-safe."""
        used  = TIER_TOKEN_BUDGETS[tier]
        saved = TIER_TOKEN_BUDGETS[Tier.FULL] - used
        rec   = CallRecord(
            query_preview=query[:80].replace("\n", " "),
            tier=tier, confidence=confidence,
            tokens_used=used, tokens_saved=saved,
            latency_ms=latency_ms, model=model, provider=provider,
        )
        with self._lock:
            if self._max_records > 0:
                if len(self._records) >= self._max_records:
                    self._records.pop(0)
                self._records.append(rec)
            self._n            += 1
            self._tokens_used  += used
            self._tokens_saved += saved
            self._sum_conf     += confidence
            self._sum_lat      += latency_ms
            self._tier_counts[tier] += 1
            if self._first_ts is None:
                self._first_ts = rec.timestamp
        return rec

    def summary(self) -> UsageSummary:
        """Return a snapshot of current aggregate statistics."""
        with self._lock:
            n = self._n
            if n == 0:
                return UsageSummary(
                    total_calls=0, total_tokens_used=0,
                    total_tokens_saved=0, savings_pct=0.0,
                    avg_confidence=0.0, avg_latency_ms=0.0,
                    tier_breakdown={t: 0 for t in Tier}, since=None,
                )
            baseline    = n * TIER_TOKEN_BUDGETS[Tier.FULL]
            savings_pct = (self._tokens_saved / baseline * 100) if baseline else 0.0
            return UsageSummary(
                total_calls=n,
                total_tokens_used=self._tokens_used,
                total_tokens_saved=self._tokens_saved,
                savings_pct=savings_pct,
                avg_confidence=self._sum_conf / n,
                avg_latency_ms=self._sum_lat / n,
                tier_breakdown=dict(self._tier_counts),
                since=self._first_ts,
            )

    def recent(self, n: int = 20) -> List[CallRecord]:
        """Return the n most recent call records."""
        with self._lock:
            return list(self._records[-n:])

    def reset(self) -> None:
        """Clear all accumulated data."""
        with self._lock:
            self._records.clear()
            self._n = self._tokens_used = self._tokens_saved = 0
            self._sum_conf = self._sum_lat = 0.0
            self._tier_counts  = {t: 0 for t in Tier}
            self._first_ts     = None

    def print_dashboard(self) -> None:
        print(str(self.summary()))
