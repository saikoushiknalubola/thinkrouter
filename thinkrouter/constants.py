"""
thinkrouter.constants
~~~~~~~~~~~~~~~~~~~~~
Tier definitions, token budgets, and type aliases.
All other modules import from here to avoid circular dependencies.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, Literal


class Tier(IntEnum):
    """
    Three-level compute budget for LLM queries.

    NO_THINK  — direct answer, no chain-of-thought needed.
    SHORT     — moderate reasoning, a few steps.
    FULL      — extended multi-stage reasoning.
    """
    NO_THINK = 0
    SHORT    = 1
    FULL     = 2


# Human-readable labels used in dashboard output and logging.
TIER_LABELS: Dict[Tier, str] = {
    Tier.NO_THINK: "no_think",
    Tier.SHORT:    "short_think",
    Tier.FULL:     "full_think",
}

# Thinking-token budgets per tier.
# Conservative estimates from Zhao et al. (2025) SelfBudgeter arXiv:2505.11274.
TIER_TOKEN_BUDGETS: Dict[Tier, int] = {
    Tier.NO_THINK: 50,
    Tier.SHORT:    800,
    Tier.FULL:     8_000,
}

TIER_DESCRIPTIONS: Dict[Tier, str] = {
    Tier.NO_THINK: "Direct answer — no chain-of-thought required",
    Tier.SHORT:    "Short reasoning trace — moderate multi-step problem",
    Tier.FULL:     "Full extended reasoning — complex multi-stage problem",
}

ProviderLiteral = Literal["openai", "anthropic", "generic"]
