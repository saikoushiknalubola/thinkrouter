"""
thinkrouter.constants
~~~~~~~~~~~~~~~~~~~~~
Tier definitions, token budgets, and provider-specific reasoning
parameter mappings.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Dict, Literal


class Tier(IntEnum):
    """
    Three-level compute budget for LLM queries.

    NO_THINK  — direct answer, no chain-of-thought.
    SHORT     — moderate reasoning, a few steps.
    FULL      — extended multi-stage reasoning.
    """
    NO_THINK = 0
    SHORT    = 1
    FULL     = 2


TIER_LABELS: Dict[Tier, str] = {
    Tier.NO_THINK: "no_think",
    Tier.SHORT:    "short_think",
    Tier.FULL:     "full_think",
}

# Thinking-token budgets per tier.
# Based on: Zhao et al. (2025) SelfBudgeter arXiv:2505.11274
TIER_TOKEN_BUDGETS: Dict[Tier, int] = {
    Tier.NO_THINK: 50,
    Tier.SHORT:    800,
    Tier.FULL:     8_000,
}

# Anthropic extended thinking budget_tokens per tier.
# Used in thinking={"type": "enabled", "budget_tokens": N}
ANTHROPIC_THINKING_BUDGETS: Dict[Tier, int] = {
    Tier.NO_THINK: 0,       # thinking disabled entirely
    Tier.SHORT:    1_024,
    Tier.FULL:     10_000,
}

# OpenAI reasoning_effort per tier.
# Used for o1, o1-mini, o3, o3-mini models.
OPENAI_REASONING_EFFORT: Dict[Tier, str] = {
    Tier.NO_THINK: "low",
    Tier.SHORT:    "low",
    Tier.FULL:     "high",
}

TIER_DESCRIPTIONS: Dict[Tier, str] = {
    Tier.NO_THINK: "Direct answer — no chain-of-thought required",
    Tier.SHORT:    "Short reasoning trace — moderate multi-step problem",
    Tier.FULL:     "Full extended reasoning — complex multi-stage problem",
}

# Models that support reasoning_effort (OpenAI)
OPENAI_REASONING_MODELS = frozenset({
    "o1", "o1-mini", "o1-preview",
    "o3", "o3-mini",
    "o4-mini",
})

# Models that support extended thinking (Anthropic)
ANTHROPIC_THINKING_MODELS = frozenset({
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
})

ProviderLiteral = Literal["openai", "anthropic", "generic"]
