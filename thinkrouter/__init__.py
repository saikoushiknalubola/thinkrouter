"""
thinkrouter
~~~~~~~~~~~

Cut LLM reasoning-token costs by 60% with one line of code.

    from thinkrouter import ThinkRouter

    client   = ThinkRouter(provider="openai")
    response = client.chat("What is the capital of France?")
    # Routes to NO_THINK → 50 tokens, not 8,000

    client.usage.print_dashboard()
    # Compute savings: 98.8%  |  Avg latency: 0.4ms

GitHub : https://github.com/thinkrouter/thinkrouter
Docs   : https://github.com/thinkrouter/thinkrouter#readme
PyPI   : https://pypi.org/project/thinkrouter
"""

from .classifier import (
    BaseClassifier,
    ClassifierResult,
    DistilBertClassifier,
    HeuristicClassifier,
    get_classifier,
)
from .constants import Tier, TIER_LABELS, TIER_TOKEN_BUDGETS
from .router import RouterResponse, ThinkRouter
from .usage import CallRecord, UsageSummary, UsageTracker

__version__ = "0.1.0"
__author__  = "ThinkRouter Contributors"
__license__ = "MIT"

__all__ = [
    # ── Main entry point ──────────────────────────────────────────────────
    "ThinkRouter",
    # ── Response ──────────────────────────────────────────────────────────
    "RouterResponse",
    # ── Classifier ────────────────────────────────────────────────────────
    "BaseClassifier",
    "ClassifierResult",
    "DistilBertClassifier",
    "HeuristicClassifier",
    "get_classifier",
    # ── Usage tracking ────────────────────────────────────────────────────
    "UsageTracker",
    "UsageSummary",
    "CallRecord",
    # ── Constants ─────────────────────────────────────────────────────────
    "Tier",
    "TIER_LABELS",
    "TIER_TOKEN_BUDGETS",
    # ── Metadata ──────────────────────────────────────────────────────────
    "__version__",
]
