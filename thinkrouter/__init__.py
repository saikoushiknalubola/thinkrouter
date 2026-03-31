"""
thinkrouter
~~~~~~~~~~~
Cut LLM reasoning-token costs by 60% with one line of code.

    from thinkrouter import ThinkRouter

    # OpenAI o1 — reasoning_effort auto-applied per query
    client = ThinkRouter(provider="openai", model="o1")
    r = client.chat("What is 2 + 3?")
    # r.reasoning_effort == "low"   ← 50 thinking tokens, not 8,000

    # Anthropic with extended thinking
    client = ThinkRouter(provider="anthropic", model="claude-opus-4-6")
    r = client.chat("Prove sqrt(2) is irrational.")
    # r.thinking_budget == 10000   ← full budget for a real proof

    client.usage.print_dashboard()

GitHub : https://github.com/saikoushiknalubola/thinkrouter
PyPI   : https://pypi.org/project/thinkrouter
"""
from .classifier import (
    BaseClassifier,
    ClassifierResult,
    DistilBertClassifier,
    HeuristicClassifier,
    get_classifier,
)
from .config import Config, DEFAULT_CONFIG
from .constants import TIER_LABELS, TIER_TOKEN_BUDGETS, Tier
from .exceptions import (
    AuthenticationError,
    ClassifierError,
    ConfigurationError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    ThinkRouterError,
)
from .router import RouterResponse, ThinkRouter
from .usage import CallRecord, UsageSummary, UsageTracker

__version__ = "0.3.0"
__author__  = "ThinkRouter Contributors"
__license__ = "MIT"

__all__ = [
    # Core
    "ThinkRouter",
    "RouterResponse",
    # Classifier
    "BaseClassifier",
    "ClassifierResult",
    "DistilBertClassifier",
    "HeuristicClassifier",
    "get_classifier",
    # Config
    "Config",
    "DEFAULT_CONFIG",
    # Usage
    "UsageTracker",
    "UsageSummary",
    "CallRecord",
    # Constants
    "Tier",
    "TIER_LABELS",
    "TIER_TOKEN_BUDGETS",
    # Exceptions
    "ThinkRouterError",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "ModelNotFoundError",
    "ClassifierError",
    "ConfigurationError",
    # Metadata
    "__version__",
]
