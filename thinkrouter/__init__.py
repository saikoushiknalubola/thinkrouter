"""
thinkrouter
~~~~~~~~~~~
Cut LLM reasoning-token costs by 60% with one line of code.

    from thinkrouter import ThinkRouter

    client   = ThinkRouter(provider="openai", model="o1")
    response = client.chat("What is the capital of France?")
    # reasoning_effort="low" applied — minimal thinking tokens used

    client   = ThinkRouter(provider="anthropic", model="claude-opus-4-6")
    response = client.chat("Prove that sqrt(2) is irrational.")
    # thinking budget_tokens=10000 applied automatically

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
from .constants import TIER_LABELS, TIER_TOKEN_BUDGETS, Tier
from .exceptions import (
    AuthenticationError,
    ClassifierError,
    ConfigurationError,
    ProviderError,
    RateLimitError,
    ThinkRouterError,
)
from .router import RouterResponse, ThinkRouter
from .usage import CallRecord, UsageSummary, UsageTracker

__version__ = "0.2.0"
__author__  = "ThinkRouter Contributors"
__license__ = "MIT"

__all__ = [
    "ThinkRouter",
    "RouterResponse",
    "BaseClassifier",
    "ClassifierResult",
    "DistilBertClassifier",
    "HeuristicClassifier",
    "get_classifier",
    "UsageTracker",
    "UsageSummary",
    "CallRecord",
    "Tier",
    "TIER_LABELS",
    "TIER_TOKEN_BUDGETS",
    "ThinkRouterError",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "ClassifierError",
    "ConfigurationError",
    "__version__",
]
