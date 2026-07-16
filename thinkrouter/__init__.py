"""
thinkrouter v0.4.0 — semantic routing layer for LLM systems.

    from thinkrouter import ThinkRouter

    # Domain routing — routes to specialist model per query domain
    client = ThinkRouter(provider="ollama")   # free local models
    r = client.chat("Write a binary search tree in Python.")
    print(r.domain_result.domain)   # Domain.CODE
    print(r.model_target.model)     # deepseek-coder-v2

    # OpenAI with domain routing
    client = ThinkRouter(provider="openai", preferred_provider="openai")
    r = client.chat("Prove the Pythagorean theorem.")
    print(r.domain_result.domain)   # Domain.MATH

    client.usage.print_dashboard()
"""
from .classifier import (
    BaseClassifier, ClassifierResult,
    DistilBertClassifier, HeuristicClassifier, get_classifier,
)
from .config import Config, DEFAULT_CONFIG
from .constants import TIER_LABELS, TIER_TOKEN_BUDGETS, Tier
from .domain import Domain, DomainClassifier, DomainResult, DOMAIN_DESCRIPTIONS
from .exceptions import (
    AuthenticationError, ClassifierError, ConfigurationError,
    ModelNotFoundError, ProviderError, RateLimitError, ThinkRouterError,
)
from .registry import DEFAULT_REGISTRY, ModelRegistry, ModelTarget
from .router import RouterResponse, ThinkRouter
from .usage import CallRecord, UsageSummary, UsageTracker

__version__ = "0.4.0"
__author__   = "ThinkRouter Contributors"
__license__  = "MIT"

__all__ = [
    "ThinkRouter", "RouterResponse",
    "BaseClassifier", "ClassifierResult",
    "DistilBertClassifier", "HeuristicClassifier", "get_classifier",
    "Domain", "DomainClassifier", "DomainResult", "DOMAIN_DESCRIPTIONS",
    "ModelRegistry", "ModelTarget", "DEFAULT_REGISTRY",
    "Config", "DEFAULT_CONFIG",
    "UsageTracker", "UsageSummary", "CallRecord",
    "Tier", "TIER_LABELS", "TIER_TOKEN_BUDGETS",
    "ThinkRouterError", "ProviderError", "RateLimitError",
    "AuthenticationError", "ModelNotFoundError",
    "ClassifierError", "ConfigurationError",
    "__version__",
]
