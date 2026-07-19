"""
thinkrouter v0.6.0 — Phase 3: Semantic Cache.

    from thinkrouter import ThinkRouter

    client = ThinkRouter(provider="openai", cache_enabled=True)

    # Warmup cache with known query types
    client.cache.warmup(
        queries=["Write a binary search in Python.", "Implement merge sort."],
        domains=[Domain.CODE, Domain.CODE],
        models=["deepseek-coder-v2", "deepseek-coder-v2"],
    )

    r = client.chat("Write a binary search function.")
    print(r.was_cached)              # True — hit from warmup
    print(r.cache_result.similarity) # 0.9842

    client.cache.print_stats()
    # Hit rate: 100.0%  |  Avg similarity: 0.984  |  Atlas: 2
"""
from .atlas import Atlas, AtlasRecord, AtlasStats, SimilarResult
from .cache import CacheResult, CacheStats, SemanticCache
from .classifier import (
    BaseClassifier, ClassifierResult,
    DistilBertClassifier, HeuristicClassifier, get_classifier,
)
from .config import Config, DEFAULT_CONFIG
from .constants import TIER_LABELS, TIER_TOKEN_BUDGETS, Tier
from .domain import Domain, DomainClassifier, DomainResult, DOMAIN_DESCRIPTIONS
from .embedder import (
    BaseEmbedder, EmbeddingResult,
    HashSketchEmbedder, OpenAIEmbedder, LocalEmbedder, get_embedder,
)
from .exceptions import (
    AuthenticationError, ClassifierError, ConfigurationError,
    ModelNotFoundError, ProviderError, RateLimitError, ThinkRouterError,
)
from .registry import DEFAULT_REGISTRY, ModelRegistry, ModelTarget
from .router import RouterResponse, ThinkRouter
from .usage import CallRecord, UsageSummary, UsageTracker

__version__ = "0.6.0"
__author__   = "ThinkRouter Contributors"
__license__  = "MIT"

__all__ = [
    "ThinkRouter", "RouterResponse",
    "BaseClassifier", "ClassifierResult",
    "DistilBertClassifier", "HeuristicClassifier", "get_classifier",
    "Domain", "DomainClassifier", "DomainResult", "DOMAIN_DESCRIPTIONS",
    "ModelRegistry", "ModelTarget", "DEFAULT_REGISTRY",
    "BaseEmbedder", "EmbeddingResult",
    "HashSketchEmbedder", "OpenAIEmbedder", "LocalEmbedder", "get_embedder",
    "Atlas", "AtlasRecord", "AtlasStats", "SimilarResult",
    "SemanticCache", "CacheResult", "CacheStats",
    "Config", "DEFAULT_CONFIG",
    "UsageTracker", "UsageSummary", "CallRecord",
    "Tier", "TIER_LABELS", "TIER_TOKEN_BUDGETS",
    "ThinkRouterError", "ProviderError", "RateLimitError",
    "AuthenticationError", "ModelNotFoundError",
    "ClassifierError", "ConfigurationError",
    "__version__",
]
