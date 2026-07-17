"""
thinkrouter v0.5.0 — Phase 2: Embedding Layer.

    from thinkrouter import ThinkRouter

    client = ThinkRouter(provider="openai", atlas_enabled=True)
    r = client.chat("Write a binary search tree in Python.")

    # Phase 1: domain routing
    print(r.domain_result.domain)    # Domain.CODE
    print(r.model_target.model)      # deepseek-coder-v2

    # Phase 2: atlas record
    print(r.record_id)               # UUID stored in local atlas
    client.update_quality(r.record_id, 0.95)  # feedback loop

    # Atlas stats
    client.atlas.print_stats()
"""
from .atlas import Atlas, AtlasRecord, AtlasStats, SimilarResult
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

__version__ = "0.5.0"
__author__   = "ThinkRouter Contributors"
__license__  = "MIT"

__all__ = [
    # Core
    "ThinkRouter", "RouterResponse",
    # Classifier
    "BaseClassifier", "ClassifierResult",
    "DistilBertClassifier", "HeuristicClassifier", "get_classifier",
    # Domain
    "Domain", "DomainClassifier", "DomainResult", "DOMAIN_DESCRIPTIONS",
    # Registry
    "ModelRegistry", "ModelTarget", "DEFAULT_REGISTRY",
    # Phase 2 — Embedder
    "BaseEmbedder", "EmbeddingResult",
    "HashSketchEmbedder", "OpenAIEmbedder", "LocalEmbedder", "get_embedder",
    # Phase 2 — Atlas
    "Atlas", "AtlasRecord", "AtlasStats", "SimilarResult",
    # Config
    "Config", "DEFAULT_CONFIG",
    # Usage
    "UsageTracker", "UsageSummary", "CallRecord",
    # Constants
    "Tier", "TIER_LABELS", "TIER_TOKEN_BUDGETS",
    # Exceptions
    "ThinkRouterError", "ProviderError", "RateLimitError",
    "AuthenticationError", "ModelNotFoundError",
    "ClassifierError", "ConfigurationError",
    "__version__",
]
