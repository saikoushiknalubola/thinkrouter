"""thinkrouter v0.7.0 — Confidence model · Cost tracker · Fallback chains."""
from .atlas import Atlas, AtlasRecord, AtlasStats, SimilarResult
from .cache import CacheResult, CacheStats, SemanticCache
from .classifier import (
    BaseClassifier, ClassifierResult,
    DistilBertClassifier, HeuristicClassifier, get_classifier,
)
from .confidence import (
    ConfidenceResult, HeuristicConfidenceModel, AtlasConfidenceModel,
    Recommendation, get_confidence_model,
)
from .config import Config, DEFAULT_CONFIG
from .constants import TIER_LABELS, TIER_TOKEN_BUDGETS, Tier
from .cost import CostRecord, CostSummary, CostTracker, MODEL_PRICING, get_cost_usd
from .domain import Domain, DomainClassifier, DomainResult, DOMAIN_DESCRIPTIONS
from .embedder import (
    BaseEmbedder, EmbeddingResult,
    HashSketchEmbedder, OpenAIEmbedder, LocalEmbedder, get_embedder,
)
from .exceptions import (
    AuthenticationError, ClassifierError, ConfigurationError,
    ModelNotFoundError, ProviderError, RateLimitError, ThinkRouterError,
)
from .fallback import FallbackChain, FallbackResult
from .registry import DEFAULT_REGISTRY, ModelRegistry, ModelTarget
from .router import RouterResponse, ThinkRouter
from .usage import CallRecord, UsageSummary, UsageTracker

__version__ = "0.7.0"
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
    "ConfidenceResult", "HeuristicConfidenceModel", "AtlasConfidenceModel",
    "Recommendation", "get_confidence_model",
    "CostRecord", "CostSummary", "CostTracker", "MODEL_PRICING", "get_cost_usd",
    "FallbackChain", "FallbackResult",
    "Config", "DEFAULT_CONFIG",
    "UsageTracker", "UsageSummary", "CallRecord",
    "Tier", "TIER_LABELS", "TIER_TOKEN_BUDGETS",
    "ThinkRouterError", "ProviderError", "RateLimitError",
    "AuthenticationError", "ModelNotFoundError",
    "ClassifierError", "ConfigurationError",
    "__version__",
]
