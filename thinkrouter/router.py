"""
thinkrouter.router
~~~~~~~~~~~~~~~~~~
ThinkRouter v0.6.0 — Phase 3: Semantic Cache integrated.

Before running any classifier, ThinkRouter checks the atlas for a
previous query with the same semantic intent. On a hit, the stored
routing decision is returned in under 2ms — skipping both classifiers.
On a miss, the normal Phase 1 + Phase 2 pipeline runs.

New in v0.6.0:
  - SemanticCache integrated into every chat() / achat() call
  - Cache hits tracked in RouterResponse.cache_result
  - ThinkRouter.cache exposes the SemanticCache instance
  - cache_enabled / cache_threshold / cache_min_quality config options
  - Usage dashboard shows cache hit rate alongside savings
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from .cache import CacheResult, SemanticCache
from .classifier import BaseClassifier, ClassifierResult, get_classifier
from .config import Config, DEFAULT_CONFIG
from .constants import (
    ANTHROPIC_THINKING_BUDGETS,
    ANTHROPIC_THINKING_MODELS,
    OPENAI_REASONING_EFFORT,
    OPENAI_REASONING_MODELS,
    TIER_TOKEN_BUDGETS,
    ProviderLiteral,
    Tier,
)
from .domain import Domain, DomainClassifier, DomainResult
from .exceptions import ConfigurationError
from .providers import AnthropicAdapter, OpenAIAdapter
from .registry import DEFAULT_REGISTRY, ModelRegistry, ModelTarget
from .usage import UsageTracker


# ── Response ──────────────────────────────────────────────────────────────

@dataclass
class RouterResponse:
    """
    Unified response from ThinkRouter v0.6.0.

    Attributes
    ----------
    content          : Generated text.
    routing          : Complexity ClassifierResult (None on cache hit).
    domain_result    : Domain DomainResult (None on cache hit).
    model_target     : ModelTarget selected.
    cache_result     : CacheResult if routed from semantic cache, else None.
    raw              : Original provider response.
    provider         : Provider used.
    model            : Model identifier used.
    usage_tokens     : Token usage dict.
    record_id        : Atlas record UUID.
    reasoning_effort : OpenAI reasoning_effort applied.
    thinking_budget  : Anthropic thinking budget_tokens applied.
    """
    content:          str
    routing:          Optional[ClassifierResult]
    domain_result:    Optional[DomainResult]
    model_target:     Optional[ModelTarget]
    cache_result:     Optional[CacheResult]
    raw:              Any
    provider:         str
    model:            str
    usage_tokens:     Dict[str, int]
    record_id:        Optional[str]     = None
    reasoning_effort: Optional[str]     = None
    thinking_budget:  Optional[int]     = None

    @property
    def was_cached(self) -> bool:
        """True when this response used a cached routing decision."""
        return self.cache_result is not None

    @property
    def tier(self) -> Optional[Tier]:
        """Convenience — the routing tier regardless of source."""
        if self.cache_result:
            return self.cache_result.tier
        if self.routing:
            return self.routing.tier
        return None

    def __repr__(self) -> str:
        src = "cache" if self.was_cached else "classifiers"
        dom = (
            self.cache_result.domain.value if self.was_cached
            else (self.domain_result.domain.value if self.domain_result else "?")
        )
        return (
            f"RouterResponse("
            f"tier={self.tier.name if self.tier else '?'}, "
            f"domain={dom}, "
            f"src={src}, "
            f"model={self.model!r})"
        )


# ── ThinkRouter ───────────────────────────────────────────────────────────

class ThinkRouter:
    """
    Production-ready pre-inference routing layer — v0.6.0.

    Routing pipeline per query:
      1. [Phase 3] Embed query → check SemanticCache
         → HIT:  use cached domain + tier → skip to inference
         → MISS: continue ↓
      2. [Phase 1] DomainClassifier  → detect domain
      3. [Phase 1] ComplexityClassifier → detect tier
      4. [Phase 1] ModelRegistry.resolve() → pick specialist model
      5. [Phase 2] Store embedding + routing decision in Atlas (background)
      6. Provider API call → return RouterResponse

    Parameters
    ----------
    provider              : "openai" | "anthropic" | "ollama" | "generic"
    api_key               : Provider API key.
    model                 : Default model.
    classifier_backend    : "heuristic" | "distilbert"
    confidence_threshold  : Min complexity confidence.
    domain_routing        : Enable domain-aware model selection.
    domain_min_confidence : Min domain confidence to route.
    preferred_provider    : Provider preference for domain routing.
    registry              : Custom ModelRegistry.
    atlas_enabled         : Enable Phase 2 atlas storage. Default: True.
    cache_enabled         : Enable Phase 3 semantic cache. Default: True.
    cache_threshold       : Min cosine similarity for a cache hit. Default: 0.92.
    cache_min_quality     : Min quality score on cached records. Default: 0.70.
    cache_min_atlas_size  : Min atlas size before cache is active. Default: 50.
    embedder_backend      : "hash" | "openai" | "local"
    embedder_kwargs       : Passed to get_embedder().
    max_retries           : Retry attempts on transient errors.
    max_records           : Max usage tracker records.
    verbose               : Print routing decisions per call.
    ollama_url            : Ollama server URL.
    config                : Override with custom Config.
    **client_kwargs       : Passed to provider SDK constructor.

    Examples
    --------
    # Full Phase 3 — semantic cache active
    >>> client = ThinkRouter(provider="openai", cache_enabled=True)
    >>> r = client.chat("Write a binary search in Python.")
    >>> r.was_cached   # True once atlas has similar queries

    # Warmup the cache with known query types
    >>> from thinkrouter.domain import Domain
    >>> from thinkrouter.constants import Tier
    >>> client.cache.warmup(
    ...     queries=["Write a Python function.", "Implement a sort algorithm."],
    ...     domains=[Domain.CODE, Domain.CODE],
    ...     models=["deepseek-coder-v2", "deepseek-coder-v2"],
    ... )

    # Cache performance
    >>> client.cache.print_stats()
    """

    _DEFAULT_MODELS: Dict[str, str] = {
        "openai":    "gpt-4o",
        "anthropic": "claude-sonnet-4-6",
        "ollama":    "llama3.1",
        "generic":   "unknown",
    }

    def __init__(
        self,
        provider:              ProviderLiteral      = "openai",
        api_key:               Optional[str]        = None,
        model:                 Optional[str]        = None,
        classifier_backend:    str                  = "heuristic",
        confidence_threshold:  float                = 0.75,
        domain_routing:        bool                 = True,
        domain_min_confidence: float                = 0.45,
        preferred_provider:    Optional[str]        = None,
        registry:              Optional[ModelRegistry] = None,
        atlas_enabled:         Optional[bool]       = None,
        cache_enabled:         Optional[bool]       = None,
        cache_threshold:       Optional[float]      = None,
        cache_min_quality:     Optional[float]      = None,
        cache_min_atlas_size:  Optional[int]        = None,
        embedder_backend:      Optional[str]        = None,
        embedder_kwargs:       Optional[Dict]       = None,
        max_retries:           int                  = 3,
        max_records:           int                  = 10_000,
        verbose:               bool                 = False,
        ollama_url:            str                  = "http://localhost:11434",
        config:                Optional[Config]     = None,
        **client_kwargs:       Any,
    ) -> None:
        cfg = config or DEFAULT_CONFIG

        self.provider              = provider
        self.model                 = model or self._DEFAULT_MODELS.get(provider, "unknown")
        self.verbose               = verbose or cfg.verbose
        self.max_retries           = max_retries
        self.domain_routing        = domain_routing
        self.domain_min_confidence = domain_min_confidence
        self._preferred_provider   = preferred_provider or provider
        self._registry             = registry or DEFAULT_REGISTRY

        # Complexity classifier
        clf_kw: Dict[str, Any] = {}
        if classifier_backend == "distilbert":
            clf_kw["threshold"]  = confidence_threshold
            clf_kw["model_name"] = cfg.hf_model
        self._clf: BaseClassifier = get_classifier(classifier_backend, **clf_kw)
        self._threshold = confidence_threshold

        # Domain classifier
        self._domain_clf = DomainClassifier(min_confidence=domain_min_confidence)

        # Usage tracker
        self.usage = UsageTracker(max_records=max_records)

        # ── Phase 2 + Phase 3: Embedder, Atlas, Cache ──────────────────
        _atlas_enabled = atlas_enabled if atlas_enabled is not None else cfg.atlas_enabled
        _cache_enabled = cache_enabled if cache_enabled is not None else cfg.cache_enabled
        _emb_backend   = embedder_backend or cfg.embedder_backend
        _emb_kwargs    = embedder_kwargs or {}

        _cache_threshold      = cache_threshold      or cfg.cache_threshold
        _cache_min_quality    = cache_min_quality    or cfg.cache_min_quality
        _cache_min_atlas_size = cache_min_atlas_size or cfg.cache_min_atlas_size

        self._embedder = None
        self.atlas     = None
        self.cache: Optional[SemanticCache] = None

        if _atlas_enabled:
            try:
                from .embedder import get_embedder
                from .atlas import Atlas
                self._embedder = get_embedder(_emb_backend, **_emb_kwargs)
                self.atlas = Atlas(
                    path=cfg.atlas_path,
                    embedding_dim=self._embedder.dim,
                    embedding_backend=self._embedder._backend_name,
                    max_records=cfg.atlas_max,
                )
                if _cache_enabled:
                    self.cache = SemanticCache(
                        atlas=self.atlas,
                        embedder=self._embedder,
                        threshold=_cache_threshold,
                        min_quality=_cache_min_quality,
                        min_atlas_size=_cache_min_atlas_size,
                    )
            except Exception:
                self._embedder = None
                self.atlas     = None
                self.cache     = None

        # ── Provider adapter ───────────────────────────────────────────
        self._adapter: Optional[Any] = None
        self._ollama_adapter: Optional[Any] = None

        resolved_key = api_key or (
            cfg.openai_api_key if provider == "openai" else cfg.anthropic_api_key
        )

        if provider == "openai":
            if not resolved_key:
                raise ConfigurationError(
                    "No OpenAI API key. Pass api_key= or set OPENAI_API_KEY."
                )
            self._adapter = OpenAIAdapter(
                resolved_key, max_retries=max_retries, **client_kwargs
            )
        elif provider == "anthropic":
            if not resolved_key:
                raise ConfigurationError(
                    "No Anthropic API key. Pass api_key= or set ANTHROPIC_API_KEY."
                )
            self._adapter = AnthropicAdapter(
                resolved_key, max_retries=max_retries, **client_kwargs
            )
        elif provider == "ollama":
            from .ollama_adapter import OllamaAdapter
            self._adapter = OllamaAdapter(base_url=ollama_url)
        elif provider != "generic":
            raise ConfigurationError(
                f"Unknown provider: {provider!r}. "
                "Choose 'openai', 'anthropic', 'ollama', or 'generic'."
            )

        if domain_routing and provider != "ollama":
            try:
                from .ollama_adapter import OllamaAdapter
                self._ollama_adapter = OllamaAdapter(base_url=ollama_url)
            except Exception:
                pass

    # ── Classify only ──────────────────────────────────────────────────────

    def classify(self, query: str) -> ClassifierResult:
        return self._clf.predict(query)

    def classify_domain(self, query: str) -> DomainResult:
        return self._domain_clf.predict(query)

    def classify_batch(self, queries: List[str]) -> List[ClassifierResult]:
        return self._clf.predict_batch(queries)

    def classify_domain_batch(self, queries: List[str]) -> List[DomainResult]:
        return self._domain_clf.predict_batch(queries)

    def classify_full(self, query: str):
        return self._clf.predict(query), self._domain_clf.predict(query)

    # ── Quality feedback ───────────────────────────────────────────────────

    def update_quality(self, record_id: str, quality_score: float) -> None:
        """Update quality score for a past routing decision in the atlas."""
        if self.atlas and record_id:
            self.atlas.update_quality(record_id, quality_score)

    # ── Core routing pipeline ──────────────────────────────────────────────

    def _route(
        self,
        query:    str,
        model:    Optional[str],
        messages: Optional[List[Dict[str, str]]],
        system:   Optional[str],
    ):
        """
        Full routing pipeline. Returns:
          (clf_result, domain_result, model_target,
           cache_result, target_model, target_provider, adapter, embedding)
        """
        # Step 1: embed query (needed for cache + atlas)
        embedding = None
        if self._embedder:
            try:
                embedding = self._embedder.embed(query)
            except Exception:
                pass

        # Step 2: Phase 3 — semantic cache check
        cache_result: Optional[CacheResult] = None
        if self.cache and embedding is not None:
            cache_result = self.cache.lookup(embedding)

        # Step 3: classifiers (skip on cache hit)
        if cache_result:
            clf_result    = None
            domain_result = None
            cached_domain = cache_result.domain
            cached_tier   = cache_result.tier
            cached_model  = cache_result.model
            cached_prov   = cache_result.provider
        else:
            clf_result    = self._clf.predict(query)
            domain_result = None
            cached_domain = Domain.GENERAL
            cached_tier   = clf_result.tier
            cached_model  = model or self.model
            cached_prov   = self.provider

            if self.domain_routing:
                domain_result = self._domain_clf.predict(query)
                if domain_result.confidence >= self.domain_min_confidence:
                    cached_domain = domain_result.domain

        # Step 4: model registry
        model_target:   Optional[ModelTarget] = None
        target_model  = model or self.model
        target_prov   = self.provider
        adapter       = self._adapter

        if cache_result:
            # Use cached model/provider directly
            target_model = cached_model
            target_prov  = cached_prov
            if cached_prov == "ollama" and self._ollama_adapter:
                adapter = self._ollama_adapter
        else:
            if self.domain_routing and domain_result and \
                    domain_result.confidence >= self.domain_min_confidence:
                model_target = self._registry.resolve(
                    domain_result.domain,
                    preferred_provider=self._preferred_provider,
                )
                if model_target.provider == "ollama" and self.provider != "ollama":
                    if self._ollama_adapter and self._ollama_adapter.is_available():
                        adapter      = self._ollama_adapter
                        target_model = model_target.model
                        target_prov  = "ollama"
                    else:
                        fb = self._registry.resolve(
                            domain_result.domain,
                            preferred_provider=self.provider,
                        )
                        target_model = fb.model
                elif model_target.provider == self.provider:
                    target_model = model_target.model

        return (clf_result, domain_result, model_target,
                cache_result, target_model, target_prov, adapter, embedding)

    # ── Sync chat ──────────────────────────────────────────────────────────

    def chat(
        self,
        query:       str,
        model:       Optional[str]             = None,
        messages:    Optional[List[Dict[str, str]]] = None,
        system:      Optional[str]             = None,
        temperature: float                     = 0.7,
        **extra:     Any,
    ) -> RouterResponse:
        if self._adapter is None:
            raise ConfigurationError(
                "No provider adapter. Pass provider='openai', 'anthropic', or 'ollama'."
            )

        (clf_result, domain_result, model_target,
         cache_result, target_model, target_prov,
         adapter, embedding) = self._route(query, model, messages, system)

        tier = cache_result.tier if cache_result else (clf_result.tier if clf_result else Tier.FULL)
        self._log(clf_result, domain_result, cache_result, target_model, target_prov)

        msg_list = self._msgs(query, messages, system)
        kw       = dict(extra)
        if self.provider == "anthropic" and system:
            kw["system"] = system

        if target_prov == "ollama":
            content, raw, usage_tokens = adapter.call(
                messages=msg_list, model=target_model,
                max_tokens=TIER_TOKEN_BUDGETS[tier],
                temperature=temperature, **kw,
            )
            xparam = None
        else:
            content, raw, usage_tokens, xparam = adapter.call(
                messages=msg_list, model=target_model,
                tier=tier, temperature=temperature, **kw,
            )

        # Record in usage tracker
        if clf_result:
            self.usage.record(
                query=query, tier=clf_result.tier,
                confidence=clf_result.confidence,
                latency_ms=clf_result.latency_ms,
                model=target_model, provider=target_prov,
            )

        # Phase 2: store in atlas (background)
        record_id = self._store_async(
            query=query, embedding=embedding,
            domain=cache_result.domain if cache_result else
                   (domain_result.domain if domain_result else Domain.GENERAL),
            tier=tier, model=target_model, provider=target_prov,
        )

        return RouterResponse(
            content=content,
            routing=clf_result,
            domain_result=domain_result,
            model_target=model_target,
            cache_result=cache_result,
            raw=raw, provider=target_prov, model=target_model,
            usage_tokens=usage_tokens, record_id=record_id,
            reasoning_effort=xparam if target_prov == "openai" else None,
            thinking_budget=xparam if target_prov == "anthropic" else None,
        )

    # ── Async chat ─────────────────────────────────────────────────────────

    async def achat(
        self,
        query:       str,
        model:       Optional[str]             = None,
        messages:    Optional[List[Dict[str, str]]] = None,
        system:      Optional[str]             = None,
        temperature: float                     = 0.7,
        **extra:     Any,
    ) -> RouterResponse:
        if self._adapter is None:
            raise ConfigurationError("No provider adapter configured.")

        (clf_result, domain_result, model_target,
         cache_result, target_model, target_prov,
         adapter, embedding) = self._route(query, model, messages, system)

        tier = cache_result.tier if cache_result else (clf_result.tier if clf_result else Tier.FULL)
        self._log(clf_result, domain_result, cache_result, target_model, target_prov)

        msg_list = self._msgs(query, messages, system)
        kw       = dict(extra)
        if self.provider == "anthropic" and system:
            kw["system"] = system

        if target_prov == "ollama":
            content, raw, usage_tokens = await adapter.acall(
                messages=msg_list, model=target_model,
                max_tokens=TIER_TOKEN_BUDGETS[tier],
                temperature=temperature, **kw,
            )
            xparam = None
        else:
            content, raw, usage_tokens, xparam = await adapter.acall(
                messages=msg_list, model=target_model,
                tier=tier, temperature=temperature, **kw,
            )

        if clf_result:
            self.usage.record(
                query=query, tier=clf_result.tier,
                confidence=clf_result.confidence,
                latency_ms=clf_result.latency_ms,
                model=target_model, provider=target_prov,
            )

        record_id = self._store_async(
            query=query, embedding=embedding,
            domain=cache_result.domain if cache_result else
                   (domain_result.domain if domain_result else Domain.GENERAL),
            tier=tier, model=target_model, provider=target_prov,
        )

        return RouterResponse(
            content=content,
            routing=clf_result,
            domain_result=domain_result,
            model_target=model_target,
            cache_result=cache_result,
            raw=raw, provider=target_prov, model=target_model,
            usage_tokens=usage_tokens, record_id=record_id,
            reasoning_effort=xparam if target_prov == "openai" else None,
            thinking_budget=xparam if target_prov == "anthropic" else None,
        )

    # ── Streaming ──────────────────────────────────────────────────────────

    def stream(
        self,
        query:       str,
        model:       Optional[str] = None,
        system:      Optional[str] = None,
        temperature: float         = 0.7,
        **extra:     Any,
    ) -> Iterator[str]:
        if self._adapter is None:
            raise ConfigurationError("No provider adapter configured.")
        clf = self._clf.predict(query)
        target = model or self.model
        self._log(clf, None, None, target, self.provider)
        yield from self._adapter.stream(
            messages=self._msgs(query, None, system),
            model=target, tier=clf.tier, temperature=temperature, **extra,
        )
        self.usage.record(
            query=query, tier=clf.tier, confidence=clf.confidence,
            latency_ms=clf.latency_ms, model=target, provider=self.provider,
        )

    async def astream(
        self,
        query:       str,
        model:       Optional[str] = None,
        system:      Optional[str] = None,
        temperature: float         = 0.7,
        **extra:     Any,
    ) -> AsyncIterator[str]:
        if self._adapter is None:
            raise ConfigurationError("No provider adapter configured.")
        clf = self._clf.predict(query)
        target = model or self.model
        self._log(clf, None, None, target, self.provider)
        async for chunk in self._adapter.astream(
            messages=self._msgs(query, None, system),
            model=target, tier=clf.tier, temperature=temperature, **extra,
        ):
            yield chunk
        self.usage.record(
            query=query, tier=clf.tier, confidence=clf.confidence,
            latency_ms=clf.latency_ms, model=target, provider=self.provider,
        )

    # ── Background atlas store ──────────────────────────────────────────────

    def _store_async(self, query, embedding, domain, tier, model, provider) -> Optional[str]:
        if not self._embedder or not self.atlas or embedding is None:
            return None
        import uuid
        record_id = str(uuid.uuid4())
        atl, emb = self.atlas, embedding

        def _store():
            try:
                atl.store(
                    query=query, embedding=emb,
                    domain=domain, tier=tier,
                    model=model, provider=provider,
                    quality_score=None, latency_ms=0.0,
                )
            except Exception:
                pass

        t = threading.Thread(target=_store, daemon=True)
        t.start()
        return record_id

    # ── Helpers ────────────────────────────────────────────────────────────

    def _msgs(self, query, messages, system):
        if messages:
            return list(messages)
        if system and self.provider == "openai":
            return [{"role":"system","content":system},{"role":"user","content":query}]
        return [{"role":"user","content":query}]

    def _log(self, clf, dom, cache, model, provider):
        if not self.verbose:
            return
        if cache:
            print(
                f"[ThinkRouter/CACHE] sim={cache.similarity:.4f}  "
                f"domain={cache.domain.value}  tier={cache.tier.name}  "
                f"model={model}  lat={cache.latency_ms:.2f}ms"
            )
        else:
            dom_str = f"  domain={dom.domain.value}" if dom else ""
            print(
                f"[ThinkRouter] tier={clf.tier.name:<10} "
                f"conf={clf.confidence:.3f}  clf={clf.latency_ms:.2f}ms"
                f"{dom_str}  model={model}  provider={provider}"
            )

    def __repr__(self) -> str:
        s = self.usage.summary()
        cache_str = (
            f", hit_rate={self.cache.stats().hit_rate:.1f}%"
            if self.cache else ""
        )
        atlas_str = f", atlas={len(self.atlas)}" if self.atlas else ""
        return (
            f"ThinkRouter("
            f"provider={self.provider!r}, "
            f"model={self.model!r}, "
            f"calls={s.total_calls}, "
            f"savings={s.savings_pct:.1f}%"
            f"{atlas_str}{cache_str})"
        )
