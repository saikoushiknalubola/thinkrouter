"""
thinkrouter.router
~~~~~~~~~~~~~~~~~~
ThinkRouter v0.7.0 — Phase 4 advanced features.

New in v0.7.0:
  - Confidence model: hallucination risk prediction before every call
  - Cost tracker: real USD spend tracked per routing decision
  - Fallback chain: automatic failover across providers
  - RouterResponse.confidence_result, .cost_usd, .fallback_result
  - ThinkRouter.cost_tracker, .confidence_model
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .cache import CacheResult, SemanticCache
from .classifier import BaseClassifier, ClassifierResult, get_classifier
from .confidence import ConfidenceResult, HeuristicConfidenceModel, get_confidence_model
from .config import Config, DEFAULT_CONFIG
from .constants import (
    TIER_TOKEN_BUDGETS,
    ProviderLiteral,
    Tier,
)
from .cost import CostRecord, CostTracker
from .domain import Domain, DomainClassifier, DomainResult
from .exceptions import ConfigurationError
from .fallback import FallbackChain, FallbackResult
from .providers import AnthropicAdapter, OpenAIAdapter
from .registry import DEFAULT_REGISTRY, ModelRegistry, ModelTarget
from .usage import UsageTracker


# ── Response ──────────────────────────────────────────────────────────────

@dataclass
class RouterResponse:
    """
    Unified response from ThinkRouter v0.7.0.

    Attributes
    ----------
    content            : Generated text.
    routing            : Complexity ClassifierResult (None on cache hit).
    domain_result      : Domain DomainResult (None on cache hit).
    model_target       : ModelTarget selected.
    cache_result       : CacheResult if semantic cache was used.
    confidence_result  : ConfidenceResult — hallucination risk assessment.
    cost_record        : CostRecord — actual USD cost of this call.
    fallback_result    : FallbackResult if fallback chain was used.
    raw                : Original provider response.
    provider           : Provider that responded.
    model              : Model identifier used.
    usage_tokens       : Token usage dict.
    record_id          : Atlas UUID — use with update_quality().
    reasoning_effort   : OpenAI reasoning_effort applied.
    thinking_budget    : Anthropic budget_tokens applied.
    """
    content:           str
    routing:           Optional[ClassifierResult]
    domain_result:     Optional[DomainResult]
    model_target:      Optional[ModelTarget]
    cache_result:      Optional[CacheResult]
    confidence_result: Optional[ConfidenceResult]
    cost_record:       Optional[CostRecord]
    fallback_result:   Optional[FallbackResult]
    raw:               Any
    provider:          str
    model:             str
    usage_tokens:      Dict[str, int]
    record_id:         Optional[str]     = None
    reasoning_effort:  Optional[str]     = None
    thinking_budget:   Optional[int]     = None

    @property
    def was_cached(self) -> bool:
        return self.cache_result is not None

    @property
    def fallback_used(self) -> bool:
        return self.fallback_result is not None and self.fallback_result.fallback_used

    @property
    def is_high_risk(self) -> bool:
        return self.confidence_result is not None and self.confidence_result.is_high_risk

    @property
    def cost_usd(self) -> float:
        return self.cost_record.cost_usd if self.cost_record else 0.0

    @property
    def tier(self) -> Optional[Tier]:
        if self.cache_result:
            return self.cache_result.tier
        if self.routing:
            return self.routing.tier
        return None

    def __repr__(self) -> str:
        src  = "cache" if self.was_cached else "classifiers"
        dom  = (
            self.cache_result.domain.value if self.was_cached
            else (self.domain_result.domain.value if self.domain_result else "?")
        )
        risk = f", risk={self.confidence_result.risk_score:.2f}" if self.confidence_result else ""
        cost = f", cost=${self.cost_usd:.6f}" if self.cost_record else ""
        fb   = ", fallback=True" if self.fallback_used else ""
        return (
            f"RouterResponse("
            f"tier={self.tier.name if self.tier else '?'}, "
            f"domain={dom}, src={src}"
            f"{risk}{cost}{fb}, model={self.model!r})"
        )


# ── ThinkRouter ───────────────────────────────────────────────────────────

class ThinkRouter:
    """
    Production-ready semantic routing layer — v0.7.0.

    Full routing pipeline per query:
      1. [Phase 3] Semantic cache check — return in <2ms on hit
      2. [Phase 1] Domain + complexity classifiers
      3. [Phase 1] Model registry lookup
      4. [Phase 4] Confidence model — hallucination risk
      5. Provider API call (with optional fallback chain)
      6. [Phase 2] Store in Atlas (background thread)
      7. Cost tracking

    Parameters
    ----------
    provider               : "openai" | "anthropic" | "ollama" | "generic"
    api_key                : Provider API key.
    model                  : Default model.
    classifier_backend     : "heuristic" | "distilbert"
    confidence_threshold   : Min complexity confidence.
    domain_routing         : Enable domain-aware model selection.
    domain_min_confidence  : Min domain confidence to route.
    preferred_provider     : Provider preference for domain routing.
    registry               : Custom ModelRegistry.
    atlas_enabled          : Enable Phase 2 atlas storage.
    cache_enabled          : Enable Phase 3 semantic cache.
    cache_threshold        : Min cosine similarity for cache hit.
    cache_min_quality      : Min quality score on cached records.
    cache_min_atlas_size   : Min atlas size before cache activates.
    embedder_backend       : "hash" | "openai" | "local"
    confidence_enabled     : Enable Phase 4 confidence model. Default: True.
    confidence_backend     : "heuristic" | "atlas"
    escalation_model       : Model to use when confidence is ESCALATE.
    cost_tracking          : Enable real-time cost tracking. Default: True.
    fallback_providers     : Ordered list of fallback provider names.
    fallback_api_keys      : API keys for fallback providers.
    max_retries            : Retry attempts on transient errors.
    max_records            : Max usage tracker records.
    verbose                : Print routing decision per call.
    ollama_url             : Ollama server URL.
    config                 : Override with custom Config.
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
        confidence_enabled:    bool                 = True,
        confidence_backend:    str                  = "heuristic",
        escalation_model:      Optional[str]        = None,
        cost_tracking:         bool                 = True,
        fallback_providers:    Optional[List[str]]  = None,
        fallback_api_keys:     Optional[Dict[str,str]] = None,
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
        self._escalation_model     = escalation_model

        # Classifiers
        clf_kw: Dict[str, Any] = {}
        if classifier_backend == "distilbert":
            clf_kw["threshold"]  = confidence_threshold
            clf_kw["model_name"] = cfg.hf_model
        self._clf: BaseClassifier = get_classifier(classifier_backend, **clf_kw)
        self._threshold           = confidence_threshold
        self._domain_clf          = DomainClassifier(min_confidence=domain_min_confidence)

        # Usage tracker
        self.usage = UsageTracker(max_records=max_records)

        # Phase 4 — Confidence model
        self.confidence_model: Optional[HeuristicConfidenceModel] = None
        if confidence_enabled:
            self.confidence_model = get_confidence_model(confidence_backend)

        # Cost tracker
        self.cost_tracker: Optional[CostTracker] = None
        if cost_tracking:
            self.cost_tracker = CostTracker()

        # Phase 2 + 3 — Embedder, Atlas, Cache
        _atlas_enabled = atlas_enabled if atlas_enabled is not None else cfg.atlas_enabled
        _cache_enabled = cache_enabled if cache_enabled is not None else cfg.cache_enabled
        _emb_backend   = embedder_backend or cfg.embedder_backend
        _emb_kwargs    = embedder_kwargs or {}

        self._embedder            = None
        self.atlas                = None
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
                    _thresh  = cache_threshold      or cfg.cache_threshold
                    _min_q   = cache_min_quality    or cfg.cache_min_quality
                    _min_sz  = cache_min_atlas_size or cfg.cache_min_atlas_size
                    self.cache = SemanticCache(
                        atlas=self.atlas, embedder=self._embedder,
                        threshold=_thresh, min_quality=_min_q,
                        min_atlas_size=_min_sz,
                    )
                    # Upgrade confidence model to atlas-backed if atlas available
                    if confidence_enabled and confidence_backend == "heuristic" and self.atlas:
                        pass  # stay heuristic until atlas is large enough
            except Exception:
                self._embedder = None
                self.atlas     = None
                self.cache     = None

        # Primary adapter
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

        # Fallback chain
        self._fallback: Optional[FallbackChain] = None
        if fallback_providers and self._adapter is not None:
            self._fallback = self._build_fallback(
                primary_provider=provider,
                primary_adapter=self._adapter,
                primary_model=self.model,
                fallback_providers=fallback_providers,
                fallback_api_keys=fallback_api_keys or {},
                ollama_url=ollama_url,
                max_retries=max_retries,
                cfg=cfg,
            )

    # ── Fallback chain builder ─────────────────────────────────────────────

    def _build_fallback(
        self,
        primary_provider: str,
        primary_adapter:  Any,
        primary_model:    str,
        fallback_providers: List[str],
        fallback_api_keys:  Dict[str, str],
        ollama_url:         str,
        max_retries:        int,
        cfg:                Config,
    ) -> Optional[FallbackChain]:
        adapters = [(primary_provider, primary_adapter)]
        models   = [primary_model]

        for prov in fallback_providers:
            if prov == "ollama":
                try:
                    from .ollama_adapter import OllamaAdapter
                    adapters.append(("ollama", OllamaAdapter(base_url=ollama_url)))
                    models.append(self._DEFAULT_MODELS["ollama"])
                except Exception:
                    pass
            elif prov == "openai":
                key = fallback_api_keys.get("openai", cfg.openai_api_key)
                if key:
                    adapters.append(("openai", OpenAIAdapter(key, max_retries=max_retries)))
                    models.append(self._DEFAULT_MODELS["openai"])
            elif prov == "anthropic":
                key = fallback_api_keys.get("anthropic", cfg.anthropic_api_key)
                if key:
                    adapters.append(("anthropic", AnthropicAdapter(key, max_retries=max_retries)))
                    models.append(self._DEFAULT_MODELS["anthropic"])

        if len(adapters) < 2:
            return None
        return FallbackChain(adapters=adapters, models=models)

    # ── Classify-only API ──────────────────────────────────────────────────

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

    def assess_confidence(self, query: str, model: Optional[str] = None) -> Optional[ConfidenceResult]:
        if not self.confidence_model:
            return None
        return self.confidence_model.predict(query, model or self.model)

    # ── Quality feedback ───────────────────────────────────────────────────

    def update_quality(self, record_id: str, quality_score: float) -> None:
        if self.atlas and record_id:
            self.atlas.update_quality(record_id, quality_score)

    # ── Core pipeline ──────────────────────────────────────────────────────

    def _pipeline(self, query: str, override_model: Optional[str]):
        """
        Run the full pre-inference pipeline. Returns everything the
        adapter call needs plus all routing metadata.
        """
        # 1. Embed
        embedding = None
        if self._embedder:
            try:
                embedding = self._embedder.embed(query)
            except Exception:
                pass

        # 2. Semantic cache
        cache_result: Optional[CacheResult] = None
        if self.cache and embedding is not None:
            cache_result = self.cache.lookup(embedding)

        # 3. Classifiers (skip on cache hit)
        if cache_result:
            clf_result    = None
            domain_result = None
            tier          = cache_result.tier
            domain        = cache_result.domain
        else:
            clf_result    = self._clf.predict(query)
            domain_result = None
            tier          = clf_result.tier
            domain        = Domain.GENERAL

            if self.domain_routing:
                domain_result = self._domain_clf.predict(query)
                if domain_result.confidence >= self.domain_min_confidence:
                    domain = domain_result.domain

        # 4. Confidence model
        conf_result: Optional[ConfidenceResult] = None
        if self.confidence_model and not cache_result:
            target_for_conf = override_model or self.model
            conf_result     = self.confidence_model.predict(query, target_for_conf)

        # 5. Model registry
        model_target:  Optional[ModelTarget] = None
        target_model  = override_model or self.model
        target_prov   = self.provider
        adapter       = self._adapter

        if not cache_result:
            if self.domain_routing and domain_result and \
                    domain_result.confidence >= self.domain_min_confidence:
                use_prov = self._preferred_provider
                # Escalate to stronger model if confidence model says so
                if conf_result and conf_result.recommendation.value in ("escalate",) \
                        and self._escalation_model:
                    target_model = self._escalation_model
                else:
                    model_target = self._registry.resolve(
                        domain_result.domain,
                        preferred_provider=use_prov,
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
        else:
            target_model = cache_result.model
            target_prov  = cache_result.provider
            if target_prov == "ollama" and self._ollama_adapter:
                adapter = self._ollama_adapter

        return (clf_result, domain_result, model_target, cache_result,
                conf_result, tier, domain, target_model, target_prov,
                adapter, embedding)

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

        (clf_result, domain_result, model_target, cache_result,
         conf_result, tier, domain, target_model, target_prov,
         adapter, embedding) = self._pipeline(query, model)

        self._log(clf_result, domain_result, cache_result, conf_result, target_model, target_prov)

        msg_list = self._msgs(query, messages, system)
        kw       = dict(extra)
        if self.provider == "anthropic" and system:
            kw["system"] = system

        fallback_result: Optional[FallbackResult] = None

        # Use fallback chain if configured, else direct call
        if self._fallback and not cache_result:
            content, raw, usage_tokens, xparam, fallback_result = self._fallback.call(
                messages=msg_list, tier=tier, temperature=temperature, **kw,
            )
            actual_prov  = fallback_result.succeeded
            actual_model = self._fallback._models[
                [p for p,_ in self._fallback._adapters].index(actual_prov)
            ]
        elif target_prov == "ollama":
            content, raw, usage_tokens = adapter.call(
                messages=msg_list, model=target_model,
                max_tokens=TIER_TOKEN_BUDGETS[tier],
                temperature=temperature, **kw,
            )
            xparam = None
            actual_prov = "ollama"
            actual_model = target_model
        else:
            content, raw, usage_tokens, xparam = adapter.call(
                messages=msg_list, model=target_model,
                tier=tier, temperature=temperature, **kw,
            )
            actual_prov = target_prov
            actual_model = target_model

        if clf_result:
            self.usage.record(
                query=query, tier=clf_result.tier,
                confidence=clf_result.confidence,
                latency_ms=clf_result.latency_ms,
                model=actual_model, provider=actual_prov,
            )

        cost_rec = None
        if self.cost_tracker:
            cost_rec = self.cost_tracker.record(
                model=actual_model, provider=actual_prov,
                domain=domain, tier=tier,
                input_tokens=usage_tokens.get("prompt_tokens", 0),
                output_tokens=usage_tokens.get("completion_tokens", 0),
            )

        record_id = self._store_async(query, embedding, domain, tier, actual_model, actual_prov)

        return RouterResponse(
            content=content, routing=clf_result,
            domain_result=domain_result, model_target=model_target,
            cache_result=cache_result, confidence_result=conf_result,
            cost_record=cost_rec, fallback_result=fallback_result,
            raw=raw, provider=actual_prov, model=actual_model,
            usage_tokens=usage_tokens, record_id=record_id,
            reasoning_effort=xparam if actual_prov == "openai" else None,
            thinking_budget=xparam if actual_prov == "anthropic" else None,
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

        (clf_result, domain_result, model_target, cache_result,
         conf_result, tier, domain, target_model, target_prov,
         adapter, embedding) = self._pipeline(query, model)

        self._log(clf_result, domain_result, cache_result, conf_result, target_model, target_prov)

        msg_list = self._msgs(query, messages, system)
        kw       = dict(extra)
        if self.provider == "anthropic" and system:
            kw["system"] = system

        fallback_result = None

        if self._fallback and not cache_result:
            content, raw, usage_tokens, xparam, fallback_result = await self._fallback.acall(
                messages=msg_list, tier=tier, temperature=temperature, **kw,
            )
            actual_prov  = fallback_result.succeeded
            actual_model = self._fallback._models[
                [p for p,_ in self._fallback._adapters].index(actual_prov)
            ]
        elif target_prov == "ollama":
            content, raw, usage_tokens = await adapter.acall(
                messages=msg_list, model=target_model,
                max_tokens=TIER_TOKEN_BUDGETS[tier],
                temperature=temperature, **kw,
            )
            xparam = None
            actual_prov = "ollama"
            actual_model = target_model
        else:
            content, raw, usage_tokens, xparam = await adapter.acall(
                messages=msg_list, model=target_model,
                tier=tier, temperature=temperature, **kw,
            )
            actual_prov = target_prov
            actual_model = target_model

        if clf_result:
            self.usage.record(
                query=query, tier=clf_result.tier,
                confidence=clf_result.confidence,
                latency_ms=clf_result.latency_ms,
                model=actual_model, provider=actual_prov,
            )

        cost_rec = None
        if self.cost_tracker:
            cost_rec = self.cost_tracker.record(
                model=actual_model, provider=actual_prov,
                domain=domain, tier=tier,
                input_tokens=usage_tokens.get("prompt_tokens", 0),
                output_tokens=usage_tokens.get("completion_tokens", 0),
            )

        record_id = self._store_async(query, embedding, domain, tier, actual_model, actual_prov)

        return RouterResponse(
            content=content, routing=clf_result,
            domain_result=domain_result, model_target=model_target,
            cache_result=cache_result, confidence_result=conf_result,
            cost_record=cost_rec, fallback_result=fallback_result,
            raw=raw, provider=actual_prov, model=actual_model,
            usage_tokens=usage_tokens, record_id=record_id,
            reasoning_effort=xparam if actual_prov == "openai" else None,
            thinking_budget=xparam if actual_prov == "anthropic" else None,
        )

    # ── Stream ─────────────────────────────────────────────────────────────

    def stream(self, query, model=None, system=None, temperature=0.7, **extra):
        if self._adapter is None:
            raise ConfigurationError("No provider adapter configured.")
        clf    = self._clf.predict(query)
        target = model or self.model
        self._log(clf, None, None, None, target, self.provider)
        yield from self._adapter.stream(
            messages=self._msgs(query, None, system),
            model=target, tier=clf.tier, temperature=temperature, **extra,
        )
        self.usage.record(
            query=query, tier=clf.tier, confidence=clf.confidence,
            latency_ms=clf.latency_ms, model=target, provider=self.provider,
        )

    async def astream(self, query, model=None, system=None, temperature=0.7, **extra):
        if self._adapter is None:
            raise ConfigurationError("No provider adapter configured.")
        clf    = self._clf.predict(query)
        target = model or self.model
        self._log(clf, None, None, None, target, self.provider)
        async for chunk in self._adapter.astream(
            messages=self._msgs(query, None, system),
            model=target, tier=clf.tier, temperature=temperature, **extra,
        ):
            yield chunk
        self.usage.record(
            query=query, tier=clf.tier, confidence=clf.confidence,
            latency_ms=clf.latency_ms, model=target, provider=self.provider,
        )

    # ── Background atlas store ─────────────────────────────────────────────

    def _store_async(self, query, embedding, domain, tier, model, provider):
        if not self._embedder or not self.atlas or embedding is None:
            return None
        import uuid
        record_id = str(uuid.uuid4())
        atl, emb  = self.atlas, embedding

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

        threading.Thread(target=_store, daemon=True).start()
        return record_id

    # ── Helpers ────────────────────────────────────────────────────────────

    def _msgs(self, query, messages, system):
        if messages:
            return list(messages)
        if system and self.provider == "openai":
            return [{"role":"system","content":system},{"role":"user","content":query}]
        return [{"role":"user","content":query}]

    def _log(self, clf, dom, cache, conf, model, provider):
        if not self.verbose:
            return
        if cache:
            print(f"[ThinkRouter/CACHE] sim={cache.similarity:.4f} domain={cache.domain.value} "
                  f"tier={cache.tier.name} model={model}")
        else:
            dom_s  = f" domain={dom.domain.value}" if dom else ""
            risk_s = f" risk={conf.risk_score:.2f}({conf.recommendation.value})" if conf else ""
            print(f"[ThinkRouter] tier={clf.tier.name:<10} conf={clf.confidence:.3f}"
                  f"{dom_s}{risk_s} model={model} provider={provider}")

    def __repr__(self) -> str:
        s      = self.usage.summary()
        cache_ = f", hit_rate={self.cache.stats().hit_rate:.1f}%" if self.cache else ""
        atlas_ = f", atlas={len(self.atlas)}" if self.atlas else ""
        cost_  = ""
        if self.cost_tracker:
            cs = self.cost_tracker.summary()
            cost_ = f", saved=${cs.saved_usd:.4f}"
        return (
            f"ThinkRouter(provider={self.provider!r}, model={self.model!r}, "
            f"calls={s.total_calls}, savings={s.savings_pct:.1f}%"
            f"{atlas_}{cache_}{cost_})"
        )
