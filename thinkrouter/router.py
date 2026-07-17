"""
thinkrouter.router
~~~~~~~~~~~~~~~~~~
ThinkRouter v0.5.0 — Phase 2: Embedding Layer integrated.

Every routed query is now automatically embedded and stored in the
atlas after inference. The atlas grows silently in the background
without adding any latency to the inference call.

New in v0.5.0:
  - Atlas integration: auto-store (embedding, domain, tier, model, quality)
  - Embedder integration: configurable backend (hash/openai/local)
  - RouterResponse.record_id: atlas record UUID for quality feedback
  - ThinkRouter.update_quality(): update quality score post-inference
  - ThinkRouter.atlas: direct atlas access for stats / similar queries
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

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
    Unified response from ThinkRouter.

    Attributes
    ----------
    content          : Generated text.
    routing          : Complexity ClassifierResult.
    domain_result    : Domain DomainResult.
    model_target     : ModelTarget selected.
    raw              : Original provider response.
    provider         : Provider used.
    model            : Model identifier used.
    usage_tokens     : Token usage dict.
    record_id        : Atlas record UUID — use to update quality score.
    reasoning_effort : OpenAI reasoning_effort applied.
    thinking_budget  : Anthropic thinking budget_tokens applied.
    """
    content:          str
    routing:          ClassifierResult
    domain_result:    Optional[DomainResult]
    model_target:     Optional[ModelTarget]
    raw:              Any
    provider:         str
    model:            str
    usage_tokens:     Dict[str, int]
    record_id:        Optional[str] = None
    reasoning_effort: Optional[str] = None
    thinking_budget:  Optional[int] = None

    def __repr__(self) -> str:
        dom = f", domain={self.domain_result.domain.value}" if self.domain_result else ""
        return (
            f"RouterResponse("
            f"tier={self.routing.tier.name}"
            f"{dom}"
            f", budget={self.routing.token_budget} tokens"
            f", model={self.model!r}"
            f", record_id={self.record_id!r})"
        )


# ── ThinkRouter ───────────────────────────────────────────────────────────

class ThinkRouter:
    """
    Production-ready pre-inference routing layer — v0.5.0.

    Adds Phase 2 embedding layer on top of Phase 1 domain routing:
    every routed query is embedded and stored in the local atlas
    after inference, in a background thread (zero latency overhead).

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
    atlas_enabled         : Enable atlas storage. Default: True.
    embedder_backend      : "hash" | "openai" | "local"
    embedder_kwargs       : Passed to get_embedder().
    max_retries           : Retry attempts on transient errors.
    max_records           : Max usage tracker records.
    verbose               : Print routing decision per call.
    ollama_url            : Ollama server URL.
    config                : Override with custom Config.
    **client_kwargs       : Passed to provider SDK constructor.

    Examples
    --------
    # Full Phase 2 — domain routing + atlas storage
    >>> client = ThinkRouter(provider="openai", atlas_enabled=True)
    >>> r = client.chat("Write a binary search tree in Python.")
    >>> print(r.record_id)   # atlas UUID
    >>> client.update_quality(r.record_id, 0.95)  # after reviewing response

    # Check atlas growth
    >>> client.atlas.print_stats()

    # Find semantically similar past queries (Phase 3 preview)
    >>> vec = client._embedder.embed("Implement a linked list in Python.")
    >>> similar = client.atlas.find_similar(vec, k=5, min_score=0.85)
    """

    _DEFAULT_MODELS: Dict[str, str] = {
        "openai":    "gpt-4o",
        "anthropic": "claude-sonnet-4-6",
        "ollama":    "llama3.1",
        "generic":   "unknown",
    }

    def __init__(
        self,
        provider:              ProviderLiteral     = "openai",
        api_key:               Optional[str]       = None,
        model:                 Optional[str]       = None,
        classifier_backend:    str                 = "heuristic",
        confidence_threshold:  float               = 0.75,
        domain_routing:        bool                = True,
        domain_min_confidence: float               = 0.45,
        preferred_provider:    Optional[str]       = None,
        registry:              Optional[ModelRegistry] = None,
        atlas_enabled:         Optional[bool]      = None,
        embedder_backend:      Optional[str]       = None,
        embedder_kwargs:       Optional[Dict]      = None,
        max_retries:           int                 = 3,
        max_records:           int                 = 10_000,
        verbose:               bool                = False,
        ollama_url:            str                 = "http://localhost:11434",
        config:                Optional[Config]    = None,
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

        # ── Phase 2: Embedder & Atlas ──────────────────────────────────
        _atlas_enabled  = atlas_enabled if atlas_enabled is not None else cfg.atlas_enabled
        _emb_backend    = embedder_backend or cfg.embedder_backend
        _emb_kwargs     = embedder_kwargs or {}

        self._embedder = None
        self.atlas     = None

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
            except Exception:
                # Atlas init failure is non-fatal — routing still works
                self._embedder = None
                self.atlas     = None

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

    # ── Public classify API ────────────────────────────────────────────────

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

    # ── Quality feedback (Phase 2) ─────────────────────────────────────────

    def update_quality(self, record_id: str, quality_score: float) -> None:
        """
        Update the quality score for a past routing decision.

        Call this after reviewing the model's response — manually or
        via an automated judge. Quality scores are stored in the atlas
        and power Phase 4 confidence modelling.

        Parameters
        ----------
        record_id     : UUID from RouterResponse.record_id
        quality_score : Score in [0, 1].
        """
        if self.atlas and record_id:
            self.atlas.update_quality(record_id, quality_score)

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

        clf           = self._clf.predict(query)
        domain_result = None
        model_target  = None
        target_model  = model or self.model
        target_prov   = self.provider
        adapter       = self._adapter

        if self.domain_routing:
            domain_result = self._domain_clf.predict(query)
            if domain_result.confidence >= self.domain_min_confidence:
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

        self._log(clf, domain_result, target_model, target_prov)

        msg_list = self._msgs(query, messages, system)
        kw       = dict(extra)
        if self.provider == "anthropic" and system:
            kw["system"] = system

        if target_prov == "ollama":
            content, raw, usage_tokens = adapter.call(
                messages=msg_list, model=target_model,
                max_tokens=TIER_TOKEN_BUDGETS[clf.tier],
                temperature=temperature, **kw,
            )
            xparam = None
        else:
            content, raw, usage_tokens, xparam = adapter.call(
                messages=msg_list, model=target_model,
                tier=clf.tier, temperature=temperature, **kw,
            )

        self.usage.record(
            query=query, tier=clf.tier, confidence=clf.confidence,
            latency_ms=clf.latency_ms, model=target_model, provider=target_prov,
        )

        # Phase 2: embed and store in atlas (background thread, zero latency)
        record_id = self._store_async(
            query=query, domain_result=domain_result,
            clf=clf, model=target_model, provider=target_prov,
        )

        return RouterResponse(
            content=content, routing=clf,
            domain_result=domain_result, model_target=model_target,
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

        clf           = self._clf.predict(query)
        domain_result = None
        model_target  = None
        target_model  = model or self.model
        target_prov   = self.provider
        adapter       = self._adapter

        if self.domain_routing:
            domain_result = self._domain_clf.predict(query)
            if domain_result.confidence >= self.domain_min_confidence:
                model_target = self._registry.resolve(
                    domain_result.domain,
                    preferred_provider=self._preferred_provider,
                )
                if model_target.provider == "ollama" and self.provider != "ollama":
                    if self._ollama_adapter:
                        adapter      = self._ollama_adapter
                        target_model = model_target.model
                        target_prov  = "ollama"
                elif model_target.provider == self.provider:
                    target_model = model_target.model

        self._log(clf, domain_result, target_model, target_prov)

        msg_list = self._msgs(query, messages, system)
        kw       = dict(extra)
        if self.provider == "anthropic" and system:
            kw["system"] = system

        if target_prov == "ollama":
            content, raw, usage_tokens = await adapter.acall(
                messages=msg_list, model=target_model,
                max_tokens=TIER_TOKEN_BUDGETS[clf.tier],
                temperature=temperature, **kw,
            )
            xparam = None
        else:
            content, raw, usage_tokens, xparam = await adapter.acall(
                messages=msg_list, model=target_model,
                tier=clf.tier, temperature=temperature, **kw,
            )

        self.usage.record(
            query=query, tier=clf.tier, confidence=clf.confidence,
            latency_ms=clf.latency_ms, model=target_model, provider=target_prov,
        )

        record_id = self._store_async(
            query=query, domain_result=domain_result,
            clf=clf, model=target_model, provider=target_prov,
        )

        return RouterResponse(
            content=content, routing=clf,
            domain_result=domain_result, model_target=model_target,
            raw=raw, provider=target_prov, model=target_model,
            usage_tokens=usage_tokens, record_id=record_id,
            reasoning_effort=xparam if target_prov == "openai" else None,
            thinking_budget=xparam if target_prov == "anthropic" else None,
        )

    # ── Stream ─────────────────────────────────────────────────────────────

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
        clf      = self._clf.predict(query)
        target   = model or self.model
        msg_list = self._msgs(query, None, system)
        kw       = dict(extra)
        if self.provider == "anthropic" and system:
            kw["system"] = system
        self._log(clf, None, target, self.provider)
        yield from self._adapter.stream(
            messages=msg_list, model=target,
            tier=clf.tier, temperature=temperature, **kw,
        )
        self.usage.record(
            query=query, tier=clf.tier, confidence=clf.confidence,
            latency_ms=clf.latency_ms, model=target, provider=self.provider,
        )
        self._store_async(query=query, domain_result=None,
                          clf=clf, model=target, provider=self.provider)

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
        clf      = self._clf.predict(query)
        target   = model or self.model
        msg_list = self._msgs(query, None, system)
        kw       = dict(extra)
        if self.provider == "anthropic" and system:
            kw["system"] = system
        self._log(clf, None, target, self.provider)
        async for chunk in self._adapter.astream(
            messages=msg_list, model=target,
            tier=clf.tier, temperature=temperature, **kw,
        ):
            yield chunk
        self.usage.record(
            query=query, tier=clf.tier, confidence=clf.confidence,
            latency_ms=clf.latency_ms, model=target, provider=self.provider,
        )
        self._store_async(query=query, domain_result=None,
                          clf=clf, model=target, provider=self.provider)

    # ── Phase 2: background atlas storage ─────────────────────────────────

    def _store_async(
        self,
        query:         str,
        domain_result: Optional[DomainResult],
        clf:           ClassifierResult,
        model:         str,
        provider:      str,
    ) -> Optional[str]:
        """
        Embed the query and store in atlas — runs in a daemon thread.
        Returns the record_id immediately (the thread runs in background).
        Returns None if atlas is disabled or embedder is unavailable.
        """
        if not self._embedder or not self.atlas:
            return None

        import uuid
        record_id = str(uuid.uuid4())

        domain = (
            domain_result.domain
            if domain_result and domain_result.confidence >= self.domain_min_confidence
            else Domain.GENERAL
        )

        embedder = self._embedder
        atlas    = self.atlas

        def _store():
            try:
                vec = embedder.embed(query)
                atlas.store(
                    query=query,
                    embedding=vec,
                    domain=domain,
                    tier=clf.tier,
                    model=model,
                    provider=provider,
                    quality_score=None,   # updated later via update_quality()
                    latency_ms=clf.latency_ms,
                )
            except Exception:
                pass  # atlas failure is always non-fatal

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

    def _log(self, clf, dom, model, provider):
        if not self.verbose:
            return
        dom_str = f"  domain={dom.domain.value}({dom.confidence:.2f})" if dom else ""
        atlas_str = f"  atlas={len(self.atlas)}" if self.atlas else ""
        print(
            f"[ThinkRouter] tier={clf.tier.name:<10} "
            f"conf={clf.confidence:.3f}  clf={clf.latency_ms:.2f}ms"
            f"{dom_str}  model={model}  provider={provider}{atlas_str}"
        )

    def __repr__(self) -> str:
        s = self.usage.summary()
        atlas_str = f", atlas={len(self.atlas)}" if self.atlas is not None else ""
        return (
            f"ThinkRouter("
            f"provider={self.provider!r}, "
            f"model={self.model!r}, "
            f"domain_routing={self.domain_routing}, "
            f"calls={s.total_calls}, "
            f"savings={s.savings_pct:.1f}%"
            f"{atlas_str})"
        )
