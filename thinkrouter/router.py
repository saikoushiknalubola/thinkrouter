"""
thinkrouter.router
~~~~~~~~~~~~~~~~~~
ThinkRouter — production routing layer with Phase 1 domain routing.

v0.4.0 additions:
  - Domain classifier integrated alongside complexity classifier
  - Model registry consulted to pick specialist model per domain
  - Ollama (local free models) supported as a provider
  - DomainResult included in RouterResponse
  - domain_routing flag (default True) enables/disables domain routing
"""
from __future__ import annotations

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
    domain_result    : Domain ClassifierResult (None if domain_routing=False).
    model_target     : ModelTarget used (None if domain_routing=False).
    raw              : Original provider response object.
    provider         : Provider used for this call.
    model            : Model identifier used.
    usage_tokens     : Token usage dict from provider.
    reasoning_effort : OpenAI reasoning_effort applied (o1/o3 only).
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
    reasoning_effort: Optional[str] = None
    thinking_budget:  Optional[int] = None

    def __repr__(self) -> str:
        dom = f", domain={self.domain_result.domain.value}" if self.domain_result else ""
        eff = f", reasoning_effort={self.reasoning_effort!r}" if self.reasoning_effort else ""
        bud = f", thinking_budget={self.thinking_budget}" if self.thinking_budget else ""
        return (
            f"RouterResponse("
            f"tier={self.routing.tier.name}"
            f"{dom}"
            f", budget={self.routing.token_budget} tokens"
            f"{eff}{bud}"
            f", model={self.model!r})"
        )


# ── ThinkRouter ───────────────────────────────────────────────────────────

class ThinkRouter:
    """
    Production-ready pre-inference routing layer — v0.4.0.

    Combines complexity routing (Tier classification) with domain routing
    (specialist model selection) for maximum quality and cost efficiency.

    Parameters
    ----------
    provider             : "openai" | "anthropic" | "ollama" | "generic"
    api_key              : Provider API key (not needed for ollama/generic).
    model                : Default model. Overridden by domain routing if enabled.
    classifier_backend   : "heuristic" | "distilbert"
    confidence_threshold : Min complexity confidence. Below → Tier.FULL.
    domain_routing       : Enable domain-aware model selection. Default: True.
    domain_min_confidence: Min domain confidence to override default model.
    preferred_provider   : Preferred provider for domain routing.
                           Default: "ollama" (free local models first).
    registry             : Custom ModelRegistry. Default: module-level registry.
    max_retries          : Retry attempts on transient errors.
    max_records          : Max call records in usage tracker.
    verbose              : Print routing decisions per call.
    ollama_url           : Ollama server URL. Default: http://localhost:11434
    config               : Override with custom Config instance.
    **client_kwargs      : Passed to provider SDK constructor.

    Examples
    --------
    # Domain routing with Ollama (free local models)
    >>> client = ThinkRouter(provider="ollama")
    >>> r = client.chat("Write a binary search tree in Python.")
    >>> r.domain_result.domain
    <Domain.CODE: 'code'>
    >>> r.model_target.model
    'deepseek-coder-v2'

    # Domain routing with OpenAI (uses best OpenAI model per domain)
    >>> client = ThinkRouter(provider="openai", preferred_provider="openai")
    >>> r = client.chat("Prove the Pythagorean theorem.")
    >>> r.domain_result.domain
    <Domain.MATH: 'math'>

    # Disable domain routing — complexity-only (v0.3 behaviour)
    >>> client = ThinkRouter(provider="openai", domain_routing=False)
    """

    _DEFAULT_MODELS: Dict[str, str] = {
        "openai":    "gpt-4o",
        "anthropic": "claude-sonnet-4-6",
        "ollama":    "llama3.1",
        "generic":   "unknown",
    }

    def __init__(
        self,
        provider:              ProviderLiteral = "openai",
        api_key:               Optional[str]   = None,
        model:                 Optional[str]   = None,
        classifier_backend:    str             = "heuristic",
        confidence_threshold:  float           = 0.75,
        domain_routing:        bool            = True,
        domain_min_confidence: float           = 0.45,
        preferred_provider:    Optional[str]   = None,
        registry:              Optional[ModelRegistry] = None,
        max_retries:           int             = 3,
        max_records:           int             = 10_000,
        verbose:               bool            = False,
        ollama_url:            str             = "http://localhost:11434",
        config:                Optional[Config] = None,
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

        # Provider adapter
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
            self._adapter = OpenAIAdapter(resolved_key, max_retries=max_retries, **client_kwargs)

        elif provider == "anthropic":
            if not resolved_key:
                raise ConfigurationError(
                    "No Anthropic API key. Pass api_key= or set ANTHROPIC_API_KEY."
                )
            self._adapter = AnthropicAdapter(resolved_key, max_retries=max_retries, **client_kwargs)

        elif provider == "ollama":
            from .ollama_adapter import OllamaAdapter
            self._adapter = OllamaAdapter(base_url=ollama_url)

        elif provider != "generic":
            raise ConfigurationError(
                f"Unknown provider: {provider!r}. "
                "Choose 'openai', 'anthropic', 'ollama', or 'generic'."
            )

        # Secondary Ollama adapter for domain routing fallback
        if domain_routing and provider != "ollama":
            try:
                from .ollama_adapter import OllamaAdapter
                self._ollama_adapter = OllamaAdapter(base_url=ollama_url)
            except Exception:
                pass

    # ── Classify only ─────────────────────────────────────────────────────

    def classify(self, query: str) -> ClassifierResult:
        return self._clf.predict(query)

    def classify_domain(self, query: str) -> DomainResult:
        return self._domain_clf.predict(query)

    def classify_batch(self, queries: List[str]) -> List[ClassifierResult]:
        return self._clf.predict_batch(queries)

    def classify_domain_batch(self, queries: List[str]) -> List[DomainResult]:
        return self._domain_clf.predict_batch(queries)

    def classify_full(self, query: str):
        """Return both complexity and domain results together."""
        return self._clf.predict(query), self._domain_clf.predict(query)

    # ── Sync chat ─────────────────────────────────────────────────────────

    def chat(
        self,
        query:       str,
        model:       Optional[str]             = None,
        messages:    Optional[List[Dict[str, str]]] = None,
        system:      Optional[str]             = None,
        temperature: float                     = 0.7,
        **extra:     Any,
    ) -> RouterResponse:
        clf_result  = self._clf.predict(query)
        domain_result: Optional[DomainResult] = None
        model_target:  Optional[ModelTarget]  = None
        target_model   = model or self.model
        target_provider = self.provider
        adapter        = self._adapter

        # Domain routing: pick specialist model if enabled
        if self.domain_routing:
            domain_result = self._domain_clf.predict(query)
            if domain_result.confidence >= self.domain_min_confidence:
                model_target = self._registry.resolve(
                    domain_result.domain,
                    preferred_provider=self._preferred_provider,
                )
                # Switch to Ollama adapter if domain routing selected ollama
                if model_target.provider == "ollama" and self.provider != "ollama":
                    if self._ollama_adapter and self._ollama_adapter.is_available():
                        adapter      = self._ollama_adapter
                        target_model = model_target.model
                        target_provider = "ollama"
                    else:
                        # Ollama not available — fall back to current provider's best
                        fallback = self._registry.resolve(
                            domain_result.domain,
                            preferred_provider=self.provider,
                        )
                        target_model = fallback.model
                elif model_target.provider == self.provider:
                    target_model = model_target.model

        self._log(clf_result, domain_result, target_model, target_provider)

        if adapter is None:
            raise ConfigurationError(
                "No provider adapter configured. "
                "Pass provider='openai', 'anthropic', or 'ollama'."
            )

        msg_list = self._msgs(query, messages, system)
        kw       = dict(extra)
        if self.provider == "anthropic" and system:
            kw["system"] = system

        # Call via Ollama adapter (simpler interface)
        if target_provider == "ollama":
            content, raw, usage_tokens = adapter.call(
                messages=msg_list, model=target_model,
                max_tokens=TIER_TOKEN_BUDGETS[clf_result.tier],
                temperature=temperature, **kw,
            )
            xparam = None
        else:
            content, raw, usage_tokens, xparam = adapter.call(
                messages=msg_list, model=target_model,
                tier=clf_result.tier, temperature=temperature, **kw,
            )

        self.usage.record(
            query=query, tier=clf_result.tier, confidence=clf_result.confidence,
            latency_ms=clf_result.latency_ms, model=target_model, provider=target_provider,
        )

        return RouterResponse(
            content=content, routing=clf_result,
            domain_result=domain_result, model_target=model_target,
            raw=raw, provider=target_provider, model=target_model,
            usage_tokens=usage_tokens,
            reasoning_effort=xparam if target_provider == "openai" else None,
            thinking_budget=xparam if target_provider == "anthropic" else None,
        )

    # ── Async chat ────────────────────────────────────────────────────────

    async def achat(
        self,
        query:       str,
        model:       Optional[str]             = None,
        messages:    Optional[List[Dict[str, str]]] = None,
        system:      Optional[str]             = None,
        temperature: float                     = 0.7,
        **extra:     Any,
    ) -> RouterResponse:
        clf_result    = self._clf.predict(query)
        domain_result = None
        model_target  = None
        target_model  = model or self.model
        target_provider = self.provider
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
                        adapter         = self._ollama_adapter
                        target_model    = model_target.model
                        target_provider = "ollama"
                elif model_target.provider == self.provider:
                    target_model = model_target.model

        self._log(clf_result, domain_result, target_model, target_provider)

        if adapter is None:
            raise ConfigurationError("No provider adapter configured.")

        msg_list = self._msgs(query, messages, system)
        kw       = dict(extra)
        if self.provider == "anthropic" and system:
            kw["system"] = system

        if target_provider == "ollama":
            content, raw, usage_tokens = await adapter.acall(
                messages=msg_list, model=target_model,
                max_tokens=TIER_TOKEN_BUDGETS[clf_result.tier],
                temperature=temperature, **kw,
            )
            xparam = None
        else:
            content, raw, usage_tokens, xparam = await adapter.acall(
                messages=msg_list, model=target_model,
                tier=clf_result.tier, temperature=temperature, **kw,
            )

        self.usage.record(
            query=query, tier=clf_result.tier, confidence=clf_result.confidence,
            latency_ms=clf_result.latency_ms, model=target_model, provider=target_provider,
        )

        return RouterResponse(
            content=content, routing=clf_result,
            domain_result=domain_result, model_target=model_target,
            raw=raw, provider=target_provider, model=target_model,
            usage_tokens=usage_tokens,
            reasoning_effort=xparam if target_provider == "openai" else None,
            thinking_budget=xparam if target_provider == "anthropic" else None,
        )

    # ── Stream ────────────────────────────────────────────────────────────

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

    # ── Helpers ───────────────────────────────────────────────────────────

    def _msgs(self, query, messages, system):
        if messages:
            return list(messages)
        if system and self.provider == "openai":
            return [{"role":"system","content":system},{"role":"user","content":query}]
        return [{"role":"user","content":query}]

    def _log(self, clf, dom, model, provider):
        if not self.verbose:
            return
        dom_str = f"  domain={dom.domain.value}  conf={dom.confidence:.2f}" if dom else ""
        print(
            f"[ThinkRouter] tier={clf.tier.name:<10} "
            f"conf={clf.confidence:.3f}  "
            f"clf={clf.latency_ms:.2f}ms"
            f"{dom_str}  "
            f"model={model}  provider={provider}"
        )

    def __repr__(self) -> str:
        s = self.usage.summary()
        return (
            f"ThinkRouter("
            f"provider={self.provider!r}, "
            f"model={self.model!r}, "
            f"domain_routing={self.domain_routing}, "
            f"calls={s.total_calls}, "
            f"savings={s.savings_pct:.1f}%)"
        )
