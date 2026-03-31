"""
thinkrouter.router
~~~~~~~~~~~~~~~~~~
ThinkRouter — the main class developers interact with.

Intercepts every chat call, classifies difficulty in <1ms, applies
the provider-native reasoning budget control, and forwards to the
underlying API.

What "budget control" actually means per provider:
  OpenAI o1/o3    → reasoning_effort="low" or "high"
  Anthropic think → thinking={"type":"enabled","budget_tokens":N}
  Standard models → max_tokens + concise-answer system hint
"""
from __future__ import annotations

import os
from dataclasses import dataclass
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
from .exceptions import ConfigurationError
from .providers import AnthropicAdapter, OpenAIAdapter
from .usage import UsageTracker


# ── Response container ────────────────────────────────────────────────────

@dataclass
class RouterResponse:
    """
    Unified response from ThinkRouter regardless of provider.

    Attributes
    ----------
    content          : Generated text.
    routing          : ClassifierResult — the routing decision applied.
    raw              : Original provider SDK response object.
    provider         : "openai" | "anthropic"
    model            : Model identifier used.
    usage_tokens     : {"prompt_tokens", "completion_tokens", "total_tokens"}
    reasoning_effort : OpenAI reasoning_effort applied (o1/o3 only).
    thinking_budget  : Anthropic thinking budget_tokens applied (Claude only).
    """
    content:          str
    routing:          ClassifierResult
    raw:              Any
    provider:         str
    model:            str
    usage_tokens:     Dict[str, int]
    reasoning_effort: Optional[str] = None
    thinking_budget:  Optional[int] = None

    def __repr__(self) -> str:
        extras = []
        if self.reasoning_effort:
            extras.append(f"reasoning_effort={self.reasoning_effort!r}")
        if self.thinking_budget is not None:
            extras.append(f"thinking_budget={self.thinking_budget}")
        extra_str = (", " + ", ".join(extras)) if extras else ""
        return (
            f"RouterResponse("
            f"tier={self.routing.tier.name}, "
            f"budget={self.routing.token_budget} tokens"
            f"{extra_str}, "
            f"model={self.model!r})"
        )


# ── ThinkRouter ───────────────────────────────────────────────────────────

class ThinkRouter:
    """
    Production-ready pre-inference routing layer.

    Parameters
    ----------
    provider             : "openai" | "anthropic" | "generic"
    api_key              : Provider key. Falls back to env var.
    model                : Default model for all calls.
    classifier_backend   : "heuristic" (default) or "distilbert"
    confidence_threshold : Min classifier confidence to accept routing.
                           Queries below this fall back to Tier.FULL.
    max_retries          : Retry attempts on rate limits / 5xx errors.
    max_records          : Max call records in the usage tracker.
    verbose              : Print routing decision per call.
    config               : Override with a custom Config instance.
    **client_kwargs      : Passed to the provider SDK client constructor.

    Examples
    --------
    # OpenAI o1 — reasoning_effort auto-set per query
    >>> client = ThinkRouter(provider="openai", model="o1")
    >>> r = client.chat("What is 7 * 8?")
    >>> r.reasoning_effort
    'low'

    # Anthropic extended thinking
    >>> client = ThinkRouter(provider="anthropic", model="claude-opus-4-6")
    >>> r = client.chat("Prove sqrt(2) is irrational.")
    >>> r.thinking_budget
    10000

    # Async
    >>> r = await client.achat("Explain gradient descent.")

    # Classify only — no API call
    >>> client.classify("Write a quicksort.")
    ClassifierResult(tier=FULL, confidence=0.87, budget=8000 tokens, latency=0.4ms)

    # CLI shortcut
    # thinkrouter classify "What is 2+3?"
    # thinkrouter demo
    """

    _DEFAULT_MODELS: Dict[str, str] = {
        "openai":    "gpt-4o",
        "anthropic": "claude-sonnet-4-6",
        "generic":   "unknown",
    }

    def __init__(
        self,
        provider:             ProviderLiteral = "openai",
        api_key:              Optional[str]   = None,
        model:                Optional[str]   = None,
        classifier_backend:   str             = "heuristic",
        confidence_threshold: float           = 0.75,
        max_retries:          int             = 3,
        max_records:          int             = 10_000,
        verbose:              bool            = False,
        config:               Optional[Config] = None,
        **client_kwargs:      Any,
    ) -> None:
        cfg = config or DEFAULT_CONFIG

        self.provider    = provider
        self.model       = model or self._DEFAULT_MODELS.get(provider, "unknown")
        self.verbose     = verbose or cfg.verbose
        self.max_retries = max_retries

        # Classifier
        clf_kw: Dict[str, Any] = {}
        if classifier_backend == "distilbert":
            clf_kw["threshold"]  = confidence_threshold
            clf_kw["model_name"] = cfg.hf_model
        self._clf: BaseClassifier = get_classifier(classifier_backend, **clf_kw)
        self._threshold = confidence_threshold

        # Usage tracker
        self.usage = UsageTracker(max_records=max_records)

        # Provider adapter
        self._adapter: Optional[Any] = None
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
        elif provider == "generic":
            pass  # classify-only mode
        else:
            raise ConfigurationError(
                f"Unknown provider: {provider!r}. Choose 'openai', 'anthropic', or 'generic'."
            )

    # ── Classify (no API call) ─────────────────────────────────────────────

    def classify(self, query: str) -> ClassifierResult:
        """Classify a single query without making any API call."""
        return self._clf.predict(query)

    def classify_batch(self, queries: List[str]) -> List[ClassifierResult]:
        """Classify a list of queries without making any API calls."""
        return self._clf.predict_batch(queries)

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
        """
        Route and execute a single chat completion.

        Parameters
        ----------
        query       : User query. Used for classification and as the user
                      message when `messages` is not provided.
        model       : Override the default model for this call.
        messages    : Full message history. If supplied, `query` is used
                      only for classification.
        system      : System prompt.
        temperature : Sampling temperature (ignored for o1/o3).
        **extra     : Forwarded to the provider API.
        """
        if self._adapter is None:
            raise ConfigurationError(
                "No provider adapter. Pass provider='openai' or 'anthropic'."
            )
        clf    = self._clf.predict(query)
        target = model or self.model
        self._log(clf, target)

        msg_list = self._msgs(query, messages, system)
        kw       = dict(extra)
        if self.provider == "anthropic" and system:
            kw["system"] = system

        content, raw, usage, xparam = self._adapter.call(
            messages=msg_list, model=target,
            tier=clf.tier, temperature=temperature, **kw,
        )
        self.usage.record(
            query=query, tier=clf.tier, confidence=clf.confidence,
            latency_ms=clf.latency_ms, model=target, provider=self.provider,
        )
        return self._response(content, clf, raw, target, usage, xparam)

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
        """Async version of chat(). Use inside asyncio / FastAPI."""
        if self._adapter is None:
            raise ConfigurationError("No provider adapter.")

        clf    = self._clf.predict(query)
        target = model or self.model
        self._log(clf, target)

        msg_list = self._msgs(query, messages, system)
        kw       = dict(extra)
        if self.provider == "anthropic" and system:
            kw["system"] = system

        content, raw, usage, xparam = await self._adapter.acall(
            messages=msg_list, model=target,
            tier=clf.tier, temperature=temperature, **kw,
        )
        self.usage.record(
            query=query, tier=clf.tier, confidence=clf.confidence,
            latency_ms=clf.latency_ms, model=target, provider=self.provider,
        )
        return self._response(content, clf, raw, target, usage, xparam)

    # ── Sync stream ───────────────────────────────────────────────────────

    def stream(
        self,
        query:       str,
        model:       Optional[str] = None,
        system:      Optional[str] = None,
        temperature: float         = 0.7,
        **extra:     Any,
    ) -> Iterator[str]:
        """Stream response tokens. Routing applied before first token."""
        if self._adapter is None:
            raise ConfigurationError("No provider adapter.")

        clf      = self._clf.predict(query)
        target   = model or self.model
        msg_list = self._msgs(query, None, system)
        kw       = dict(extra)
        if self.provider == "anthropic" and system:
            kw["system"] = system

        self._log(clf, target)
        yield from self._adapter.stream(
            messages=msg_list, model=target,
            tier=clf.tier, temperature=temperature, **kw,
        )
        self.usage.record(
            query=query, tier=clf.tier, confidence=clf.confidence,
            latency_ms=clf.latency_ms, model=target, provider=self.provider,
        )

    # ── Async stream ──────────────────────────────────────────────────────

    async def astream(
        self,
        query:       str,
        model:       Optional[str] = None,
        system:      Optional[str] = None,
        temperature: float         = 0.7,
        **extra:     Any,
    ) -> AsyncIterator[str]:
        """Async streaming. Use in FastAPI / async frameworks."""
        if self._adapter is None:
            raise ConfigurationError("No provider adapter.")

        clf      = self._clf.predict(query)
        target   = model or self.model
        msg_list = self._msgs(query, None, system)
        kw       = dict(extra)
        if self.provider == "anthropic" and system:
            kw["system"] = system

        self._log(clf, target)
        async for chunk in self._adapter.astream(
            messages=msg_list, model=target,
            tier=clf.tier, temperature=temperature, **kw,
        ):
            yield chunk
        self.usage.record(
            query=query, tier=clf.tier, confidence=clf.confidence,
            latency_ms=clf.latency_ms, model=target, provider=self.provider,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _msgs(
        self,
        query:    str,
        messages: Optional[List[Dict[str, str]]],
        system:   Optional[str],
    ) -> List[Dict[str, str]]:
        if messages:
            return list(messages)
        if system and self.provider == "openai":
            return [
                {"role": "system", "content": system},
                {"role": "user",   "content": query},
            ]
        return [{"role": "user", "content": query}]

    def _response(
        self,
        content:  str,
        clf:      ClassifierResult,
        raw:      Any,
        model:    str,
        usage:    Dict[str, int],
        xparam:   Any,
    ) -> RouterResponse:
        return RouterResponse(
            content=content, routing=clf, raw=raw,
            provider=self.provider, model=model,
            usage_tokens=usage,
            reasoning_effort=xparam if self.provider == "openai" else None,
            thinking_budget=xparam if self.provider == "anthropic" else None,
        )

    def _log(self, clf: ClassifierResult, model: str) -> None:
        if not self.verbose:
            return
        extra = ""
        if self.provider == "openai" and model in OPENAI_REASONING_MODELS:
            extra = f"  reasoning_effort={OPENAI_REASONING_EFFORT[clf.tier]!r}"
        elif self.provider == "anthropic" and model in ANTHROPIC_THINKING_MODELS:
            b     = ANTHROPIC_THINKING_BUDGETS[clf.tier]
            extra = f"  thinking_budget={b}"
        print(
            f"[ThinkRouter] tier={clf.tier.name:<10} "
            f"conf={clf.confidence:.3f}  "
            f"clf={clf.latency_ms:.2f}ms  "
            f"model={model}{extra}"
        )

    def __repr__(self) -> str:
        s = self.usage.summary()
        return (
            f"ThinkRouter("
            f"provider={self.provider!r}, "
            f"model={self.model!r}, "
            f"calls={s.total_calls}, "
            f"savings={s.savings_pct:.1f}%)"
        )
