"""
thinkrouter.router
~~~~~~~~~~~~~~~~~~

ThinkRouter — the core class.

Drop-in replacement for openai.OpenAI and anthropic.Anthropic clients.
Intercepts every chat call, classifies difficulty in <5ms, applies the
minimum thinking-token budget, and forwards to the provider.

Quick start::

    from thinkrouter import ThinkRouter

    # OpenAI
    client   = ThinkRouter(provider="openai", api_key="sk-...")
    response = client.chat("What is the capital of France?")
    print(response.content)    # "The capital of France is Paris."
    print(response.routing)    # ClassifierResult(tier=NO_THINK, budget=50 tokens, ...)

    # Anthropic
    client   = ThinkRouter(provider="anthropic", api_key="sk-ant-...")
    response = client.chat("Prove that sqrt(2) is irrational.")
    print(response.routing)    # ClassifierResult(tier=FULL, budget=8000 tokens, ...)

    # Classify without an API call
    result   = client.classify("Design a distributed caching system.")
    print(result.tier)         # Tier.FULL

    # Usage dashboard
    client.usage.print_dashboard()
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from .classifier import BaseClassifier, ClassifierResult, get_classifier
from .constants import TIER_TOKEN_BUDGETS, ProviderLiteral
from .usage import UsageTracker


# ─── Response container ───────────────────────────────────────────────────────

@dataclass
class RouterResponse:
    """
    Unified response from ThinkRouter.chat() regardless of provider.

    Attributes
    ----------
    content      : Generated text content.
    routing      : ClassifierResult that determined the compute budget.
    raw          : Original response object from the provider SDK.
    provider     : "openai" | "anthropic" | "generic"
    model        : Model identifier used for the request.
    usage_tokens : {"prompt_tokens": N, "completion_tokens": M, "total_tokens": K}
    """
    content:      str
    routing:      ClassifierResult
    raw:          Any
    provider:     str
    model:        str
    usage_tokens: Dict[str, int]

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return (
            f"RouterResponse("
            f"tier={self.routing.tier.name}, "
            f"budget={self.routing.token_budget} tokens, "
            f"model={self.model!r}, "
            f"preview={preview!r})"
        )


# ─── Provider adapters ────────────────────────────────────────────────────────

class _OpenAIAdapter:
    """Wraps openai.OpenAI — install with pip install thinkrouter[openai]."""

    def __init__(self, api_key: Optional[str], **kwargs: Any) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenAI provider requires the openai package.\n"
                "Install with:  pip install thinkrouter[openai]"
            ) from exc
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "No OpenAI API key found. Pass api_key= or set OPENAI_API_KEY."
            )
        self._client = openai.OpenAI(api_key=key, **kwargs)

    def call(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        max_tokens:  int,
        temperature: float,
        **extra:     Any,
    ) -> tuple:
        resp    = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            **extra,
        )
        content = resp.choices[0].message.content or ""
        usage   = {
            "prompt_tokens":     getattr(resp.usage, "prompt_tokens",     0),
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
            "total_tokens":      getattr(resp.usage, "total_tokens",      0),
        }
        return content, resp, usage

    def stream(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        max_tokens:  int,
        temperature: float,
        **extra:     Any,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **extra,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class _AnthropicAdapter:
    """Wraps anthropic.Anthropic — install with pip install thinkrouter[anthropic]."""

    def __init__(self, api_key: Optional[str], **kwargs: Any) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic provider requires the anthropic package.\n"
                "Install with:  pip install thinkrouter[anthropic]"
            ) from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No Anthropic API key found. Pass api_key= or set ANTHROPIC_API_KEY."
            )
        self._client = anthropic.Anthropic(api_key=key, **kwargs)

    def call(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        max_tokens:  int,
        temperature: float,
        system:      Optional[str] = None,
        **extra:     Any,
    ) -> tuple:
        kwargs: Dict[str, Any] = dict(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature, **extra,
        )
        if system:
            kwargs["system"] = system
        resp    = self._client.messages.create(**kwargs)
        content = resp.content[0].text if resp.content else ""
        usage   = {
            "prompt_tokens":     resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens":      resp.usage.input_tokens + resp.usage.output_tokens,
        }
        return content, resp, usage

    def stream(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        max_tokens:  int,
        temperature: float,
        system:      Optional[str] = None,
        **extra:     Any,
    ) -> Iterator[str]:
        kwargs: Dict[str, Any] = dict(**extra)
        if system:
            kwargs["system"] = system
        with self._client.messages.stream(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature, **kwargs,
        ) as s:
            yield from s.text_stream


# ─── ThinkRouter ─────────────────────────────────────────────────────────────

class ThinkRouter:
    """
    Pre-inference routing layer for LLM reasoning models.

    Parameters
    ----------
    provider             : "openai" | "anthropic" | "generic"
    api_key              : Provider key. Falls back to env var if omitted.
    model                : Default model for all chat() calls.
    classifier_backend   : "heuristic" (default, zero deps) or "distilbert"
    confidence_threshold : Min classifier confidence to accept routing decision.
                           Queries below this fall back to Tier.FULL (safe).
    max_records          : Max call records retained in the usage tracker.
    verbose              : Print routing decision for every call when True.
    **client_kwargs      : Passed directly to the provider client constructor.

    Examples
    --------
    >>> client = ThinkRouter(provider="openai")
    >>> response = client.chat("What is 7 * 8?")
    >>> response.routing.tier
    <Tier.NO_THINK: 0>
    >>> response.routing.token_budget
    50
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
        max_records:          int             = 10_000,
        verbose:              bool            = False,
        **client_kwargs:      Any,
    ) -> None:
        self.provider  = provider
        self.model     = model or self._DEFAULT_MODELS.get(provider, "unknown")
        self.verbose   = verbose

        # Classifier
        clf_kwargs: Dict[str, Any] = {}
        if classifier_backend == "distilbert":
            clf_kwargs["threshold"] = confidence_threshold
        self._classifier: BaseClassifier = get_classifier(
            classifier_backend, **clf_kwargs
        )
        self._threshold = confidence_threshold

        # Usage tracker
        self.usage = UsageTracker(max_records=max_records)

        # Provider adapter
        self._adapter: Optional[Any] = None
        if provider == "openai":
            self._adapter = _OpenAIAdapter(api_key=api_key, **client_kwargs)
        elif provider == "anthropic":
            self._adapter = _AnthropicAdapter(api_key=api_key, **client_kwargs)
        elif provider != "generic":
            raise ValueError(
                f"Unknown provider: {provider!r}. "
                "Choose 'openai', 'anthropic', or 'generic'."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, query: str) -> ClassifierResult:
        """
        Classify a single query without making an API call.
        Useful for inspecting routing decisions during development.
        """
        return self._classifier.predict(query)

    def classify_batch(self, queries: List[str]) -> List[ClassifierResult]:
        """Classify a list of queries without making API calls."""
        return self._classifier.predict_batch(queries)

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
        query       : User query text. Used for classification and as the
                      user message when `messages` is not supplied.
        model       : Override the default model for this call.
        messages    : Full message history. If supplied, `query` is used
                      only for classification.
        system      : System prompt.
        temperature : Sampling temperature. Default 0.7.
        **extra     : Forwarded to the provider API.

        Returns
        -------
        RouterResponse
        """
        if self._adapter is None:
            raise RuntimeError(
                "No provider configured. Pass provider='openai' or "
                "provider='anthropic' to ThinkRouter()."
            )

        clf    = self._classifier.predict(query)
        budget = TIER_TOKEN_BUDGETS[clf.tier]

        if self.verbose:
            print(
                f"[ThinkRouter] tier={clf.tier.name:<10} "
                f"conf={clf.confidence:.3f}  "
                f"budget={budget:,} tokens  "
                f"clf={clf.latency_ms:.1f}ms"
            )

        # Build message list
        target  = model or self.model
        msg_list = messages or []
        if not messages:
            if system and self.provider == "openai":
                msg_list = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": query},
                ]
            else:
                msg_list = [{"role": "user", "content": query}]

        call_kw: Dict[str, Any] = dict(extra)
        if self.provider == "anthropic" and system:
            call_kw["system"] = system

        content, raw, usage_tokens = self._adapter.call(
            messages=msg_list, model=target,
            max_tokens=budget, temperature=temperature, **call_kw,
        )

        self.usage.record(
            query=query, tier=clf.tier,
            confidence=clf.confidence, latency_ms=clf.latency_ms,
        )

        return RouterResponse(
            content=content, routing=clf, raw=raw,
            provider=self.provider, model=target,
            usage_tokens=usage_tokens,
        )

    def stream(
        self,
        query:       str,
        model:       Optional[str] = None,
        system:      Optional[str] = None,
        temperature: float         = 0.7,
        **extra:     Any,
    ) -> Iterator[str]:
        """
        Stream a routed response token-by-token.

        The routing decision is made before the first token is requested,
        so the budget applies to the entire generation.

        Yields
        ------
        str   Text delta chunks as they arrive.

        Example
        -------
        >>> for chunk in client.stream("Explain how TCP works."):
        ...     print(chunk, end="", flush=True)
        """
        if self._adapter is None:
            raise RuntimeError("No provider configured.")

        clf    = self._classifier.predict(query)
        budget = TIER_TOKEN_BUDGETS[clf.tier]

        if self.verbose:
            print(
                f"[ThinkRouter/stream] tier={clf.tier.name}  "
                f"budget={budget:,} tokens"
            )

        msg_list: List[Dict[str, str]] = []
        if system and self.provider == "openai":
            msg_list.append({"role": "system", "content": system})
        msg_list.append({"role": "user", "content": query})

        call_kw: Dict[str, Any] = dict(extra)
        if self.provider == "anthropic" and system:
            call_kw["system"] = system

        yield from self._adapter.stream(
            messages=msg_list, model=model or self.model,
            max_tokens=budget, temperature=temperature, **call_kw,
        )

        self.usage.record(
            query=query, tier=clf.tier,
            confidence=clf.confidence, latency_ms=clf.latency_ms,
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
