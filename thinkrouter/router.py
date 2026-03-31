"""
thinkrouter.router
~~~~~~~~~~~~~~~~~~

ThinkRouter — production-ready routing layer.

KEY FIX over v0.1.x:
  - OpenAI o1/o3 models: uses `reasoning_effort` ("low"/"high") to control
    actual thinking compute, not just max_completion_tokens.
  - Anthropic Claude: uses `thinking={"type":"enabled","budget_tokens":N}`
    to control actual thinking budget directly via the API.
  - Standard models (gpt-4o, claude-3-5-haiku): routes by limiting
    max_tokens and using system-prompt budget hints.
  - Retry with exponential backoff on rate limits and transient errors.
  - Full async support via achat() and astream().

Quick start::

    from thinkrouter import ThinkRouter

    # OpenAI o1 — reasoning_effort controlled per query
    client   = ThinkRouter(provider="openai", model="o1")
    response = client.chat("What is the capital of France?")
    # reasoning_effort="low" applied — minimal thinking tokens used

    # Anthropic Claude with extended thinking
    client   = ThinkRouter(provider="anthropic", model="claude-opus-4-6")
    response = client.chat("Prove that sqrt(2) is irrational.")
    # thinking budget_tokens=10000 applied automatically

    # Async
    response = await client.achat("Explain merge sort.")

    # Savings dashboard
    client.usage.print_dashboard()
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from .classifier import BaseClassifier, ClassifierResult, get_classifier
from .constants import (
    ANTHROPIC_THINKING_BUDGETS,
    ANTHROPIC_THINKING_MODELS,
    OPENAI_REASONING_EFFORT,
    OPENAI_REASONING_MODELS,
    TIER_TOKEN_BUDGETS,
    ProviderLiteral,
    Tier,
)
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    ProviderError,
    RateLimitError,
)
from .usage import UsageTracker


# ── Response ──────────────────────────────────────────────────────────────────

@dataclass
class RouterResponse:
    """
    Unified response from ThinkRouter regardless of provider.

    Attributes
    ----------
    content        : Generated text content.
    routing        : ClassifierResult — the routing decision made.
    raw            : Original provider SDK response object.
    provider       : "openai" | "anthropic" | "generic"
    model          : Model identifier used.
    usage_tokens   : Token usage dict from provider.
    reasoning_effort: OpenAI reasoning_effort applied (if applicable).
    thinking_budget: Anthropic thinking budget_tokens applied (if applicable).
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
        extra = ""
        if self.reasoning_effort:
            extra = f", reasoning_effort={self.reasoning_effort!r}"
        if self.thinking_budget is not None:
            extra = f", thinking_budget={self.thinking_budget}"
        return (
            f"RouterResponse("
            f"tier={self.routing.tier.name}, "
            f"budget={self.routing.token_budget} tokens"
            f"{extra}, "
            f"model={self.model!r})"
        )


# ── Retry helper ──────────────────────────────────────────────────────────────

def _with_retry(fn, max_retries: int = 3, base_delay: float = 1.0):
    """
    Synchronous retry with exponential backoff.
    Retries on RateLimitError and transient ProviderError.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError as exc:
            last_exc = exc
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
        except ProviderError as exc:
            if exc.status_code in (500, 502, 503, 504):
                last_exc = exc
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise
    raise last_exc


async def _with_retry_async(fn, max_retries: int = 3, base_delay: float = 1.0):
    """Async retry with exponential backoff."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return await fn()
        except RateLimitError as exc:
            last_exc = exc
            await asyncio.sleep(base_delay * (2 ** attempt))
        except ProviderError as exc:
            if exc.status_code in (500, 502, 503, 504):
                last_exc = exc
                await asyncio.sleep(base_delay * (2 ** attempt))
            else:
                raise
    raise last_exc


# ── OpenAI adapter ────────────────────────────────────────────────────────────

class _OpenAIAdapter:
    """
    Wraps openai.OpenAI and openai.AsyncOpenAI.

    For reasoning models (o1, o3): sets reasoning_effort based on tier.
    For standard models (gpt-4o): sets max_tokens and injects a budget hint.
    """

    def __init__(self, api_key: Optional[str], **kwargs: Any) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ConfigurationError(
                "OpenAI provider requires:  pip install thinkrouter[openai]"
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ConfigurationError(
                "No OpenAI API key found. Pass api_key= or set OPENAI_API_KEY."
            )
        self._client       = openai.OpenAI(api_key=key, **kwargs)
        self._async_client = openai.AsyncOpenAI(api_key=key, **kwargs)

    def _build_params(
        self,
        messages:     List[Dict[str, str]],
        model:        str,
        tier:         Tier,
        temperature:  float,
        extra:        Dict[str, Any],
    ) -> tuple:
        """Build API params and return (params, reasoning_effort_applied)."""
        params: Dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            **extra,
        )
        reasoning_effort = None

        if model in OPENAI_REASONING_MODELS:
            # o1/o3: use reasoning_effort to control thinking compute
            reasoning_effort           = OPENAI_REASONING_EFFORT[tier]
            params["reasoning_effort"] = reasoning_effort
            # Remove temperature — not supported on reasoning models
            params.pop("temperature", None)
        else:
            # Standard models: limit max_tokens
            params["max_tokens"] = TIER_TOKEN_BUDGETS[tier]
            # Inject a budget hint for models that support system instructions
            if tier == Tier.NO_THINK:
                budget_msg = {
                    "role":    "system",
                    "content": "Answer directly and concisely. No extended reasoning needed.",
                }
                if not any(m.get("role") == "system" for m in messages):
                    params["messages"] = [budget_msg] + list(messages)

        return params, reasoning_effort

    def _parse_response(self, resp) -> tuple:
        content = resp.choices[0].message.content or ""
        usage   = {
            "prompt_tokens":     getattr(resp.usage, "prompt_tokens",     0),
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
            "total_tokens":      getattr(resp.usage, "total_tokens",      0),
        }
        return content, resp, usage

    def _handle_error(self, exc: Exception) -> None:
        try:
            import openai
            if isinstance(exc, openai.RateLimitError):
                raise RateLimitError(str(exc), status_code=429, provider="openai") from exc
            if isinstance(exc, openai.AuthenticationError):
                raise AuthenticationError(str(exc), status_code=401, provider="openai") from exc
            if isinstance(exc, openai.APIStatusError):
                raise ProviderError(str(exc), status_code=exc.status_code, provider="openai") from exc
        except ImportError:
            pass
        raise ProviderError(str(exc), provider="openai") from exc

    def call(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float,
        **extra:     Any,
    ) -> tuple:
        params, reasoning_effort = self._build_params(
            messages, model, tier, temperature, extra
        )

        def _do():
            try:
                return self._client.chat.completions.create(**params)
            except Exception as exc:
                self._handle_error(exc)

        resp    = _with_retry(_do)
        content, raw, usage = self._parse_response(resp)
        return content, raw, usage, reasoning_effort

    async def acall(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float,
        **extra:     Any,
    ) -> tuple:
        params, reasoning_effort = self._build_params(
            messages, model, tier, temperature, extra
        )

        async def _do():
            try:
                return await self._async_client.chat.completions.create(**params)
            except Exception as exc:
                self._handle_error(exc)

        resp    = await _with_retry_async(_do)
        content, raw, usage = self._parse_response(resp)
        return content, raw, usage, reasoning_effort

    def stream(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float,
        **extra:     Any,
    ) -> Iterator[str]:
        params, _ = self._build_params(messages, model, tier, temperature, extra)
        params["stream"] = True
        try:
            for chunk in self._client.chat.completions.create(**params):
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            self._handle_error(exc)

    async def astream(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float,
        **extra:     Any,
    ) -> AsyncIterator[str]:
        params, _ = self._build_params(messages, model, tier, temperature, extra)
        params["stream"] = True
        try:
            async for chunk in await self._async_client.chat.completions.create(**params):
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            self._handle_error(exc)


# ── Anthropic adapter ─────────────────────────────────────────────────────────

class _AnthropicAdapter:
    """
    Wraps anthropic.Anthropic and anthropic.AsyncAnthropic.

    For thinking-capable models (claude-opus-4-6, claude-3-7-sonnet, etc.):
    uses thinking={"type":"enabled","budget_tokens":N} to control the actual
    extended thinking budget directly.
    For standard Claude models: uses max_tokens.
    """

    def __init__(self, api_key: Optional[str], **kwargs: Any) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ConfigurationError(
                "Anthropic provider requires:  pip install thinkrouter[anthropic]"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigurationError(
                "No Anthropic API key found. Pass api_key= or set ANTHROPIC_API_KEY."
            )
        self._client       = anthropic.Anthropic(api_key=key, **kwargs)
        self._async_client = anthropic.AsyncAnthropic(api_key=key, **kwargs)

    def _build_params(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float,
        system:      Optional[str],
        extra:       Dict[str, Any],
    ) -> tuple:
        """Build params and return (params, thinking_budget_applied)."""
        budget = ANTHROPIC_THINKING_BUDGETS[tier]

        params: Dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            **extra,
        )
        if system:
            params["system"] = system

        thinking_budget = None

        if model in ANTHROPIC_THINKING_MODELS:
            if tier == Tier.NO_THINK:
                # Disable extended thinking entirely for simple queries
                params["thinking"] = {"type": "disabled"}
                params["max_tokens"] = 1_024
            else:
                # Set the actual thinking budget via the API parameter
                thinking_budget     = budget
                params["thinking"]  = {
                    "type":         "enabled",
                    "budget_tokens": budget,
                }
                params["max_tokens"] = budget + 2_048
                # Extended thinking requires temperature=1
                params["temperature"] = 1.0
        else:
            params["max_tokens"] = TIER_TOKEN_BUDGETS[tier]

        return params, thinking_budget

    def _parse_response(self, resp) -> tuple:
        content = ""
        for block in resp.content:
            if hasattr(block, "text"):
                content = block.text
                break
        usage = {
            "prompt_tokens":     resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens":      resp.usage.input_tokens + resp.usage.output_tokens,
        }
        return content, resp, usage

    def _handle_error(self, exc: Exception) -> None:
        try:
            import anthropic
            if isinstance(exc, anthropic.RateLimitError):
                raise RateLimitError(str(exc), status_code=429, provider="anthropic") from exc
            if isinstance(exc, anthropic.AuthenticationError):
                raise AuthenticationError(str(exc), status_code=401, provider="anthropic") from exc
            if isinstance(exc, anthropic.APIStatusError):
                raise ProviderError(str(exc), status_code=exc.status_code, provider="anthropic") from exc
        except ImportError:
            pass
        raise ProviderError(str(exc), provider="anthropic") from exc

    def call(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float,
        system:      Optional[str] = None,
        **extra:     Any,
    ) -> tuple:
        params, thinking_budget = self._build_params(
            messages, model, tier, temperature, system, extra
        )

        def _do():
            try:
                return self._client.messages.create(**params)
            except Exception as exc:
                self._handle_error(exc)

        resp = _with_retry(_do)
        content, raw, usage = self._parse_response(resp)
        return content, raw, usage, thinking_budget

    async def acall(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float,
        system:      Optional[str] = None,
        **extra:     Any,
    ) -> tuple:
        params, thinking_budget = self._build_params(
            messages, model, tier, temperature, system, extra
        )

        async def _do():
            try:
                return await self._async_client.messages.create(**params)
            except Exception as exc:
                self._handle_error(exc)

        resp = await _with_retry_async(_do)
        content, raw, usage = self._parse_response(resp)
        return content, raw, usage, thinking_budget

    def stream(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float,
        system:      Optional[str] = None,
        **extra:     Any,
    ) -> Iterator[str]:
        params, _ = self._build_params(
            messages, model, tier, temperature, system, extra
        )
        try:
            with self._client.messages.stream(**params) as s:
                yield from s.text_stream
        except Exception as exc:
            self._handle_error(exc)

    async def astream(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float,
        system:      Optional[str] = None,
        **extra:     Any,
    ) -> AsyncIterator[str]:
        params, _ = self._build_params(
            messages, model, tier, temperature, system, extra
        )
        try:
            async with self._async_client.messages.stream(**params) as s:
                async for chunk in s.text_stream:
                    yield chunk
        except Exception as exc:
            self._handle_error(exc)


# ── ThinkRouter ───────────────────────────────────────────────────────────────

class ThinkRouter:
    """
    Production-ready pre-inference routing layer for LLM reasoning models.

    Correctly controls reasoning compute for:
      - OpenAI o1 / o3 via reasoning_effort parameter
      - Anthropic Claude via thinking.budget_tokens parameter
      - Standard models via max_tokens + system prompt hints

    Parameters
    ----------
    provider             : "openai" | "anthropic" | "generic"
    api_key              : Provider key. Falls back to env var.
    model                : Default model for all calls.
    classifier_backend   : "heuristic" (default) or "distilbert"
    confidence_threshold : Min confidence to accept routing. Below → FULL.
    max_retries          : Retry attempts on rate limits / transient errors.
    max_records          : Max call records in usage tracker.
    verbose              : Print routing decision per call.
    **client_kwargs      : Passed to the provider SDK client.

    Examples
    --------
    # OpenAI o1 — reasoning_effort auto-set per query
    >>> client = ThinkRouter(provider="openai", model="o1")
    >>> r = client.chat("What is 7 * 8?")
    >>> r.reasoning_effort
    'low'

    # Anthropic with extended thinking
    >>> client = ThinkRouter(provider="anthropic", model="claude-opus-4-6")
    >>> r = client.chat("Prove P != NP.")
    >>> r.thinking_budget
    10000

    # Async
    >>> r = await client.achat("Explain gradient descent.")

    # Classify only (no API call)
    >>> client.classify("Write a quicksort implementation.")
    ClassifierResult(tier=FULL, confidence=0.87, budget=8000 tokens, latency=0.4ms)
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
        **client_kwargs:      Any,
    ) -> None:
        self.provider     = provider
        self.model        = model or self._DEFAULT_MODELS.get(provider, "unknown")
        self.verbose      = verbose
        self.max_retries  = max_retries
        self._threshold   = confidence_threshold

        clf_kw: Dict[str, Any] = {}
        if classifier_backend == "distilbert":
            clf_kw["threshold"] = confidence_threshold
        self._classifier: BaseClassifier = get_classifier(classifier_backend, **clf_kw)

        self.usage = UsageTracker(max_records=max_records)

        self._adapter: Optional[Any] = None
        if provider == "openai":
            self._adapter = _OpenAIAdapter(api_key=api_key, **client_kwargs)
        elif provider == "anthropic":
            self._adapter = _AnthropicAdapter(api_key=api_key, **client_kwargs)
        elif provider != "generic":
            raise ConfigurationError(
                f"Unknown provider: {provider!r}. Choose 'openai', 'anthropic', or 'generic'."
            )

    # ── Classify ──────────────────────────────────────────────────────────────

    def classify(self, query: str) -> ClassifierResult:
        """Classify a single query without making an API call."""
        return self._classifier.predict(query)

    def classify_batch(self, queries: List[str]) -> List[ClassifierResult]:
        """Classify a list of queries without making API calls."""
        return self._classifier.predict_batch(queries)

    # ── Sync chat ─────────────────────────────────────────────────────────────

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
        Route and execute a chat completion.

        For OpenAI reasoning models: applies reasoning_effort.
        For Anthropic thinking models: applies thinking.budget_tokens.
        For standard models: applies max_tokens + system hints.
        """
        if self._adapter is None:
            raise ConfigurationError(
                "No provider configured. Pass provider='openai' or 'anthropic'."
            )

        clf    = self._classifier.predict(query)
        target = model or self.model

        self._log(clf, target)

        msg_list = self._build_messages(query, messages, system)
        call_kw  = dict(extra)
        if self.provider == "anthropic" and system:
            call_kw["system"] = system

        result = self._adapter.call(
            messages=msg_list,
            model=target,
            tier=clf.tier,
            temperature=temperature,
            **call_kw,
        )
        content, raw, usage_tokens, extra_param = result

        self.usage.record(
            query=query, tier=clf.tier,
            confidence=clf.confidence, latency_ms=clf.latency_ms,
        )

        return RouterResponse(
            content=content, routing=clf, raw=raw,
            provider=self.provider, model=target,
            usage_tokens=usage_tokens,
            reasoning_effort=extra_param if self.provider == "openai" else None,
            thinking_budget=extra_param if self.provider == "anthropic" else None,
        )

    # ── Async chat ────────────────────────────────────────────────────────────

    async def achat(
        self,
        query:       str,
        model:       Optional[str]             = None,
        messages:    Optional[List[Dict[str, str]]] = None,
        system:      Optional[str]             = None,
        temperature: float                     = 0.7,
        **extra:     Any,
    ) -> RouterResponse:
        """Async version of chat(). Use with asyncio or in FastAPI endpoints."""
        if self._adapter is None:
            raise ConfigurationError("No provider configured.")

        clf    = self._classifier.predict(query)
        target = model or self.model
        self._log(clf, target)

        msg_list = self._build_messages(query, messages, system)
        call_kw  = dict(extra)
        if self.provider == "anthropic" and system:
            call_kw["system"] = system

        result = await self._adapter.acall(
            messages=msg_list,
            model=target,
            tier=clf.tier,
            temperature=temperature,
            **call_kw,
        )
        content, raw, usage_tokens, extra_param = result

        self.usage.record(
            query=query, tier=clf.tier,
            confidence=clf.confidence, latency_ms=clf.latency_ms,
        )

        return RouterResponse(
            content=content, routing=clf, raw=raw,
            provider=self.provider, model=target,
            usage_tokens=usage_tokens,
            reasoning_effort=extra_param if self.provider == "openai" else None,
            thinking_budget=extra_param if self.provider == "anthropic" else None,
        )

    # ── Sync stream ───────────────────────────────────────────────────────────

    def stream(
        self,
        query:       str,
        model:       Optional[str] = None,
        system:      Optional[str] = None,
        temperature: float         = 0.7,
        **extra:     Any,
    ) -> Iterator[str]:
        """Stream response tokens. Routing decision applied before first token."""
        if self._adapter is None:
            raise ConfigurationError("No provider configured.")

        clf      = self._classifier.predict(query)
        target   = model or self.model
        msg_list = self._build_messages(query, None, system)
        call_kw  = dict(extra)
        if self.provider == "anthropic" and system:
            call_kw["system"] = system

        self._log(clf, target)

        yield from self._adapter.stream(
            messages=msg_list, model=target,
            tier=clf.tier, temperature=temperature, **call_kw,
        )

        self.usage.record(
            query=query, tier=clf.tier,
            confidence=clf.confidence, latency_ms=clf.latency_ms,
        )

    # ── Async stream ──────────────────────────────────────────────────────────

    async def astream(
        self,
        query:       str,
        model:       Optional[str] = None,
        system:      Optional[str] = None,
        temperature: float         = 0.7,
        **extra:     Any,
    ) -> AsyncIterator[str]:
        """Async streaming. Use in async frameworks like FastAPI."""
        if self._adapter is None:
            raise ConfigurationError("No provider configured.")

        clf      = self._classifier.predict(query)
        target   = model or self.model
        msg_list = self._build_messages(query, None, system)
        call_kw  = dict(extra)
        if self.provider == "anthropic" and system:
            call_kw["system"] = system

        self._log(clf, target)

        async for chunk in self._adapter.astream(
            messages=msg_list, model=target,
            tier=clf.tier, temperature=temperature, **call_kw,
        ):
            yield chunk

        self.usage.record(
            query=query, tier=clf.tier,
            confidence=clf.confidence, latency_ms=clf.latency_ms,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_messages(
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

    def _log(self, clf: ClassifierResult, model: str) -> None:
        if not self.verbose:
            return
        extra = ""
        if self.provider == "openai" and model in OPENAI_REASONING_MODELS:
            extra = f"  reasoning_effort={OPENAI_REASONING_EFFORT[clf.tier]!r}"
        if self.provider == "anthropic" and model in ANTHROPIC_THINKING_MODELS:
            budget = ANTHROPIC_THINKING_BUDGETS[clf.tier]
            extra  = f"  thinking_budget={budget}"
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
