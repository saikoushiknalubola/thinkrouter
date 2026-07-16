"""
thinkrouter.providers
~~~~~~~~~~~~~~~~~~~~~
Provider adapters for OpenAI and Anthropic.

Each adapter correctly applies the provider's native reasoning
budget controls:

  OpenAI o1/o3   → reasoning_effort="low"|"high"
  Anthropic       → thinking={"type":"enabled","budget_tokens":N}
  Standard models → max_tokens + system-prompt budget hint

Both sync and async paths are implemented.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

from .constants import (
    ANTHROPIC_THINKING_BUDGETS,
    ANTHROPIC_THINKING_MODELS,
    OPENAI_REASONING_EFFORT,
    OPENAI_REASONING_MODELS,
    TIER_TOKEN_BUDGETS,
    Tier,
)
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)


# ── Retry helper ──────────────────────────────────────────────────────────

def _retry_sync(fn, max_retries: int = 3, base_delay: float = 1.0):
    last: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError as exc:
            last = exc
            time.sleep(base_delay * (2 ** attempt))
        except ProviderError as exc:
            if exc.status_code in (500, 502, 503, 504):
                last = exc
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise
    assert last is not None
    raise last


async def _retry_async(fn, max_retries: int = 3, base_delay: float = 1.0):
    last: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return await fn()
        except RateLimitError as exc:
            last = exc
            await asyncio.sleep(base_delay * (2 ** attempt))
        except ProviderError as exc:
            if exc.status_code in (500, 502, 503, 504):
                last = exc
                await asyncio.sleep(base_delay * (2 ** attempt))
            else:
                raise
    assert last is not None
    raise last


# ── OpenAI adapter ────────────────────────────────────────────────────────

class OpenAIAdapter:
    """
    Wraps openai.OpenAI + openai.AsyncOpenAI.

    Routing strategy:
      o1/o3 models : reasoning_effort="low"|"high"
      Standard     : max_tokens + optional system-prompt budget hint
    """

    def __init__(self, api_key: str, max_retries: int = 3, **client_kwargs: Any) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ConfigurationError(
                "OpenAI provider requires:  pip install thinkrouter[openai]"
            ) from exc

        self._max_retries  = max_retries
        self._client       = openai.OpenAI(api_key=api_key, **client_kwargs)
        self._async_client = openai.AsyncOpenAI(api_key=api_key, **client_kwargs)

    # ── Param builder ─────────────────────────────────────────────────────

    def _params(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float,
        stream:      bool,
        extra:       Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Build the API call parameters.
        Returns (params_dict, reasoning_effort_applied | None).
        """
        params: Dict[str, Any] = dict(
            model=model, messages=list(messages),
            temperature=temperature, stream=stream, **extra,
        )
        reasoning_effort = None

        if model in OPENAI_REASONING_MODELS:
            # Reasoning models: use reasoning_effort to control thinking compute.
            # Remove temperature — not supported on o1/o3.
            params.pop("temperature", None)
            reasoning_effort           = OPENAI_REASONING_EFFORT[tier]
            params["reasoning_effort"] = reasoning_effort
        else:
            # Standard models: limit output tokens.
            params["max_tokens"] = TIER_TOKEN_BUDGETS[tier]
            # For NO_THINK queries, nudge the model to answer concisely
            # via a system message if one is not already present.
            if tier == Tier.NO_THINK:
                has_system = any(m.get("role") == "system" for m in messages)
                if not has_system:
                    hint = {"role": "system", "content":
                            "Answer directly and concisely. No extended reasoning required."}
                    params["messages"] = [hint] + list(messages)

        return params, reasoning_effort

    # ── Error mapping ─────────────────────────────────────────────────────

    def _raise(self, exc: Exception) -> None:
        try:
            import openai
            if isinstance(exc, openai.RateLimitError):
                raise RateLimitError(str(exc), 429, "openai") from exc
            if isinstance(exc, openai.AuthenticationError):
                raise AuthenticationError(str(exc), 401, "openai") from exc
            if isinstance(exc, openai.NotFoundError):
                raise ModelNotFoundError(str(exc), 404, "openai") from exc
            if isinstance(exc, openai.APIStatusError):
                raise ProviderError(str(exc), exc.status_code, "openai") from exc
        except ImportError:
            pass
        raise ProviderError(str(exc), provider="openai") from exc

    def _parse(self, resp) -> Tuple[str, Dict[str, int]]:
        content = resp.choices[0].message.content or ""
        usage   = {
            "prompt_tokens":     getattr(resp.usage, "prompt_tokens",     0),
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
            "total_tokens":      getattr(resp.usage, "total_tokens",      0),
        }
        return content, usage

    # ── Sync ──────────────────────────────────────────────────────────────

    def call(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float = 0.7,
        **extra:     Any,
    ) -> Tuple[str, Any, Dict[str, int], Optional[str]]:
        params, effort = self._params(messages, model, tier, temperature, False, extra)

        def _do():
            try:
                return self._client.chat.completions.create(**params)
            except Exception as exc:
                self._raise(exc)

        resp             = _retry_sync(_do, self._max_retries)
        content, usage   = self._parse(resp)
        return content, resp, usage, effort

    def stream(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float = 0.7,
        **extra:     Any,
    ) -> Iterator[str]:
        params, _ = self._params(messages, model, tier, temperature, True, extra)
        try:
            for chunk in self._client.chat.completions.create(**params):
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            self._raise(exc)

    # ── Async ─────────────────────────────────────────────────────────────

    async def acall(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float = 0.7,
        **extra:     Any,
    ) -> Tuple[str, Any, Dict[str, int], Optional[str]]:
        params, effort = self._params(messages, model, tier, temperature, False, extra)

        async def _do():
            try:
                return await self._async_client.chat.completions.create(**params)
            except Exception as exc:
                self._raise(exc)

        resp           = await _retry_async(_do, self._max_retries)
        content, usage = self._parse(resp)
        return content, resp, usage, effort

    async def astream(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float = 0.7,
        **extra:     Any,
    ) -> AsyncIterator[str]:
        params, _ = self._params(messages, model, tier, temperature, True, extra)
        try:
            async for chunk in await self._async_client.chat.completions.create(**params):
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            self._raise(exc)


# ── Anthropic adapter ─────────────────────────────────────────────────────

class AnthropicAdapter:
    """
    Wraps anthropic.Anthropic + anthropic.AsyncAnthropic.

    Routing strategy:
      Thinking-capable models : thinking={"type":"enabled","budget_tokens":N}
                                 NO_THINK → thinking disabled entirely
      Standard models         : max_tokens only
    """

    def __init__(self, api_key: str, max_retries: int = 3, **client_kwargs: Any) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ConfigurationError(
                "Anthropic provider requires:  pip install thinkrouter[anthropic]"
            ) from exc

        self._max_retries  = max_retries
        self._client       = anthropic.Anthropic(api_key=api_key, **client_kwargs)
        self._async_client = anthropic.AsyncAnthropic(api_key=api_key, **client_kwargs)

    def _params(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float,
        system:      Optional[str],
        extra:       Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[int]]:
        params: Dict[str, Any] = dict(
            model=model, messages=list(messages),
            temperature=temperature, **extra,
        )
        if system:
            params["system"] = system

        thinking_budget: Optional[int] = None

        if model in ANTHROPIC_THINKING_MODELS:
            budget = ANTHROPIC_THINKING_BUDGETS[tier]
            if tier == Tier.NO_THINK or budget == 0:
                # Disable extended thinking — fastest path
                params["thinking"]    = {"type": "disabled"}
                params["max_tokens"]  = 1_024
            else:
                # Extended thinking — budget_tokens controls actual thinking compute
                thinking_budget      = budget
                params["thinking"]   = {"type": "enabled", "budget_tokens": budget}
                params["max_tokens"] = budget + 2_048
                # Extended thinking requires temperature=1
                params["temperature"] = 1.0
        else:
            params["max_tokens"] = TIER_TOKEN_BUDGETS[tier]

        return params, thinking_budget

    def _raise(self, exc: Exception) -> None:
        try:
            import anthropic
            if isinstance(exc, anthropic.RateLimitError):
                raise RateLimitError(str(exc), 429, "anthropic") from exc
            if isinstance(exc, anthropic.AuthenticationError):
                raise AuthenticationError(str(exc), 401, "anthropic") from exc
            if isinstance(exc, anthropic.NotFoundError):
                raise ModelNotFoundError(str(exc), 404, "anthropic") from exc
            if isinstance(exc, anthropic.APIStatusError):
                raise ProviderError(str(exc), exc.status_code, "anthropic") from exc
        except ImportError:
            pass
        raise ProviderError(str(exc), provider="anthropic") from exc

    def _parse(self, resp) -> Tuple[str, Dict[str, int]]:
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
        return content, usage

    def call(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float = 0.7,
        system:      Optional[str] = None,
        **extra:     Any,
    ) -> Tuple[str, Any, Dict[str, int], Optional[int]]:
        params, budget = self._params(messages, model, tier, temperature, system, extra)

        def _do():
            try:
                return self._client.messages.create(**params)
            except Exception as exc:
                self._raise(exc)

        resp             = _retry_sync(_do, self._max_retries)
        content, usage   = self._parse(resp)
        return content, resp, usage, budget

    def stream(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float = 0.7,
        system:      Optional[str] = None,
        **extra:     Any,
    ) -> Iterator[str]:
        params, _ = self._params(messages, model, tier, temperature, system, extra)
        try:
            with self._client.messages.stream(**params) as s:
                yield from s.text_stream
        except Exception as exc:
            self._raise(exc)

    async def acall(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float = 0.7,
        system:      Optional[str] = None,
        **extra:     Any,
    ) -> Tuple[str, Any, Dict[str, int], Optional[int]]:
        params, budget = self._params(messages, model, tier, temperature, system, extra)

        async def _do():
            try:
                return await self._async_client.messages.create(**params)
            except Exception as exc:
                self._raise(exc)

        resp           = await _retry_async(_do, self._max_retries)
        content, usage = self._parse(resp)
        return content, resp, usage, budget

    async def astream(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        tier:        Tier,
        temperature: float = 0.7,
        system:      Optional[str] = None,
        **extra:     Any,
    ) -> AsyncIterator[str]:
        params, _ = self._params(messages, model, tier, temperature, system, extra)
        try:
            async with self._async_client.messages.stream(**params) as s:
                async for chunk in s.text_stream:
                    yield chunk
        except Exception as exc:
            self._raise(exc)
