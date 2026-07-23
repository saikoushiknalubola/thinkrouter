"""
thinkrouter.fallback
~~~~~~~~~~~~~~~~~~~~
Fallback chain router — v0.7.0.

Routes to a primary model and silently fails over to the next
provider in the chain on rate limits or transient errors.
Zero code changes needed in the application layer.

Usage::

    from thinkrouter import ThinkRouter

    client = ThinkRouter(
        provider="openai",
        fallback_providers=["anthropic", "ollama"],
    )

    # If OpenAI is rate-limited → automatically retries on Anthropic
    # If Anthropic fails → falls over to Ollama (free, local)
    r = client.chat("Write a binary search tree in Python.")
    print(r.provider)          # whichever provider actually responded
    print(r.fallback_used)     # True if primary failed
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .constants import Tier
from .exceptions import ProviderError



# ── Fallback result metadata ───────────────────────────────────────────────

@dataclass
class FallbackResult:
    """Metadata about a fallback chain execution."""
    attempted:       List[str]    # providers tried in order
    succeeded:       str          # provider that succeeded
    fallback_used:   bool         # True if primary was not used
    attempts:        int
    total_latency_ms: float
    errors:          List[str]    # error messages from failed attempts


def _is_ollama_adapter(adapter: object) -> bool:
    """
    Reliable Ollama detection that works under unittest.mock.
    Checks the actual class name rather than attributes, so MagicMock
    instances always return False.
    """
    return type(adapter).__name__ == "OllamaAdapter"


# ── Fallback chain ─────────────────────────────────────────────────────────

class FallbackChain:
    """
    Ordered list of provider adapters tried in sequence.

    On any rate limit or transient server error (429, 5xx) the chain
    moves to the next provider. On success it returns immediately.
    On permanent errors (401 auth, 404 model not found) it raises
    without trying further providers — those errors won't be fixed
    by switching providers.

    Parameters
    ----------
    adapters    : Ordered list of (provider_name, adapter_instance) tuples.
    models      : Model to use per provider (same index as adapters).
    retry_delay : Base delay in seconds before retrying within a provider.
    """

    # Error status codes that are NOT worth retrying on next provider
    _PERMANENT_ERRORS = {401, 403, 404}

    def __init__(
        self,
        adapters:    List[Tuple[str, Any]],  # [(provider_name, adapter), ...]
        models:      List[str],
        retry_delay: float = 0.5,
    ) -> None:
        if not adapters:
            raise ValueError("FallbackChain requires at least one adapter.")
        if len(adapters) != len(models):
            raise ValueError("adapters and models must have the same length.")
        self._adapters    = adapters
        self._models      = models
        self._retry_delay = retry_delay

    def call(
        self,
        messages:    List[Dict[str, str]],
        tier:        Tier,
        temperature: float = 0.7,
        **extra:     Any,
    ) -> Tuple[str, Any, Dict[str, int], Any, FallbackResult]:
        """
        Execute the chain synchronously.

        Returns
        -------
        (content, raw, usage_tokens, xparam, FallbackResult)
        """
        t0       = time.perf_counter()
        errors:  List[str] = []
        tried:   List[str] = []
        primary  = self._adapters[0][0]

        for i, ((prov_name, adapter), model) in enumerate(
            zip(self._adapters, self._models)
        ):
            tried.append(prov_name)
            try:
                # Ollama adapter has a different call signature
                if _is_ollama_adapter(adapter):
                    content, raw, usage = adapter.call(
                        messages=messages,
                        model=model,
                        max_tokens=_tier_to_max(tier),
                        temperature=temperature,
                        **_strip_ollama_unsupported(extra),
                    )
                    xparam = None
                else:
                    content, raw, usage, xparam = adapter.call(
                        messages=messages,
                        model=model,
                        tier=tier,
                        temperature=temperature,
                        **extra,
                    )

                ms = (time.perf_counter() - t0) * 1000
                result = FallbackResult(
                    attempted=tried,
                    succeeded=prov_name,
                    fallback_used=(prov_name != primary),
                    attempts=i + 1,
                    total_latency_ms=ms,
                    errors=errors,
                )
                return content, raw, usage, xparam, result

            except ProviderError as exc:
                # Permanent errors: don't try next provider
                if exc.status_code in self._PERMANENT_ERRORS:
                    raise

                errors.append(f"{prov_name}: {exc}")
                if i < len(self._adapters) - 1:
                    time.sleep(self._retry_delay)
                continue

            except Exception as exc:
                errors.append(f"{prov_name}: {exc}")
                if i < len(self._adapters) - 1:
                    time.sleep(self._retry_delay)
                continue

        # All providers failed
        raise ProviderError(
            f"All providers failed after {len(tried)} attempts. "
            f"Errors: {'; '.join(errors)}",
            provider="fallback-chain",
        )

    async def acall(
        self,
        messages:    List[Dict[str, str]],
        tier:        Tier,
        temperature: float = 0.7,
        **extra:     Any,
    ) -> Tuple[str, Any, Dict[str, int], Any, FallbackResult]:
        """Async version of call()."""
        import asyncio

        t0      = time.perf_counter()
        errors: List[str] = []
        tried:  List[str] = []
        primary = self._adapters[0][0]

        for i, ((prov_name, adapter), model) in enumerate(
            zip(self._adapters, self._models)
        ):
            tried.append(prov_name)
            try:
                if _is_ollama_adapter(adapter):
                    content, raw, usage = await adapter.acall(
                        messages=messages,
                        model=model,
                        max_tokens=_tier_to_max(tier),
                        temperature=temperature,
                        **_strip_ollama_unsupported(extra),
                    )
                    xparam = None
                else:
                    content, raw, usage, xparam = await adapter.acall(
                        messages=messages,
                        model=model,
                        tier=tier,
                        temperature=temperature,
                        **extra,
                    )

                ms = (time.perf_counter() - t0) * 1000
                result = FallbackResult(
                    attempted=tried,
                    succeeded=prov_name,
                    fallback_used=(prov_name != primary),
                    attempts=i + 1,
                    total_latency_ms=ms,
                    errors=errors,
                )
                return content, raw, usage, xparam, result

            except ProviderError as exc:
                if exc.status_code in self._PERMANENT_ERRORS:
                    raise
                errors.append(f"{prov_name}: {exc}")
                if i < len(self._adapters) - 1:
                    await asyncio.sleep(self._retry_delay)
                continue

            except Exception as exc:
                errors.append(f"{prov_name}: {exc}")
                if i < len(self._adapters) - 1:
                    await asyncio.sleep(self._retry_delay)
                continue

        raise ProviderError(
            f"All providers failed. Errors: {'; '.join(errors)}",
            provider="fallback-chain",
        )

    def __repr__(self) -> str:
        chain = " → ".join(
            f"{p}:{m}" for (p, _), m in zip(self._adapters, self._models)
        )
        return f"FallbackChain({chain})"


# ── Helpers ────────────────────────────────────────────────────────────────

def _tier_to_max(tier: Tier) -> int:
    from .constants import TIER_TOKEN_BUDGETS
    return TIER_TOKEN_BUDGETS[tier]


def _strip_ollama_unsupported(extra: Dict) -> Dict:
    """Remove parameters not supported by Ollama's OpenAI-compat API."""
    drop = {"reasoning_effort", "thinking"}
    return {k: v for k, v in extra.items() if k not in drop}
