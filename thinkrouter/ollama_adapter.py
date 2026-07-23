"""
thinkrouter.ollama_adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~
Adapter for Ollama — free local model inference.

Ollama exposes an OpenAI-compatible API at http://localhost:11434.
No API key required. Models run fully offline.

Install Ollama:  https://ollama.ai
Pull a model:    ollama pull deepseek-coder-v2
Start server:    ollama serve  (auto-starts on most systems)

Usage::

    from thinkrouter.ollama_adapter import OllamaAdapter

    adapter = OllamaAdapter()
    content, raw, usage = adapter.call(
        messages=[{"role":"user","content":"Write a binary search in Python."}],
        model="deepseek-coder-v2",
    )
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Tuple

import httpx

from .exceptions import ConfigurationError, ProviderError, RateLimitError


class OllamaAdapter:
    """
    Adapter for the Ollama local inference server.

    Ollama provides an OpenAI-compatible /v1/chat/completions endpoint.
    Runs on localhost:11434 by default.

    Parameters
    ----------
    base_url : str
        Ollama server URL. Default: http://localhost:11434
    timeout  : float
        Request timeout in seconds. Default: 120 (local models can be slow)
    """

    _is_ollama: bool = True  # type marker for FallbackChain detection

    def __init__(
        self,
        base_url: str   = "http://localhost:11434",
        timeout:  float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._sync_client  = httpx.Client(timeout=timeout)
        self._async_client = httpx.AsyncClient(timeout=timeout)

    def is_available(self) -> bool:
        """Check if the Ollama server is running."""
        try:
            resp = self._sync_client.get(f"{self.base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return names of locally available models."""
        try:
            resp = self._sync_client.get(f"{self.base_url}/api/tags")
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def _raise(self, exc: Exception, status: int = 0) -> None:
        if status == 429:
            raise RateLimitError(str(exc), 429, "ollama") from exc
        raise ProviderError(str(exc), status, "ollama") from exc

    def _parse(self, data: dict) -> Tuple[str, Dict[str, int]]:
        choices = data.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""
        usage_raw = data.get("usage", {})
        usage = {
            "prompt_tokens":     usage_raw.get("prompt_tokens", 0),
            "completion_tokens": usage_raw.get("completion_tokens", 0),
            "total_tokens":      usage_raw.get("total_tokens", 0),
        }
        return content, usage

    def call(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        max_tokens:  int   = 2048,
        temperature: float = 0.7,
        **extra:     Any,
    ) -> Tuple[str, dict, Dict[str, int]]:
        url = f"{self.base_url}/v1/chat/completions"
        payload = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            **extra,
        )
        try:
            resp = self._sync_client.post(url, json=payload)
            if resp.status_code != 200:
                self._raise(Exception(resp.text), resp.status_code)
            data = resp.json()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ConfigurationError(
                f"Cannot connect to Ollama at {self.base_url}.\n"
                "Make sure Ollama is installed and running: https://ollama.ai\n"
                f"Then pull your model:  ollama pull {model}"
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            self._raise(exc)

        content, usage = self._parse(data)
        return content, data, usage

    async def acall(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        max_tokens:  int   = 2048,
        temperature: float = 0.7,
        **extra:     Any,
    ) -> Tuple[str, dict, Dict[str, int]]:
        url = f"{self.base_url}/v1/chat/completions"
        payload = dict(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
            stream=False, **extra,
        )
        try:
            resp = await self._async_client.post(url, json=payload)
            if resp.status_code != 200:
                self._raise(Exception(resp.text), resp.status_code)
            data = resp.json()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ConfigurationError(
                f"Cannot connect to Ollama at {self.base_url}.\n"
                f"Run: ollama pull {model} && ollama serve"
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            self._raise(exc)

        content, usage = self._parse(data)
        return content, data, usage

    def stream(
        self,
        messages:    List[Dict[str, str]],
        model:       str,
        max_tokens:  int   = 2048,
        temperature: float = 0.7,
        **extra:     Any,
    ) -> Iterator[str]:
        url = f"{self.base_url}/v1/chat/completions"
        payload = dict(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
            stream=True, **extra,
        )
        try:
            with self._sync_client.stream("POST", url, json=payload) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, KeyError):
                            continue
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ConfigurationError(
                f"Cannot connect to Ollama at {self.base_url}."
            ) from exc
