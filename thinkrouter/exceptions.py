"""
thinkrouter.exceptions
~~~~~~~~~~~~~~~~~~~~~~
All custom exceptions raised by ThinkRouter.

Catching ThinkRouterError catches everything.
Catching ProviderError catches all API-level failures.
"""
from __future__ import annotations


class ThinkRouterError(Exception):
    """Base exception for all ThinkRouter errors."""


class ProviderError(ThinkRouterError):
    """Raised when the underlying LLM provider returns an error."""

    def __init__(self, message: str, status_code: int = 0, provider: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider    = provider


class RateLimitError(ProviderError):
    """Raised on HTTP 429 — rate limit exceeded."""


class AuthenticationError(ProviderError):
    """Raised on HTTP 401/403 — invalid or missing API key."""


class ClassifierError(ThinkRouterError):
    """Raised when the difficulty classifier fails to load or predict."""


class ConfigurationError(ThinkRouterError):
    """Raised when ThinkRouter is misconfigured (bad provider, missing key, etc.)."""
