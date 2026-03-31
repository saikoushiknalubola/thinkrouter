"""
thinkrouter.exceptions
~~~~~~~~~~~~~~~~~~~~~~
Clean exception hierarchy. Catching ThinkRouterError catches everything.
"""
from __future__ import annotations


class ThinkRouterError(Exception):
    """Base for all ThinkRouter exceptions."""


class ProviderError(ThinkRouterError):
    """Provider API returned an error."""
    def __init__(self, message: str, status_code: int = 0, provider: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider    = provider


class RateLimitError(ProviderError):
    """HTTP 429 — rate limit exceeded."""


class AuthenticationError(ProviderError):
    """HTTP 401/403 — invalid or missing API key."""


class ModelNotFoundError(ProviderError):
    """HTTP 404 — model not found or not available."""


class ClassifierError(ThinkRouterError):
    """Classifier failed to load or run inference."""


class ConfigurationError(ThinkRouterError):
    """ThinkRouter is misconfigured."""
