"""
thinkrouter.config
~~~~~~~~~~~~~~~~~~
All runtime configuration loaded from environment variables.
Every value has a sensible default so ThinkRouter works out of the box.

Environment variables:

  OPENAI_API_KEY              OpenAI API key
  ANTHROPIC_API_KEY           Anthropic API key
  THINKROUTER_BACKEND         Classifier backend: heuristic | distilbert (default: heuristic)
  THINKROUTER_THRESHOLD       Confidence threshold 0.0–1.0 (default: 0.75)
  THINKROUTER_MAX_RETRIES     Max retries on transient errors (default: 3)
  THINKROUTER_VERBOSE         Print routing decisions: 1 | 0 (default: 0)
  THINKROUTER_HF_MODEL        HuggingFace model id for DistilBERT backend
  THINKROUTER_MAX_RECORDS     Max usage records retained (default: 10000)
"""
from __future__ import annotations

import os


def _bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, "1" if default else "0").strip() in ("1", "true", "True")


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


class Config:
    """
    Immutable runtime configuration snapshot.
    Instantiate once and pass around — do not read os.environ repeatedly.
    """
    __slots__ = (
        "openai_api_key",
        "anthropic_api_key",
        "classifier_backend",
        "confidence_threshold",
        "max_retries",
        "verbose",
        "hf_model",
        "max_records",
    )

    def __init__(self) -> None:
        self.openai_api_key       = os.getenv("OPENAI_API_KEY",    "")
        self.anthropic_api_key    = os.getenv("ANTHROPIC_API_KEY", "")
        self.classifier_backend   = os.getenv("THINKROUTER_BACKEND",   "heuristic")
        self.confidence_threshold = _float("THINKROUTER_THRESHOLD",    0.75)
        self.max_retries          = _int("THINKROUTER_MAX_RETRIES",     3)
        self.verbose              = _bool("THINKROUTER_VERBOSE",        False)
        self.hf_model             = os.getenv(
            "THINKROUTER_HF_MODEL",
            "YOUR_HF_USERNAME/query-difficulty-classifier",
        )
        self.max_records          = _int("THINKROUTER_MAX_RECORDS",     10_000)

    def __repr__(self) -> str:
        return (
            f"Config("
            f"backend={self.classifier_backend!r}, "
            f"threshold={self.confidence_threshold}, "
            f"verbose={self.verbose})"
        )


# Module-level default — applications can override by passing Config() explicitly.
DEFAULT_CONFIG = Config()
