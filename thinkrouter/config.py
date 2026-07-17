"""
thinkrouter.config
~~~~~~~~~~~~~~~~~~
Runtime configuration loaded from environment variables.
Phase 2 adds atlas and embedder settings.

Environment variables:

  OPENAI_API_KEY              OpenAI API key
  ANTHROPIC_API_KEY           Anthropic API key
  THINKROUTER_BACKEND         Classifier: heuristic | distilbert (default: heuristic)
  THINKROUTER_THRESHOLD       Confidence threshold (default: 0.75)
  THINKROUTER_MAX_RETRIES     Max retries (default: 3)
  THINKROUTER_VERBOSE         Print routing decisions: 1|0 (default: 0)
  THINKROUTER_HF_MODEL        HuggingFace model id for DistilBERT backend
  THINKROUTER_MAX_RECORDS     Max usage records (default: 10000)

  Phase 2 — Atlas & Embedder:
  THINKROUTER_ATLAS_ENABLED   Enable atlas storage: 1|0 (default: 1)
  THINKROUTER_ATLAS_PATH      Atlas storage directory (default: ~/.thinkrouter/atlas)
  THINKROUTER_EMBEDDER        Embedder backend: hash|openai|local (default: hash)
  THINKROUTER_EMBED_DIM       Embedding dimensionality (default: 256)
  THINKROUTER_ATLAS_MAX       Max atlas records — None = unlimited (default: None)
"""
from __future__ import annotations

import os
from typing import Optional


def _bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, "1" if default else "0").strip().lower() in ("1","true","yes")


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


def _opt_int(key: str) -> Optional[int]:
    v = os.getenv(key)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


class Config:
    """Immutable runtime configuration snapshot."""
    __slots__ = (
        "openai_api_key",
        "anthropic_api_key",
        "classifier_backend",
        "confidence_threshold",
        "max_retries",
        "verbose",
        "hf_model",
        "max_records",
        # Phase 2
        "atlas_enabled",
        "atlas_path",
        "embedder_backend",
        "embed_dim",
        "atlas_max",
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
        # Phase 2
        self.atlas_enabled        = _bool("THINKROUTER_ATLAS_ENABLED",  True)
        self.atlas_path           = os.getenv("THINKROUTER_ATLAS_PATH") or None
        self.embedder_backend     = os.getenv("THINKROUTER_EMBEDDER",   "hash")
        self.embed_dim            = _int("THINKROUTER_EMBED_DIM",        256)
        self.atlas_max            = _opt_int("THINKROUTER_ATLAS_MAX")

    def __repr__(self) -> str:
        return (
            f"Config("
            f"backend={self.classifier_backend!r}, "
            f"threshold={self.confidence_threshold}, "
            f"embedder={self.embedder_backend!r}, "
            f"atlas={self.atlas_enabled})"
        )


DEFAULT_CONFIG = Config()
