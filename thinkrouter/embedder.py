"""
thinkrouter.embedder
~~~~~~~~~~~~~~~~~~~~
Query embedding module — Phase 2.

Three interchangeable backends, zero to full accuracy:

  HashSketchEmbedder   — zero dependencies, instant, deterministic.
                         Uses character n-gram frequency hashing to produce
                         a 256-dimensional sparse vector. Accurate enough
                         for nearest-neighbor lookup when the atlas is small
                         (<50k queries). No API key, no model download.

  OpenAIEmbedder       — text-embedding-3-small via OpenAI API.
                         1536-dimensional dense vector. $0.02/million tokens —
                         effectively free at routing scale. Best quality.
                         Requires pip install thinkrouter[openai].

  LocalEmbedder        — sentence-transformers all-MiniLM-L6-v2.
                         384-dimensional dense vector. Runs fully offline.
                         ~80MB model download on first use.
                         Requires pip install thinkrouter[embeddings].

All three expose the same interface::

    emb = HashSketchEmbedder()
    vec = emb.embed("Write a binary search tree in Python.")
    # numpy array, shape (256,), dtype float32

    vecs = emb.embed_batch(["query 1", "query 2"])
    # numpy array, shape (2, 256), dtype float32

Usage::

    from thinkrouter.embedder import get_embedder

    emb = get_embedder("hash")          # zero deps
    emb = get_embedder("openai")        # API-based, best quality
    emb = get_embedder("local")         # offline, good quality
"""
from __future__ import annotations

import hashlib
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .exceptions import ClassifierError, ConfigurationError


# ── Result ─────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    """
    Output of a single embedding call.

    Attributes
    ----------
    vector     : numpy float32 array of shape (dim,).
    dim        : Embedding dimensionality.
    latency_ms : Wall-clock time for the embedding call.
    backend    : Backend that produced this embedding.
    tokens     : Approximate token count (for cost tracking).
    """
    vector:     "np.ndarray"
    dim:        int
    latency_ms: float
    backend:    str
    tokens:     int = 0

    def __repr__(self) -> str:
        return (
            f"EmbeddingResult("
            f"dim={self.dim}, "
            f"backend={self.backend!r}, "
            f"latency={self.latency_ms:.1f}ms)"
        )


# ── Base class ──────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):

    @abstractmethod
    def embed(self, text: str) -> "np.ndarray":
        """Embed a single string. Returns float32 array of shape (dim,)."""

    def embed_batch(self, texts: List[str]) -> "np.ndarray":
        """Embed a list of strings. Returns float32 array of shape (n, dim)."""
        return np.stack([self.embed(t) for t in texts])

    def embed_with_meta(self, text: str) -> EmbeddingResult:
        """Embed a string and return EmbeddingResult with metadata."""
        t0  = time.perf_counter()
        vec = self.embed(text)
        ms  = (time.perf_counter() - t0) * 1000
        return EmbeddingResult(
            vector=vec, dim=len(vec),
            latency_ms=ms, backend=self._backend_name,
            tokens=len(text.split()),
        )

    @property
    def _backend_name(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""


# ── Hash-sketch embedder (zero dependencies) ────────────────────────────────

class HashSketchEmbedder(BaseEmbedder):
    """
    Zero-dependency embedding using character n-gram frequency hashing.

    Produces a 256-dimensional sparse float32 vector by:
      1. Extracting character trigrams from the lowercased text.
      2. Hashing each trigram to a bucket in [0, dim).
      3. Counting trigram frequency per bucket.
      4. L2-normalising the result.

    This is a bag-of-trigrams hash embedding — no neural network, no
    model download. Similarity is meaningful for queries in the same
    semantic neighbourhood (same domain, similar intent) and breaks
    down for very long-range semantic comparisons. Suitable for atlas
    sizes up to ~100k queries.

    Parameters
    ----------
    dim : int
        Embedding dimensionality. Default: 256.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def _backend_name(self) -> str:
        return f"hash-sketch-{self._dim}"

    def embed(self, text: str) -> "np.ndarray":
        vec   = np.zeros(self._dim, dtype=np.float32)
        lower = text.lower()

        # Character trigrams
        for i in range(len(lower) - 2):
            gram = lower[i : i + 3]
            h    = int(hashlib.md5(gram.encode()).hexdigest(), 16)
            vec[h % self._dim] += 1.0

        # Word unigrams (higher signal)
        for word in lower.split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % self._dim] += 2.0

        # Word bigrams
        words = lower.split()
        for i in range(len(words) - 1):
            bigram = words[i] + " " + words[i + 1]
            h      = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
            vec[h % self._dim] += 1.5

        # L2 normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


# ── OpenAI embedder ─────────────────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI text-embedding-3-small embedding backend.

    Produces 1536-dimensional dense float32 vectors.
    Cost: $0.02 per million tokens — effectively free for routing overhead.
    Requires: pip install thinkrouter[openai]

    Parameters
    ----------
    api_key    : OpenAI API key. Falls back to OPENAI_API_KEY env var.
    model      : Embedding model identifier.
    dimensions : Output dimensionality (1–1536). Use 512 to save storage.
    """

    MODEL = "text-embedding-3-small"

    def __init__(
        self,
        api_key:    Optional[str] = None,
        model:      str = MODEL,
        dimensions: int = 512,  # reduced from 1536 to save atlas storage
    ) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise ConfigurationError(
                "OpenAIEmbedder requires:  pip install thinkrouter[openai]"
            ) from exc

        import os
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ConfigurationError(
                "No OpenAI API key. Pass api_key= or set OPENAI_API_KEY."
            )
        self._client     = _openai.OpenAI(api_key=key)
        self._model      = model
        self._dimensions = dimensions

    @property
    def dim(self) -> int:
        return self._dimensions

    @property
    def _backend_name(self) -> str:
        return f"openai:{self._model}:{self._dimensions}d"

    def embed(self, text: str) -> "np.ndarray":
        resp = self._client.embeddings.create(
            input=text,
            model=self._model,
            dimensions=self._dimensions,
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        # L2-normalise (OpenAI returns unit vectors but ensure it)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: List[str]) -> "np.ndarray":
        """Batch embedding — one API call for multiple texts."""
        resp = self._client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=self._dimensions,
        )
        vecs = np.array(
            [item.embedding for item in resp.data], dtype=np.float32
        )
        # L2-normalise each row
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


# ── Local sentence-transformers embedder ────────────────────────────────────

class LocalEmbedder(BaseEmbedder):
    """
    Offline embedding using sentence-transformers all-MiniLM-L6-v2.

    Produces 384-dimensional dense float32 vectors.
    First call downloads the model (~80MB). Subsequent calls use cache.
    Runs fully offline — no API key, no internet required after download.
    Requires: pip install thinkrouter[embeddings]

    Parameters
    ----------
    model_name : sentence-transformers model identifier.
    device     : "cpu" | "cuda". Auto-detects GPU if not specified.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str           = DEFAULT_MODEL,
        device:     Optional[str] = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigurationError(
                "LocalEmbedder requires:  pip install thinkrouter[embeddings]"
            ) from exc

        import torch
        _device         = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model     = SentenceTransformer(model_name, device=_device)
        self._device    = _device
        self._dim_size  = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim_size

    @property
    def _backend_name(self) -> str:
        return f"local:{self._device}"

    def embed(self, text: str) -> "np.ndarray":
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> "np.ndarray":
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vecs.astype(np.float32)


# ── Factory ─────────────────────────────────────────────────────────────────

def get_embedder(backend: str = "hash", **kwargs) -> BaseEmbedder:
    """
    Instantiate an embedder by backend name.

    Parameters
    ----------
    backend : "hash" (default, zero deps) | "openai" | "local"
    **kwargs : Passed to the embedder constructor.

    Examples
    --------
    >>> emb = get_embedder("hash")
    >>> emb = get_embedder("openai", api_key="sk-...", dimensions=512)
    >>> emb = get_embedder("local", model_name="all-MiniLM-L6-v2")
    """
    if backend == "hash":
        return HashSketchEmbedder(**kwargs)
    if backend == "openai":
        return OpenAIEmbedder(**kwargs)
    if backend == "local":
        return LocalEmbedder(**kwargs)
    raise ValueError(
        f"Unknown embedder backend: {backend!r}. "
        "Choose 'hash', 'openai', or 'local'."
    )
