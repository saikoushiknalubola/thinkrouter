"""
thinkrouter.cache
~~~~~~~~~~~~~~~~~
Semantic cache — Phase 3.

Before running any classifier, ThinkRouter checks the atlas for a
previous query with the same semantic intent. If a sufficiently similar
query exists with a reliable routing decision, that decision is returned
immediately — skipping both classifiers entirely.

This is the first visible effect of the data flywheel: routing decisions
grounded in real empirical outcomes rather than heuristic rules alone.

How it works
------------
1. Embed the incoming query (0.1–0.3ms).
2. Compute cosine similarity against all atlas embeddings (matrix multiply).
3. If the nearest neighbour has:
     - similarity >= threshold (default 0.92)
     - quality_score >= min_quality (default 0.70) OR quality_score is None
4. Return CacheResult with the stored domain, tier, and model.
5. Router skips domain + complexity classifiers and goes straight to inference.

Cache hit rate grows as the atlas fills. At 10,000 records covering
common intent clusters, the majority of routine queries hit the cache.

Usage::

    from thinkrouter.cache import SemanticCache
    from thinkrouter.embedder import HashSketchEmbedder
    from thinkrouter.atlas import Atlas

    emb   = HashSketchEmbedder()
    atlas = Atlas(embedding_dim=emb.dim)
    cache = SemanticCache(atlas=atlas, embedder=emb)

    # Check cache before classifiers
    vec    = emb.embed("Write a binary search function in Python.")
    result = cache.lookup(vec)
    if result:
        print(result.domain, result.tier, result.model)
        print(f"Cache hit! similarity={result.similarity:.4f}")
    else:
        print("Cache miss — running classifiers")

    # Print cache performance
    cache.print_stats()
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from .constants import Tier
from .domain import Domain

if TYPE_CHECKING:
    import numpy as np
    from .atlas import Atlas
    from .embedder import BaseEmbedder


# ── Cache result ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CacheResult:
    """
    A routing decision returned by the semantic cache.

    Attributes
    ----------
    domain          : Cached domain classification.
    tier            : Cached complexity tier.
    model           : Model used in the cached routing decision.
    provider        : Provider used in the cached routing decision.
    similarity      : Cosine similarity between query and cached record.
    quality_score   : Quality score of the cached response (if available).
    source_id       : Atlas record UUID that matched.
    source_preview  : First 80 chars of the matched query (for debugging).
    latency_ms      : Time to produce this cache result.
    """
    domain:         Domain
    tier:           Tier
    model:          str
    provider:       str
    similarity:     float
    quality_score:  Optional[float]
    source_id:      str
    source_preview: str
    latency_ms:     float

    @property
    def is_high_confidence(self) -> bool:
        """True when similarity is very high (>= 0.95) regardless of quality."""
        return self.similarity >= 0.95

    def __repr__(self) -> str:
        return (
            f"CacheResult("
            f"domain={self.domain.value}, "
            f"tier={self.tier.name}, "
            f"model={self.model!r}, "
            f"similarity={self.similarity:.4f}, "
            f"latency={self.latency_ms:.2f}ms)"
        )


# ── Cache stats ────────────────────────────────────────────────────────────

@dataclass
class CacheStats:
    """Aggregate performance statistics for the semantic cache."""
    total_lookups:   int
    cache_hits:      int
    cache_misses:    int
    hit_rate:        float
    avg_hit_sim:     float
    avg_hit_lat_ms:  float
    avg_miss_lat_ms: float
    atlas_size:      int
    since:           Optional[datetime]

    def __str__(self) -> str:
        lines = [
            "",
            "  ThinkRouter — Semantic Cache Stats",
            "  " + "─" * 44,
            f"  Atlas size         : {self.atlas_size:,}",
            f"  Total lookups      : {self.total_lookups:,}",
            f"  Cache hits         : {self.cache_hits:,}",
            f"  Cache misses       : {self.cache_misses:,}",
            f"  Hit rate           : {self.hit_rate:.1f}%",
            f"  Avg hit similarity : {self.avg_hit_sim:.4f}",
            f"  Avg hit latency    : {self.avg_hit_lat_ms:.2f} ms",
            f"  Avg miss latency   : {self.avg_miss_lat_ms:.2f} ms",
        ]
        if self.since:
            lines.append(
                f"  Tracking since     : {self.since.strftime('%Y-%m-%d %H:%M UTC')}"
            )
        lines.append("")
        return "\n".join(lines)


# ── Semantic cache ─────────────────────────────────────────────────────────

class SemanticCache:
    """
    Semantic cache backed by the ThinkRouter atlas.

    Checks the atlas for a previously-seen query with the same intent
    before running any classifier. On a hit, returns the stored routing
    decision. On a miss, returns None — the router proceeds normally.

    Parameters
    ----------
    atlas           : Atlas instance to query.
    embedder        : Embedder used to produce query vectors.
    threshold       : Minimum cosine similarity for a cache hit.
                      Default: 0.92 — high enough to avoid false positives.
                      Range: 0.80 (broad) → 0.99 (near-exact match only).
    min_quality     : Minimum quality_score on the cached record.
                      Records with no quality score (None) always qualify.
                      Set to 0.0 to use any cached record regardless of quality.
                      Default: 0.70.
    min_atlas_size  : Do not use the cache until the atlas has at least
                      this many records. Prevents cold-start false positives.
                      Default: 50.
    domain_filter   : If set, only return hits from this domain.
                      Useful for domain-specific deployments.
    """

    def __init__(
        self,
        atlas:          "Atlas",
        embedder:       "BaseEmbedder",
        threshold:      float           = 0.92,
        min_quality:    float           = 0.70,
        min_atlas_size: int             = 50,
        domain_filter:  Optional[Domain] = None,
    ) -> None:
        self._atlas          = atlas
        self._embedder       = embedder
        self.threshold       = threshold
        self.min_quality     = min_quality
        self.min_atlas_size  = min_atlas_size
        self.domain_filter   = domain_filter

        # Stats (thread-safe)
        self._lock         = threading.Lock()
        self._lookups      = 0
        self._hits         = 0
        self._sum_hit_sim  = 0.0
        self._sum_hit_lat  = 0.0
        self._sum_miss_lat = 0.0
        self._since: Optional[datetime] = None

    # ── Core lookup ────────────────────────────────────────────────────────

    def lookup(
        self,
        embedding:     "np.ndarray",
        domain_hint:   Optional[Domain] = None,
    ) -> Optional[CacheResult]:
        """
        Look up a routing decision for the given query embedding.

        Parameters
        ----------
        embedding    : Pre-computed query embedding from the same embedder.
        domain_hint  : If provided, only return hits from this domain.
                       Overrides the instance-level domain_filter.

        Returns
        -------
        CacheResult if a sufficiently similar record is found, else None.
        """
        t0 = time.perf_counter()

        with self._lock:
            self._lookups += 1
            if self._since is None:
                self._since = datetime.now(timezone.utc)

        # Skip cache if atlas too small
        atlas_size = len(self._atlas)
        if atlas_size < self.min_atlas_size:
            ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self._sum_miss_lat += ms
            return None

        # Domain filter (instance-level or per-call)
        domain = domain_hint or self.domain_filter

        # Find nearest neighbours
        results = self._atlas.find_similar(
            embedding=embedding,
            k=5,                    # check top-5, pick best qualifying
            min_score=self.threshold,
            domain=domain,
        )

        ms = (time.perf_counter() - t0) * 1000

        for sim_result in results:
            rec = sim_result.record
            sim = sim_result.similarity

            # Quality gate: skip low-quality cached records
            if rec.quality_score is not None and rec.quality_score < self.min_quality:
                continue

            # Hit — record stats and return
            with self._lock:
                self._hits       += 1
                self._sum_hit_sim += sim
                self._sum_hit_lat += ms

            return CacheResult(
                domain=rec.domain,
                tier=rec.tier,
                model=rec.model,
                provider=rec.provider,
                similarity=sim,
                quality_score=rec.quality_score,
                source_id=rec.id,
                source_preview=rec.query_preview,
                latency_ms=ms,
            )

        # Miss
        with self._lock:
            self._sum_miss_lat += ms
        return None

    # ── Full lookup including embed ────────────────────────────────────────

    def lookup_query(
        self,
        query:       str,
        domain_hint: Optional[Domain] = None,
    ) -> Optional[CacheResult]:
        """
        Embed a query string and look it up in one call.

        Convenience method for cases where the embedding is not
        pre-computed. When using inside ThinkRouter, embed() is called
        separately so the vector can also be stored in the atlas.

        Parameters
        ----------
        query       : Raw query string.
        domain_hint : Restrict hits to this domain.
        """
        vec = self._embedder.embed(query)
        return self.lookup(vec, domain_hint=domain_hint)

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> CacheStats:
        """Return current cache performance statistics."""
        with self._lock:
            total      = self._lookups
            hits       = self._hits
            misses     = total - hits
            hit_rate   = (hits / total * 100) if total else 0.0
            avg_sim    = (self._sum_hit_sim / hits) if hits else 0.0
            avg_h_lat  = (self._sum_hit_lat / hits) if hits else 0.0
            avg_m_lat  = (self._sum_miss_lat / max(misses, 1))
            since      = self._since

        return CacheStats(
            total_lookups=total,
            cache_hits=hits,
            cache_misses=misses,
            hit_rate=hit_rate,
            avg_hit_sim=avg_sim,
            avg_hit_lat_ms=avg_h_lat,
            avg_miss_lat_ms=avg_m_lat,
            atlas_size=len(self._atlas),
            since=since,
        )

    def print_stats(self) -> None:
        print(str(self.stats()))

    def reset_stats(self) -> None:
        """Reset performance counters (does not clear the atlas)."""
        with self._lock:
            self._lookups      = 0
            self._hits         = 0
            self._sum_hit_sim  = 0.0
            self._sum_hit_lat  = 0.0
            self._sum_miss_lat = 0.0
            self._since        = None

    # ── Warm-up ────────────────────────────────────────────────────────────

    def warmup(self, queries: list, domains=None, tiers=None,
               models=None, providers=None) -> int:
        """
        Pre-populate the atlas with known routing decisions.

        Use this to seed the cache before production traffic arrives.
        Useful for deploying ThinkRouter into an existing system where
        you know the common query types.

        Parameters
        ----------
        queries   : List of representative query strings.
        domains   : List of Domain values (one per query). Default: GENERAL.
        tiers     : List of Tier values. Default: FULL.
        models    : List of model strings. Default: "gpt-4o".
        providers : List of provider strings. Default: "openai".

        Returns
        -------
        int : Number of records stored.
        """
        n = len(queries)
        _domains   = domains   or [Domain.GENERAL] * n
        _tiers     = tiers     or [Tier.FULL]      * n
        _models    = models    or ["gpt-4o"]        * n
        _providers = providers or ["openai"]        * n

        stored = 0
        for q, dom, tier, model, prov in zip(
            queries, _domains, _tiers, _models, _providers
        ):
            try:
                vec = self._embedder.embed(q)
                self._atlas.store(
                    query=q, embedding=vec,
                    domain=dom, tier=tier,
                    model=model, provider=prov,
                    quality_score=None,
                )
                stored += 1
            except Exception:
                continue
        return stored

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"SemanticCache("
            f"threshold={self.threshold}, "
            f"atlas={s.atlas_size}, "
            f"hit_rate={s.hit_rate:.1f}%)"
        )
