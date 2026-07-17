"""
thinkrouter.atlas
~~~~~~~~~~~~~~~~~
Query topology atlas — Phase 2.

The atlas is the growing database that makes ThinkRouter smarter with
every query it processes. It stores:

  (query_hash, embedding, domain, tier, model, provider, quality_score, timestamp)

As the atlas grows, it powers:
  Phase 3 — semantic cache (nearest-neighbor routing decisions in <2ms)
  Phase 4 — confidence model (hallucination risk from historical quality scores)
  Phase 5 — Atlas API (cross-company routing intelligence endpoint)

Storage
-------
SQLite for metadata and quality scores (Python built-in, zero deps).
NumPy binary (.npy) for the embedding matrix (fast memmap, no serialisation).
Both files live in a configurable directory (default: ~/.thinkrouter/atlas/).

Usage::

    from thinkrouter.atlas import Atlas
    from thinkrouter.domain import Domain
    from thinkrouter.constants import Tier

    atlas = Atlas()

    # Store a routing decision after inference
    record_id = atlas.store(
        query="Write a binary search tree in Python.",
        embedding=emb_vector,          # numpy float32 array
        domain=Domain.CODE,
        tier=Tier.FULL,
        model="deepseek-coder-v2",
        provider="ollama",
        quality_score=0.91,            # optional, from judge
    )

    # Find 5 nearest queries (Phase 3 — semantic cache)
    results = atlas.find_similar(query_embedding, k=5, min_score=0.85)
    for r in results:
        print(r.similarity, r.domain, r.tier, r.model)

    # Atlas statistics
    print(atlas.stats())
"""
from __future__ import annotations

import io
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from .constants import Tier
from .domain import Domain


# ── Record containers ──────────────────────────────────────────────────────

@dataclass
class AtlasRecord:
    """A single stored routing decision in the atlas."""
    id:            str
    query_hash:    str
    domain:        Domain
    tier:          Tier
    model:         str
    provider:      str
    quality_score: Optional[float]
    latency_ms:    float
    timestamp:     datetime
    query_preview: str          # first 120 chars for display


@dataclass
class SimilarResult:
    """A nearest-neighbor search result from the atlas."""
    record:     AtlasRecord
    similarity: float           # cosine similarity in [0, 1]


@dataclass
class AtlasStats:
    """Aggregate statistics about the atlas."""
    total_records:     int
    domain_counts:     dict
    tier_counts:       dict
    avg_quality:       float
    earliest:          Optional[datetime]
    latest:            Optional[datetime]
    embedding_backend: str
    storage_path:      str

    def __str__(self) -> str:
        lines = [
            "",
            "  ThinkRouter — Query Topology Atlas",
            "  " + "─" * 46,
            f"  Total records     : {self.total_records:,}",
            f"  Storage path      : {self.storage_path}",
            f"  Embedding backend : {self.embedding_backend}",
            f"  Avg quality score : {self.avg_quality:.3f}" if self.avg_quality else
            f"  Avg quality score : N/A (no scores yet)",
            "",
            "  Domain breakdown:",
        ]
        for domain, count in sorted(self.domain_counts.items(), key=lambda x: -x[1]):
            pct = count / self.total_records * 100 if self.total_records else 0
            lines.append(f"    {domain:<12} : {count:>6,}  ({pct:.1f}%)")
        lines.append("")
        lines.append("  Tier breakdown:")
        for tier, count in sorted(self.tier_counts.items(), key=lambda x: -x[1]):
            pct = count / self.total_records * 100 if self.total_records else 0
            lines.append(f"    {tier:<12} : {count:>6,}  ({pct:.1f}%)")
        if self.earliest and self.latest:
            lines.append(f"\n  First record  : {self.earliest.strftime('%Y-%m-%d %H:%M UTC')}")
            lines.append(f"  Latest record : {self.latest.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("")
        return "\n".join(lines)


# ── Atlas ──────────────────────────────────────────────────────────────────

class Atlas:
    """
    Query topology atlas — persistent storage for routing decisions.

    Stores every routed query with its embedding, domain classification,
    complexity tier, model used, and quality score. Powers Phase 3
    semantic caching and Phase 4 confidence modelling.

    Parameters
    ----------
    path : str | Path | None
        Directory where atlas files are stored.
        Default: ~/.thinkrouter/atlas/
    embedding_dim : int
        Dimensionality of stored embeddings.
        Must match the embedder used to produce them.
        Default: 256 (HashSketchEmbedder default).
    embedding_backend : str
        Label for the embedding backend — stored in stats only.
    max_records : int | None
        Cap the atlas size. When reached, oldest records are evicted.
        None = unlimited.
    read_only : bool
        Open in read-only mode (for inspection without writing).
    """

    _DB_FILE   = "atlas.db"
    _EMB_FILE  = "embeddings.npy"
    _IDX_FILE  = "index.npy"  # maps row index → record id

    def __init__(
        self,
        path:              Optional[str] = None,
        embedding_dim:     int           = 256,
        embedding_backend: str           = "hash-sketch-256",
        max_records:       Optional[int] = None,
        read_only:         bool          = False,
    ) -> None:
        self._dim       = embedding_dim
        self._backend   = embedding_backend
        self._max       = max_records
        self._read_only = read_only
        self._lock      = threading.Lock()

        # Storage path
        if path is None:
            path = os.path.join(Path.home(), ".thinkrouter", "atlas")
        self._path = Path(path)
        if not read_only:
            self._path.mkdir(parents=True, exist_ok=True)

        self._db_path  = self._path / self._DB_FILE
        self._emb_path = self._path / self._EMB_FILE
        self._idx_path = self._path / self._IDX_FILE

        self._init_db()
        self._load_embeddings()

    # ── Initialisation ─────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create SQLite schema if not present."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id            TEXT PRIMARY KEY,
                query_hash    TEXT NOT NULL,
                query_preview TEXT NOT NULL,
                domain        TEXT NOT NULL,
                tier          TEXT NOT NULL,
                model         TEXT NOT NULL,
                provider      TEXT NOT NULL,
                quality_score REAL,
                latency_ms    REAL NOT NULL DEFAULT 0.0,
                embedding_row INTEGER,
                timestamp     TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON records(domain)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tier   ON records(tier)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hash   ON records(query_hash)")
        conn.commit()
        conn.close()
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row

    def _load_embeddings(self) -> None:
        """Load embedding matrix into memory (numpy memmap or empty array)."""
        if self._emb_path.exists():
            stored = np.load(str(self._emb_path))
            # Validate dimensionality
            if stored.ndim == 2 and stored.shape[1] == self._dim:
                self._embeddings: np.ndarray = stored.copy()
            else:
                # Dimension mismatch — reset (new embedder backend)
                self._embeddings = np.empty((0, self._dim), dtype=np.float32)
        else:
            self._embeddings = np.empty((0, self._dim), dtype=np.float32)

    def _save_embeddings(self) -> None:
        """Persist embedding matrix to disk."""
        np.save(str(self._emb_path), self._embeddings)

    # ── Store ──────────────────────────────────────────────────────────────

    def store(
        self,
        query:         str,
        embedding:     "np.ndarray",
        domain:        Domain,
        tier:          Tier,
        model:         str,
        provider:      str,
        quality_score: Optional[float] = None,
        latency_ms:    float           = 0.0,
    ) -> str:
        """
        Store a routing decision in the atlas.

        Parameters
        ----------
        query         : Original query text (stored as preview only).
        embedding     : float32 numpy array of shape (dim,).
        domain        : Detected domain.
        tier          : Assigned complexity tier.
        model         : Model used for inference.
        provider      : Provider used.
        quality_score : Optional quality score in [0, 1] from a judge.
        latency_ms    : Total routing + inference latency.

        Returns
        -------
        str : Record UUID.
        """
        if self._read_only:
            raise RuntimeError("Atlas is opened in read-only mode.")

        record_id  = str(uuid.uuid4())
        query_hash = self._hash(query)
        timestamp  = datetime.now(timezone.utc).isoformat()
        preview    = query[:120].replace("\n", " ")

        # Validate and normalise embedding
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.shape != (self._dim,):
            raise ValueError(
                f"Embedding shape {vec.shape} does not match atlas dim {self._dim}. "
                f"Use get_embedder(backend, dim={self._dim}) consistently."
            )
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        with self._lock:
            # Append to embedding matrix
            row_idx = len(self._embeddings)
            self._embeddings = np.vstack([self._embeddings, vec[np.newaxis, :]])

            # Evict oldest if max_records reached
            if self._max and len(self._embeddings) > self._max:
                overflow = len(self._embeddings) - self._max
                self._embeddings = self._embeddings[-self._max:]
                # Evict oldest rows from DB
                self._conn.execute(
                    "DELETE FROM records WHERE embedding_row IN "
                    "(SELECT embedding_row FROM records ORDER BY embedding_row ASC LIMIT ?)",
                    (overflow,),
                )
                self._conn.commit()

            # Insert metadata
            self._conn.execute("""
                INSERT INTO records
                  (id, query_hash, query_preview, domain, tier, model,
                   provider, quality_score, latency_ms, embedding_row, timestamp)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                record_id, query_hash, preview,
                domain.value, tier.name, model, provider,
                quality_score, latency_ms, row_idx, timestamp,
            ))
            self._conn.commit()
            self._save_embeddings()

        return record_id

    # ── Update quality score ───────────────────────────────────────────────

    def update_quality(self, record_id: str, quality_score: float) -> None:
        """
        Update the quality score for an existing record.

        Call this after receiving human or automated feedback on a response.
        Quality scores power Phase 4 confidence modelling.

        Parameters
        ----------
        record_id     : UUID returned by store().
        quality_score : Score in [0, 1].
        """
        with self._lock:
            self._conn.execute(
                "UPDATE records SET quality_score=? WHERE id=?",
                (float(quality_score), record_id),
            )
            self._conn.commit()

    # ── Find similar ───────────────────────────────────────────────────────

    def find_similar(
        self,
        embedding:  "np.ndarray",
        k:          int   = 5,
        min_score:  float = 0.80,
        domain:     Optional[Domain] = None,
    ) -> List[SimilarResult]:
        """
        Find k nearest neighbours to a query embedding.

        This is the core operation for Phase 3 semantic caching.
        Cosine similarity over the full embedding matrix — O(n·d) where
        n = atlas size, d = embedding dim. Fast for n < 1M on CPU.

        Parameters
        ----------
        embedding  : Query embedding, float32 shape (dim,).
        k          : Maximum number of results to return.
        min_score  : Minimum cosine similarity threshold.
        domain     : If provided, filter results to this domain only.

        Returns
        -------
        List[SimilarResult] sorted by similarity descending.
        """
        if len(self._embeddings) == 0:
            return []

        vec = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        with self._lock:
            emb_matrix = self._embeddings.copy()

        # Cosine similarity = dot product of unit vectors
        sims = emb_matrix @ vec  # shape (n,)

        # Filter by minimum similarity
        candidate_rows = np.where(sims >= min_score)[0]
        if len(candidate_rows) == 0:
            return []

        # Sort by similarity descending and take top k
        sorted_rows = candidate_rows[np.argsort(-sims[candidate_rows])][:k]

        results = []
        for row in sorted_rows:
            row_int = int(row)
            sim     = float(sims[row_int])
            rec     = self._fetch_by_row(row_int)
            if rec is None:
                continue
            if domain and rec.domain != domain:
                continue
            results.append(SimilarResult(record=rec, similarity=sim))

        return results

    def _fetch_by_row(self, row: int) -> Optional[AtlasRecord]:
        cur = self._conn.execute(
            "SELECT * FROM records WHERE embedding_row=?", (row,)
        )
        r = cur.fetchone()
        if r is None:
            return None
        return self._row_to_record(r)

    def get(self, record_id: str) -> Optional[AtlasRecord]:
        """Retrieve a record by its UUID."""
        cur = self._conn.execute(
            "SELECT * FROM records WHERE id=?", (record_id,)
        )
        r = cur.fetchone()
        return self._row_to_record(r) if r else None

    def _row_to_record(self, r) -> AtlasRecord:
        return AtlasRecord(
            id=r["id"],
            query_hash=r["query_hash"],
            domain=Domain(r["domain"]),
            tier=Tier[r["tier"]],
            model=r["model"],
            provider=r["provider"],
            quality_score=r["quality_score"],
            latency_ms=r["latency_ms"],
            timestamp=datetime.fromisoformat(r["timestamp"]),
            query_preview=r["query_preview"],
        )

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> AtlasStats:
        """Return aggregate statistics about the atlas."""
        cur = self._conn.execute("""
            SELECT
                COUNT(*)                          AS total,
                AVG(quality_score)                AS avg_q,
                MIN(timestamp)                    AS earliest,
                MAX(timestamp)                    AS latest
            FROM records
        """)
        row   = cur.fetchone()
        total = row["total"] or 0
        avg_q = row["avg_q"] or 0.0

        # Domain counts
        cur = self._conn.execute(
            "SELECT domain, COUNT(*) AS n FROM records GROUP BY domain"
        )
        domain_counts = {r["domain"]: r["n"] for r in cur.fetchall()}

        # Tier counts
        cur = self._conn.execute(
            "SELECT tier, COUNT(*) AS n FROM records GROUP BY tier"
        )
        tier_counts = {r["tier"]: r["n"] for r in cur.fetchall()}

        earliest = None
        latest   = None
        if row["earliest"]:
            earliest = datetime.fromisoformat(row["earliest"])
        if row["latest"]:
            latest = datetime.fromisoformat(row["latest"])

        return AtlasStats(
            total_records=total,
            domain_counts=domain_counts,
            tier_counts=tier_counts,
            avg_quality=avg_q,
            earliest=earliest,
            latest=latest,
            embedding_backend=self._backend,
            storage_path=str(self._path),
        )

    def print_stats(self) -> None:
        print(str(self.stats()))

    # ── Export ─────────────────────────────────────────────────────────────

    def export_records(self, path: Optional[str] = None) -> str:
        """
        Export all records (without embeddings) to a JSON file.

        Useful for sharing routing data, building training datasets,
        and debugging. Embeddings are NOT exported (privacy + size).

        Parameters
        ----------
        path : Output path. Default: atlas_export_<timestamp>.json

        Returns
        -------
        str : Path of the written file.
        """
        import json

        if path is None:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(self._path / f"atlas_export_{ts}.json")

        cur = self._conn.execute("SELECT * FROM records ORDER BY timestamp")
        rows = cur.fetchall()
        data = {
            "metadata": {
                "total":              len(rows),
                "embedding_backend":  self._backend,
                "embedding_dim":      self._dim,
                "exported_at":        datetime.now(timezone.utc).isoformat(),
            },
            "records": [
                {
                    "id":            r["id"],
                    "query_preview": r["query_preview"],
                    "domain":        r["domain"],
                    "tier":          r["tier"],
                    "model":         r["model"],
                    "provider":      r["provider"],
                    "quality_score": r["quality_score"],
                    "latency_ms":    r["latency_ms"],
                    "timestamp":     r["timestamp"],
                }
                for r in rows
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _hash(text: str) -> str:
        """SHA-256 hash of query text for deduplication."""
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def __len__(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM records")
        return cur.fetchone()[0]

    def __repr__(self) -> str:
        return (
            f"Atlas("
            f"records={len(self)}, "
            f"dim={self._dim}, "
            f"backend={self._backend!r}, "
            f"path={str(self._path)!r})"
        )

    def close(self) -> None:
        """Close SQLite connection."""
        if hasattr(self, "_conn"):
            self._conn.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
