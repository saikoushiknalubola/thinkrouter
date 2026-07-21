"""
thinkrouter.atlas
~~~~~~~~~~~~~~~~~
Query topology atlas — Phase 2.

Thread safety: every thread gets its own SQLite connection via
threading.local(). The numpy embedding matrix is protected by a
single threading.Lock() for writes only. Reads copy the matrix
before releasing the lock and then do pure numpy — zero DB access.
This eliminates all SQLite concurrency issues permanently.
"""
from __future__ import annotations

import os
import sqlite3
import threading
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
    id:            str
    query_hash:    str
    domain:        Domain
    tier:          Tier
    model:         str
    provider:      str
    quality_score: Optional[float]
    latency_ms:    float
    timestamp:     datetime
    query_preview: str


@dataclass
class SimilarResult:
    record:     AtlasRecord
    similarity: float


@dataclass
class AtlasStats:
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
        ]
        if self.avg_quality:
            lines.append(f"  Avg quality score : {self.avg_quality:.3f}")
        else:
            lines.append("  Avg quality score : N/A")
        lines.append("")
        lines.append("  Domain breakdown:")
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

_SCHEMA = """
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
);
CREATE INDEX IF NOT EXISTS idx_domain ON records(domain);
CREATE INDEX IF NOT EXISTS idx_tier   ON records(tier);
CREATE INDEX IF NOT EXISTS idx_hash   ON records(query_hash);
CREATE INDEX IF NOT EXISTS idx_row    ON records(embedding_row);
"""


class Atlas:
    """
    Persistent query topology atlas.

    Thread safety model
    -------------------
    - Each thread gets its own SQLite connection via threading.local().
      SQLite WAL mode allows many concurrent readers and one writer
      without any application-level locking.
    - The numpy embedding matrix (self._embeddings) is protected by
      self._matrix_lock for WRITES only. Reads copy the matrix while
      holding the lock, then release it and do all numpy work outside.
    - No single lock covers both DB and matrix simultaneously, so there
      is no deadlock risk.
    """

    _DB_FILE  = "atlas.db"
    _EMB_FILE = "embeddings.npy"

    def __init__(
        self,
        path:              Optional[str] = None,
        embedding_dim:     int           = 256,
        embedding_backend: str           = "hash-sketch-256",
        max_records:       Optional[int] = None,
        read_only:         bool          = False,
    ) -> None:
        self._dim         = embedding_dim
        self._backend     = embedding_backend
        self._max         = max_records
        self._read_only   = read_only
        self._matrix_lock = threading.Lock()
        self._local       = threading.local()   # per-thread SQLite connections

        if path is None:
            path = os.path.join(Path.home(), ".thinkrouter", "atlas")
        self._path    = Path(path)
        self._db_path = self._path / self._DB_FILE
        self._emb_path= self._path / self._EMB_FILE

        if not read_only:
            self._path.mkdir(parents=True, exist_ok=True)

        # Initialise schema via a dedicated setup connection
        self._init_schema()
        # Load embedding matrix into memory
        self._load_embeddings()

    # ── Per-thread connection ──────────────────────────────────────────────

    @property
    def _conn(self) -> sqlite3.Connection:
        """
        Return a SQLite connection private to the calling thread.
        Created on first access, reused on subsequent calls from the same thread.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    # ── Schema ─────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()

    # ── Embedding matrix ───────────────────────────────────────────────────

    def _load_embeddings(self) -> None:
        if self._emb_path.exists():
            stored = np.load(str(self._emb_path))
            if stored.ndim == 2 and stored.shape[1] == self._dim:
                self._embeddings: np.ndarray = stored.copy()
                return
        self._embeddings = np.empty((0, self._dim), dtype=np.float32)

    def _save_embeddings(self) -> None:
        """Must be called while holding self._matrix_lock."""
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
        if self._read_only:
            raise RuntimeError("Atlas opened in read-only mode.")

        record_id  = str(uuid.uuid4())
        query_hash = self._hash(query)
        timestamp  = datetime.now(timezone.utc).isoformat()
        preview    = query[:120].replace("\n", " ")

        vec  = np.asarray(embedding, dtype=np.float32)
        if vec.shape != (self._dim,):
            raise ValueError(
                f"Embedding shape {vec.shape} != atlas dim ({self._dim},). "
                f"Use the same embedder backend consistently."
            )
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Protect matrix write + DB write together
        with self._matrix_lock:
            row_idx = len(self._embeddings)
            self._embeddings = np.vstack([self._embeddings, vec[np.newaxis, :]])

            # Evict oldest if capped
            if self._max and len(self._embeddings) > self._max:
                overflow = len(self._embeddings) - self._max
                self._embeddings = self._embeddings[-self._max:]
                self._conn.execute(
                    "DELETE FROM records WHERE embedding_row IN "
                    "(SELECT embedding_row FROM records "
                    " ORDER BY embedding_row ASC LIMIT ?)",
                    (overflow,),
                )
                self._conn.commit()

            self._conn.execute(
                "INSERT INTO records "
                "(id,query_hash,query_preview,domain,tier,model,"
                " provider,quality_score,latency_ms,embedding_row,timestamp) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (record_id, query_hash, preview,
                 domain.value, tier.name, model, provider,
                 quality_score, latency_ms, row_idx, timestamp),
            )
            self._conn.commit()
            self._save_embeddings()

        return record_id

    # ── Update quality ─────────────────────────────────────────────────────

    def update_quality(self, record_id: str, quality_score: float) -> None:
        self._conn.execute(
            "UPDATE records SET quality_score=? WHERE id=?",
            (float(quality_score), record_id),
        )
        self._conn.commit()

    # ── Find similar ───────────────────────────────────────────────────────

    def find_similar(
        self,
        embedding:  "np.ndarray",
        k:          int            = 5,
        min_score:  float          = 0.80,
        domain:     Optional[Domain] = None,
    ) -> List[SimilarResult]:
        """
        Return k nearest neighbours by cosine similarity.

        Thread safety: copies the matrix under matrix_lock, then releases it.
        All numpy work and DB reads happen outside any lock — each thread
        uses its own SQLite connection so no coordination is needed.
        """
        with self._matrix_lock:
            if len(self._embeddings) == 0:
                return []
            emb_matrix = self._embeddings.copy()   # snapshot — released immediately

        vec = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Pure numpy — no shared state, no lock needed
        sims = emb_matrix @ vec
        candidate_rows = np.where(sims >= min_score)[0]
        if len(candidate_rows) == 0:
            return []
        sorted_rows = candidate_rows[np.argsort(-sims[candidate_rows])][:k]

        # DB reads — use this thread's private connection, no lock needed
        results: List[SimilarResult] = []
        for row in sorted_rows:
            row_int = int(row)
            sim     = float(sims[row_int])
            try:
                cur = self._conn.execute(
                    "SELECT * FROM records WHERE embedding_row=?", (row_int,)
                )
                r = cur.fetchone()
            except Exception:
                continue
            if r is None:
                continue
            rec = self._row_to_record(r)
            if rec is None:
                continue
            if domain and rec.domain != domain:
                continue
            results.append(SimilarResult(record=rec, similarity=sim))

        return results

    # ── Get ────────────────────────────────────────────────────────────────

    def get(self, record_id: str) -> Optional[AtlasRecord]:
        try:
            cur = self._conn.execute(
                "SELECT * FROM records WHERE id=?", (record_id,)
            )
            r = cur.fetchone()
        except Exception:
            return None
        return self._row_to_record(r) if r else None

    # ── Row → record ───────────────────────────────────────────────────────

    def _row_to_record(self, r) -> Optional[AtlasRecord]:
        if r is None:
            return None
        try:
            ts = r["timestamp"]
            return AtlasRecord(
                id=r["id"],
                query_hash=r["query_hash"],
                domain=Domain(r["domain"]),
                tier=Tier[r["tier"]],
                model=r["model"],
                provider=r["provider"],
                quality_score=r["quality_score"],
                latency_ms=r["latency_ms"] or 0.0,
                timestamp=datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc),
                query_preview=r["query_preview"] or "",
            )
        except Exception:
            return None

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> AtlasStats:
        try:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS total, AVG(quality_score) AS avg_q, "
                "MIN(timestamp) AS earliest, MAX(timestamp) AS latest "
                "FROM records"
            )
            row   = cur.fetchone()
            total = row["total"] or 0
            avg_q = row["avg_q"] or 0.0

            cur = self._conn.execute(
                "SELECT domain, COUNT(*) AS n FROM records GROUP BY domain"
            )
            domain_counts = {r["domain"]: r["n"] for r in cur.fetchall()}

            cur = self._conn.execute(
                "SELECT tier, COUNT(*) AS n FROM records GROUP BY tier"
            )
            tier_counts = {r["tier"]: r["n"] for r in cur.fetchall()}

            earliest = datetime.fromisoformat(row["earliest"]) if row["earliest"] else None
            latest   = datetime.fromisoformat(row["latest"])   if row["latest"]   else None
        except Exception:
            total = 0
            avg_q = 0.0
            domain_counts = {}
            tier_counts = {}
            earliest = latest = None

        return AtlasStats(
            total_records=total, domain_counts=domain_counts,
            tier_counts=tier_counts, avg_quality=avg_q,
            earliest=earliest, latest=latest,
            embedding_backend=self._backend,
            storage_path=str(self._path),
        )

    def print_stats(self) -> None:
        print(str(self.stats()))

    # ── Export ─────────────────────────────────────────────────────────────

    def export_records(self, path: Optional[str] = None) -> str:
        import json
        if path is None:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(self._path / f"atlas_export_{ts}.json")
        cur  = self._conn.execute("SELECT * FROM records ORDER BY timestamp")
        rows = cur.fetchall()
        data = {
            "metadata": {
                "total":             len(rows),
                "embedding_backend": self._backend,
                "embedding_dim":     self._dim,
                "exported_at":       datetime.now(timezone.utc).isoformat(),
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
                } for r in rows
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _hash(text: str) -> str:
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def __len__(self) -> int:
        try:
            cur = self._conn.execute("SELECT COUNT(*) FROM records")
            return cur.fetchone()[0]
        except Exception:
            return 0

    def __repr__(self) -> str:
        return (
            f"Atlas(records={len(self)}, dim={self._dim}, "
            f"backend={self._backend!r}, path={str(self._path)!r})"
        )

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
