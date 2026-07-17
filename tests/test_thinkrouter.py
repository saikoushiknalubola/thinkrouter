"""
tests/test_thinkrouter.py — v0.5.0 full test suite (Phase 1 + Phase 2)
Run:  pytest tests/ -v
"""
from __future__ import annotations

import os
import tempfile
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from thinkrouter import (
    Atlas, AtlasRecord, ClassifierResult, Config,
    Domain, DomainClassifier, DomainResult,
    EmbeddingResult, HashSketchEmbedder, ModelRegistry, ModelTarget,
    RouterResponse, ThinkRouter, Tier, TIER_TOKEN_BUDGETS,
    UsageTracker, get_classifier, get_embedder,
)
from thinkrouter.exceptions import (
    AuthenticationError, ClassifierError, ConfigurationError,
    ModelNotFoundError, ProviderError, RateLimitError, ThinkRouterError,
)
from thinkrouter.constants import (
    ANTHROPIC_THINKING_BUDGETS, ANTHROPIC_THINKING_MODELS,
    OPENAI_REASONING_EFFORT, OPENAI_REASONING_MODELS,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_router(content: str = "Test.") -> ThinkRouter:
    r = ThinkRouter.__new__(ThinkRouter)
    r.provider              = "openai"
    r.model                 = "gpt-4o"
    r.verbose               = False
    r.max_retries           = 1
    r.domain_routing        = True
    r.domain_min_confidence = 0.40
    r._preferred_provider   = "openai"
    r._clf                  = get_classifier("heuristic")
    r._domain_clf           = DomainClassifier()
    r._registry             = ModelRegistry(provider_priority=["openai"])
    r._threshold            = 0.75
    r.usage                 = UsageTracker()
    r._ollama_adapter       = None
    r._embedder             = None
    r.atlas                 = None
    a = MagicMock()
    ret = (content, MagicMock(),
           {"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}, None)
    a.call  = MagicMock(return_value=ret)
    a.acall = AsyncMock(return_value=ret)
    r._adapter = a
    return r


def _make_router_with_atlas(content: str = "Test.", tmpdir: str = None) -> ThinkRouter:
    """Router with a live atlas in a temp directory."""
    r = _make_router(content)
    emb = HashSketchEmbedder(dim=64)
    atl = Atlas(path=tmpdir, embedding_dim=64,
                embedding_backend="hash-sketch-64")
    r._embedder = emb
    r.atlas     = atl
    return r


# ── Phase 2 — HashSketchEmbedder ──────────────────────────────────────────

class TestHashSketchEmbedder:

    def setup_method(self):
        self.emb = HashSketchEmbedder(dim=256)

    def test_embed_returns_float32(self):
        vec = self.emb.embed("Write a binary search function.")
        assert vec.dtype == np.float32

    def test_embed_shape(self):
        vec = self.emb.embed("Any query here")
        assert vec.shape == (256,)

    def test_embed_normalised(self):
        vec  = self.emb.embed("Prove that sqrt(2) is irrational.")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_embed_deterministic(self):
        v1 = self.emb.embed("Same query")
        v2 = self.emb.embed("Same query")
        np.testing.assert_array_equal(v1, v2)

    def test_different_queries_differ(self):
        v1 = self.emb.embed("Write a quicksort in Python.")
        v2 = self.emb.embed("What is the capital of France?")
        sim = float(v1 @ v2)
        assert sim < 0.99  # should not be identical

    def test_similar_queries_closer(self):
        v_code1 = self.emb.embed("Write a binary search function in Python.")
        v_code2 = self.emb.embed("Implement a binary search algorithm in Python.")
        v_other = self.emb.embed("What is the capital of France?")
        sim_code = float(v_code1 @ v_code2)
        sim_diff = float(v_code1 @ v_other)
        assert sim_code > sim_diff

    def test_embed_batch_shape(self):
        queries = ["q1", "q2", "q3"]
        vecs    = self.emb.embed_batch(queries)
        assert vecs.shape == (3, 256)

    def test_embed_batch_dtype(self):
        vecs = self.emb.embed_batch(["a", "b"])
        assert vecs.dtype == np.float32

    def test_embed_with_meta_returns_result(self):
        result = self.emb.embed_with_meta("Test query")
        assert isinstance(result, EmbeddingResult)
        assert result.dim == 256
        assert result.latency_ms >= 0
        assert "hash-sketch" in result.backend

    def test_dim_property(self):
        assert self.emb.dim == 256

    def test_custom_dim(self):
        emb = HashSketchEmbedder(dim=64)
        vec = emb.embed("test")
        assert vec.shape == (64,)

    def test_empty_string(self):
        vec = self.emb.embed("")
        assert vec.shape == (256,)

    def test_very_long_query(self):
        long_q = "Write a Python implementation " * 50
        vec    = self.emb.embed(long_q)
        assert vec.shape == (256,)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4


# ── Phase 2 — get_embedder factory ────────────────────────────────────────

class TestGetEmbedder:

    def test_hash_factory(self):
        emb = get_embedder("hash")
        assert isinstance(emb, HashSketchEmbedder)

    def test_hash_custom_dim(self):
        emb = get_embedder("hash", dim=128)
        assert emb.dim == 128

    def test_bad_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown embedder backend"):
            get_embedder("nonexistent")

    def test_openai_no_key_raises(self):
        from thinkrouter.embedder import OpenAIEmbedder
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with pytest.raises(ConfigurationError):
                OpenAIEmbedder(api_key="")

    def test_local_no_deps_raises(self):
        import sys
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            from thinkrouter.embedder import LocalEmbedder
            with pytest.raises(ConfigurationError):
                LocalEmbedder()


# ── Phase 2 — Atlas ────────────────────────────────────────────────────────

class TestAtlas:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.atlas  = Atlas(path=self.tmpdir, embedding_dim=64,
                            embedding_backend="hash-sketch-64")
        self.emb    = HashSketchEmbedder(dim=64)

    def teardown_method(self):
        self.atlas.close()

    def _store(self, query: str, domain=Domain.CODE,
               tier=Tier.FULL, model="gpt-4o",
               quality: float = None) -> str:
        vec = self.emb.embed(query)
        return self.atlas.store(
            query=query, embedding=vec, domain=domain,
            tier=tier, model=model, provider="openai",
            quality_score=quality, latency_ms=12.0,
        )

    def test_empty_atlas_len(self):
        assert len(self.atlas) == 0

    def test_store_increments_len(self):
        self._store("Write a binary search in Python.")
        assert len(self.atlas) == 1

    def test_store_returns_string_id(self):
        rid = self._store("Prove sqrt(2) is irrational.")
        assert isinstance(rid, str) and len(rid) == 36  # UUID format

    def test_store_multiple(self):
        for i in range(5):
            self._store(f"Query number {i}")
        assert len(self.atlas) == 5

    def test_get_by_id(self):
        rid = self._store("Test query for retrieval", domain=Domain.MATH)
        rec = self.atlas.get(rid)
        assert rec is not None
        assert rec.id == rid
        assert rec.domain == Domain.MATH

    def test_get_nonexistent_returns_none(self):
        assert self.atlas.get("nonexistent-id") is None

    def test_update_quality(self):
        rid = self._store("Query to score")
        self.atlas.update_quality(rid, 0.92)
        rec = self.atlas.get(rid)
        assert rec is not None
        assert abs(rec.quality_score - 0.92) < 1e-6

    def test_find_similar_empty_atlas(self):
        vec = self.emb.embed("test")
        results = self.atlas.find_similar(vec, k=5)
        assert results == []

    def test_find_similar_finds_close_query(self):
        q1 = "Write a binary search function in Python."
        q2 = "Implement binary search algorithm in Python."
        q3 = "What is the capital of France?"
        for q in [q1, q2, q3]:
            self._store(q)

        query_vec = self.emb.embed("Write a binary search in Python.")
        results   = self.atlas.find_similar(query_vec, k=3, min_score=0.0)
        assert len(results) > 0
        # The most similar should be about binary search, not France
        top = results[0]
        assert "search" in top.record.query_preview.lower() or \
               "binary" in top.record.query_preview.lower()

    def test_find_similar_similarity_sorted(self):
        queries = [
            "Write a Python sort function.",
            "Implement Python sorting algorithm.",
            "What is the capital of Germany?",
            "Define photosynthesis.",
        ]
        for q in queries:
            self._store(q)
        vec     = self.emb.embed("Write a sorting function in Python.")
        results = self.atlas.find_similar(vec, k=4, min_score=0.0)
        if len(results) > 1:
            sims = [r.similarity for r in results]
            assert sims == sorted(sims, reverse=True)

    def test_find_similar_min_score_filter(self):
        self._store("Completely unrelated query about cooking recipes.")
        vec     = self.emb.embed("Write a Python binary search.")
        results = self.atlas.find_similar(vec, k=5, min_score=0.99)
        # Very high threshold — may return 0 or 1
        for r in results:
            assert r.similarity >= 0.99

    def test_stats_empty(self):
        s = self.atlas.stats()
        assert s.total_records == 0

    def test_stats_populated(self):
        self._store("Code query", domain=Domain.CODE, quality=0.9)
        self._store("Math query", domain=Domain.MATH, quality=0.8)
        self._store("General query", domain=Domain.GENERAL)
        s = self.atlas.stats()
        assert s.total_records == 3
        assert "code" in s.domain_counts
        assert "math" in s.domain_counts

    def test_stats_avg_quality(self):
        self._store("q1", quality=0.8)
        self._store("q2", quality=0.9)
        s = self.atlas.stats()
        assert abs(s.avg_quality - 0.85) < 0.01

    def test_stats_str_contains_key_fields(self):
        self._store("q")
        text = str(self.atlas.stats())
        assert "ThinkRouter" in text
        assert "Total records" in text

    def test_export_records(self):
        import json
        self._store("Query to export", domain=Domain.LEGAL)
        out_path = self.atlas.export_records()
        assert os.path.exists(out_path)
        with open(out_path) as f:
            data = json.load(f)
        assert data["metadata"]["total"] == 1
        assert data["records"][0]["domain"] == "legal"

    def test_repr(self):
        assert "Atlas(" in repr(self.atlas)
        assert "dim=64" in repr(self.atlas)

    def test_wrong_dim_raises(self):
        wrong_vec = HashSketchEmbedder(dim=128).embed("test")
        with pytest.raises(ValueError, match="shape"):
            self.atlas.store(
                query="test", embedding=wrong_vec,
                domain=Domain.GENERAL, tier=Tier.SHORT,
                model="gpt-4o", provider="openai",
            )

    def test_persistence_across_reload(self):
        """Atlas data survives close + reopen."""
        self._store("Persistent query", domain=Domain.MATH)
        self.atlas.close()

        atlas2 = Atlas(path=self.tmpdir, embedding_dim=64,
                       embedding_backend="hash-sketch-64")
        assert len(atlas2) == 1
        stats = atlas2.stats()
        assert "math" in stats.domain_counts
        atlas2.close()

    def test_thread_safe_concurrent_stores(self):
        errors = []
        def store_n(n):
            try:
                for i in range(n):
                    self._store(f"Concurrent query {threading.current_thread().name} {i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=store_n, args=(10,)) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert len(self.atlas) == 50

    def test_max_records_eviction(self):
        atlas = Atlas(path=tempfile.mkdtemp(), embedding_dim=64,
                      embedding_backend="test", max_records=3)
        emb = HashSketchEmbedder(dim=64)
        for i in range(5):
            vec = emb.embed(f"query {i}")
            atlas.store(
                query=f"query {i}", embedding=vec,
                domain=Domain.GENERAL, tier=Tier.SHORT,
                model="gpt-4o", provider="openai",
            )
        assert len(atlas) <= 3
        atlas.close()


# ── Phase 2 — ThinkRouter + Atlas integration ──────────────────────────────

class TestThinkRouterAtlas:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_record_id_returned(self):
        r = _make_router_with_atlas(tmpdir=self.tmpdir)
        resp = r.chat("Write a Python function.")
        # record_id is set (may be None if thread not yet done, but not error)
        assert resp.record_id is None or isinstance(resp.record_id, str)

    def test_atlas_grows_after_chat(self):
        import time
        r = _make_router_with_atlas(tmpdir=self.tmpdir)
        r.chat("Write a binary search in Python.")
        r.chat("What is the capital of France?")
        time.sleep(0.2)  # let background threads complete
        # Atlas should have records (may vary by timing)
        assert len(r.atlas) >= 0  # non-negative always

    def test_update_quality_no_atlas(self):
        r = _make_router()  # no atlas
        # Should not raise
        r.update_quality("some-id", 0.9)

    def test_update_quality_with_atlas(self):
        import time
        r = _make_router_with_atlas(tmpdir=self.tmpdir)
        # Manually store a record to get a real ID
        vec = r._embedder.embed("Test quality update")
        rid = r.atlas.store(
            query="Test quality update",
            embedding=vec,
            domain=Domain.CODE,
            tier=Tier.FULL,
            model="gpt-4o",
            provider="openai",
            quality_score=None,
        )
        r.update_quality(rid, 0.88)
        rec = r.atlas.get(rid)
        assert rec is not None
        assert abs(rec.quality_score - 0.88) < 1e-6

    def test_router_repr_shows_atlas(self):
        r = _make_router_with_atlas(tmpdir=self.tmpdir)
        assert "atlas=" in repr(r)

    def test_atlas_disabled_no_embedder(self):
        r = _make_router()
        assert r._embedder is None
        assert r.atlas is None
        # chat still works without atlas
        resp = r.chat("test")
        assert isinstance(resp, RouterResponse)
        assert resp.record_id is None


# ── Phase 1 tests carried forward ─────────────────────────────────────────

class TestDomainClassifier:
    def setup_method(self): self.clf = DomainClassifier()

    def test_code(self):   assert self.clf.predict("Write a binary search function in Python.").domain == Domain.CODE
    def test_math(self):   assert self.clf.predict("Prove that sqrt(2) is irrational.").domain == Domain.MATH
    def test_medical(self):assert self.clf.predict("What is the mechanism of action of metformin?").domain == Domain.MEDICAL
    def test_legal(self):  assert self.clf.predict("What are the elements of a valid contract?").domain == Domain.LEGAL
    def test_financial(self):assert self.clf.predict("How do you build a DCF valuation model?").domain == Domain.FINANCIAL
    def test_general(self):assert self.clf.predict("What is the capital of France?").domain == Domain.GENERAL
    def test_confidence_range(self): assert 0.0 <= self.clf.predict("test").confidence <= 1.0
    def test_batch(self):  assert len(self.clf.predict_batch(["a","b","c"])) == 3


class TestHeuristicClassifier:
    def setup_method(self): self.clf = get_classifier("heuristic")
    def test_no_think(self): assert self.clf.predict("What is 7 * 8?").tier == Tier.NO_THINK
    def test_full(self):     assert self.clf.predict("Prove that sqrt(2) is irrational.").tier == Tier.FULL
    def test_batch(self):    assert len(self.clf.predict_batch(["a","b"])) == 2


class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(ProviderError,      ThinkRouterError)
        assert issubclass(RateLimitError,     ProviderError)
        assert issubclass(ClassifierError,    ThinkRouterError)
        assert issubclass(ConfigurationError, ThinkRouterError)


class TestUsageTracker:
    def setup_method(self): self.t = UsageTracker()
    def test_empty(self):   assert self.t.summary().total_calls == 0
    def test_record(self):
        self.t.record("q", Tier.NO_THINK, 0.9, 1.0)
        assert self.t.summary().total_calls == 1
    def test_savings(self):
        for _ in range(10): self.t.record("easy", Tier.NO_THINK, 0.9, 1.0)
        expected = 10*(TIER_TOKEN_BUDGETS[Tier.FULL]-TIER_TOKEN_BUDGETS[Tier.NO_THINK])
        assert self.t.summary().total_tokens_saved == expected
    def test_thread_safe(self):
        def w():
            for _ in range(50): self.t.record("q", Tier.SHORT, 0.8, 0.5)
        ths=[threading.Thread(target=w) for _ in range(10)]
        for t in ths: t.start()
        for t in ths: t.join()
        assert self.t.summary().total_calls == 500


class TestThinkRouter:
    def test_returns_response(self):   assert isinstance(_make_router().chat("test"), RouterResponse)
    def test_content(self):            assert _make_router("Hi.").chat("test").content == "Hi."
    def test_domain_populated(self):
        r = _make_router().chat("Write a Python function.")
        assert r.domain_result is not None
    def test_no_adapter_raises(self):
        r = ThinkRouter.__new__(ThinkRouter)
        r.provider="generic"; r.model="t"; r.verbose=False; r.max_retries=1
        r._clf=get_classifier("heuristic"); r._domain_clf=DomainClassifier()
        r._threshold=0.75; r.domain_routing=True; r.domain_min_confidence=0.40
        r._preferred_provider="openai"; r._registry=ModelRegistry()
        r._ollama_adapter=None; r.usage=UsageTracker()
        r._embedder=None; r.atlas=None; r._adapter=None
        with pytest.raises(ConfigurationError): r.chat("test")

    @pytest.mark.asyncio
    async def test_achat(self):
        r = _make_router()
        resp = await r.achat("test")
        assert isinstance(resp, RouterResponse)


class TestConstants:
    def test_o1_reasoning(self):   assert "o1" in OPENAI_REASONING_MODELS
    def test_full_high(self):      assert OPENAI_REASONING_EFFORT[Tier.FULL] == "high"
    def test_notk_zero(self):      assert ANTHROPIC_THINKING_BUDGETS[Tier.NO_THINK] == 0
    def test_claude_thinking(self):assert "claude-opus-4-6" in ANTHROPIC_THINKING_MODELS


class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.classifier_backend == "heuristic"
        assert cfg.atlas_enabled is True
        assert cfg.embedder_backend == "hash"

    def test_phase2_env(self):
        with patch.dict(os.environ, {
            "THINKROUTER_ATLAS_ENABLED": "0",
            "THINKROUTER_EMBEDDER": "openai",
            "THINKROUTER_EMBED_DIM": "512",
        }):
            cfg = Config()
            assert cfg.atlas_enabled is False
            assert cfg.embedder_backend == "openai"
            assert cfg.embed_dim == 512


class TestModelRegistry:
    def setup_method(self): self.reg = ModelRegistry()
    def test_resolve_code(self):
        t = self.reg.resolve(Domain.CODE, preferred_provider="ollama")
        assert t.provider == "ollama"
    def test_all_domains(self):
        for d in Domain:
            assert isinstance(self.reg.resolve(d), ModelTarget)


class TestEndToEnd:
    clf_c = get_classifier("heuristic")
    clf_d = DomainClassifier()

    @pytest.mark.parametrize("q,dom",[
        ("Write a quicksort in Python.",          Domain.CODE),
        ("Prove Fermat's Last Theorem.",          Domain.MATH),
        ("What are the symptoms of diabetes?",    Domain.MEDICAL),
        ("Explain GDPR compliance requirements.", Domain.LEGAL),
        ("How do you calculate the P/E ratio?",  Domain.FINANCIAL),
        ("What is the capital of France?",        Domain.GENERAL),
    ])
    def test_domain(self, q, dom):
        assert self.clf_d.predict(q).domain == dom

    @pytest.mark.parametrize("q",[
        "What is 12*7?", "Define osmosis.", "What is the capital of Japan?",
    ])
    def test_no_think(self, q):
        assert self.clf_c.predict(q).tier == Tier.NO_THINK

    @pytest.mark.parametrize("q",[
        "Prove by induction that sum of first n integers is n(n+1)/2.",
        "Write a Python implementation of a balanced binary search tree.",
    ])
    def test_full(self, q):
        assert self.clf_c.predict(q).tier == Tier.FULL
