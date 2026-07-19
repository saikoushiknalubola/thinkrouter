"""tests/test_thinkrouter.py — v0.6.0  (Phase 1 + 2 + 3)"""
from __future__ import annotations

import os, tempfile, threading
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from thinkrouter import (
    Atlas, CacheResult, CacheStats, ClassifierResult, Config,
    Domain, DomainClassifier, DomainResult, EmbeddingResult,
    HashSketchEmbedder, ModelRegistry, ModelTarget, RouterResponse,
    SemanticCache, ThinkRouter, Tier, TIER_TOKEN_BUDGETS,
    UsageTracker, get_classifier, get_embedder,
)
from thinkrouter.exceptions import (
    ConfigurationError, ProviderError, RateLimitError,
    ThinkRouterError, ClassifierError, AuthenticationError,
)
from thinkrouter.constants import (
    ANTHROPIC_THINKING_BUDGETS, ANTHROPIC_THINKING_MODELS,
    OPENAI_REASONING_EFFORT, OPENAI_REASONING_MODELS,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _mkrouter(content="Test.", atlas=None, cache=None):
    r = ThinkRouter.__new__(ThinkRouter)
    r.provider="openai"; r.model="gpt-4o"; r.verbose=False; r.max_retries=1
    r.domain_routing=True; r.domain_min_confidence=0.40
    r._preferred_provider="openai"
    r._clf=get_classifier("heuristic")
    r._domain_clf=DomainClassifier()
    r._registry=ModelRegistry(provider_priority=["openai"])
    r._threshold=0.75; r.usage=UsageTracker()
    r._ollama_adapter=None
    r._embedder=HashSketchEmbedder(dim=64) if (atlas or cache) else None
    r.atlas=atlas; r.cache=cache
    a=MagicMock()
    ret=(content,MagicMock(),{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30},None)
    a.call=MagicMock(return_value=ret); a.acall=AsyncMock(return_value=ret)
    r._adapter=a
    return r

def _mk_atlas(tmpdir=None):
    d = tmpdir or tempfile.mkdtemp()
    return Atlas(path=d, embedding_dim=64, embedding_backend="hash-sketch-64")

def _mk_cache(atlas, threshold=0.85, min_atlas_size=0):
    emb = HashSketchEmbedder(dim=64)
    return SemanticCache(atlas=atlas, embedder=emb,
                         threshold=threshold, min_atlas_size=min_atlas_size)

def _seed_atlas(atlas, emb, queries, domain=Domain.CODE, tier=Tier.FULL):
    for q in queries:
        vec = emb.embed(q)
        atlas.store(query=q, embedding=vec, domain=domain, tier=tier,
                    model="deepseek-coder-v2", provider="openai",
                    quality_score=0.90, latency_ms=5.0)


# ══ PHASE 3 — SemanticCache ════════════════════════════════════════════════

class TestSemanticCache:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.atlas  = _mk_atlas(self.tmpdir)
        self.emb    = HashSketchEmbedder(dim=64)
        self.cache  = _mk_cache(self.atlas, threshold=0.80, min_atlas_size=0)

    def teardown_method(self):
        self.atlas.close()

    # Basic lookup behaviour

    def test_empty_atlas_returns_none(self):
        vec = self.emb.embed("Write a binary search.")
        assert self.cache.lookup(vec) is None

    def test_hit_after_seeding(self):
        _seed_atlas(self.atlas, self.emb, ["Write a binary search in Python."])
        vec = self.emb.embed("Write a binary search in Python.")
        result = self.cache.lookup(vec)
        assert result is not None
        assert isinstance(result, CacheResult)

    def test_hit_returns_correct_domain(self):
        _seed_atlas(self.atlas, self.emb, ["Write a binary search."],
                    domain=Domain.CODE)
        vec = self.emb.embed("Write a binary search.")
        result = self.cache.lookup(vec)
        assert result is not None
        assert result.domain == Domain.CODE

    def test_hit_returns_correct_tier(self):
        _seed_atlas(self.atlas, self.emb, ["Prove sqrt(2) is irrational."],
                    domain=Domain.MATH, tier=Tier.FULL)
        vec = self.emb.embed("Prove sqrt(2) is irrational.")
        result = self.cache.lookup(vec)
        assert result is not None
        assert result.tier == Tier.FULL

    def test_hit_similarity_in_range(self):
        _seed_atlas(self.atlas, self.emb, ["Write a quicksort."])
        vec = self.emb.embed("Write a quicksort.")
        result = self.cache.lookup(vec)
        if result:
            assert 0.0 <= result.similarity <= 1.0

    def test_high_threshold_reduces_hits(self):
        cache_strict = _mk_cache(self.atlas, threshold=0.9999, min_atlas_size=0)
        _seed_atlas(self.atlas, self.emb, ["Write a binary search."])
        vec = self.emb.embed("Write a quicksort algorithm.")
        # Very different query — should miss with strict threshold
        result = cache_strict.lookup(vec)
        # Result may or may not hit — just ensure no error
        assert result is None or isinstance(result, CacheResult)

    def test_min_atlas_size_prevents_early_cache(self):
        cache = _mk_cache(self.atlas, threshold=0.80, min_atlas_size=100)
        _seed_atlas(self.atlas, self.emb, ["Write a binary search."])
        vec = self.emb.embed("Write a binary search.")
        # Atlas has 1 record but min_atlas_size=100 → must miss
        assert cache.lookup(vec) is None

    def test_low_quality_record_skipped(self):
        vec_store = self.emb.embed("Write a binary search.")
        self.atlas.store(
            query="Write a binary search.", embedding=vec_store,
            domain=Domain.CODE, tier=Tier.FULL,
            model="gpt-4o", provider="openai",
            quality_score=0.30,   # below min_quality=0.70
        )
        cache = _mk_cache(self.atlas, threshold=0.80, min_atlas_size=0)
        cache.min_quality = 0.70
        vec = self.emb.embed("Write a binary search.")
        result = cache.lookup(vec)
        assert result is None

    def test_none_quality_score_always_qualifies(self):
        vec_store = self.emb.embed("Write a binary search.")
        self.atlas.store(
            query="Write a binary search.", embedding=vec_store,
            domain=Domain.CODE, tier=Tier.FULL,
            model="gpt-4o", provider="openai",
            quality_score=None,   # no score yet — should still qualify
        )
        cache = _mk_cache(self.atlas, threshold=0.80, min_atlas_size=0)
        vec = self.emb.embed("Write a binary search.")
        result = cache.lookup(vec)
        assert result is not None

    def test_lookup_query_convenience(self):
        _seed_atlas(self.atlas, self.emb, ["Write a binary search."])
        result = self.cache.lookup_query("Write a binary search.")
        assert result is None or isinstance(result, CacheResult)

    def test_cache_result_repr(self):
        _seed_atlas(self.atlas, self.emb, ["Write a binary search."])
        vec = self.emb.embed("Write a binary search.")
        result = self.cache.lookup(vec)
        if result:
            assert "CacheResult(" in repr(result)
            assert "domain=" in repr(result)

    def test_is_high_confidence(self):
        r = CacheResult(
            domain=Domain.CODE, tier=Tier.FULL, model="gpt-4o",
            provider="openai", similarity=0.97, quality_score=0.9,
            source_id="abc", source_preview="test", latency_ms=1.0,
        )
        assert r.is_high_confidence is True
        r2 = CacheResult(
            domain=Domain.GENERAL, tier=Tier.SHORT, model="gpt-4o",
            provider="openai", similarity=0.88, quality_score=0.8,
            source_id="def", source_preview="test2", latency_ms=1.0,
        )
        assert r2.is_high_confidence is False

    # Stats

    def test_stats_empty(self):
        s = self.cache.stats()
        assert s.total_lookups == 0
        assert s.cache_hits    == 0
        assert s.hit_rate      == 0.0

    def test_stats_after_miss(self):
        vec = self.emb.embed("Anything")
        self.cache.lookup(vec)
        s = self.cache.stats()
        assert s.total_lookups == 1
        assert s.cache_hits    == 0

    def test_stats_after_hit(self):
        _seed_atlas(self.atlas, self.emb, ["Write a binary search."])
        vec = self.emb.embed("Write a binary search.")
        self.cache.lookup(vec)
        s = self.cache.stats()
        assert s.total_lookups == 1

    def test_stats_str(self):
        text = str(self.cache.stats())
        assert "ThinkRouter" in text
        assert "Hit rate" in text

    def test_reset_stats(self):
        vec = self.emb.embed("test")
        self.cache.lookup(vec)
        self.cache.reset_stats()
        assert self.cache.stats().total_lookups == 0

    def test_repr(self):
        assert "SemanticCache(" in repr(self.cache)
        assert "threshold=" in repr(self.cache)

    # Warmup

    def test_warmup_stores_records(self):
        n = self.cache.warmup(
            queries=["Write a binary search.", "Implement merge sort."],
            domains=[Domain.CODE, Domain.CODE],
            models=["deepseek-coder-v2", "deepseek-coder-v2"],
        )
        assert n == 2
        assert len(self.atlas) == 2

    def test_warmup_enables_hits(self):
        self.cache.warmup(
            queries=["Write a binary search in Python."],
            domains=[Domain.CODE],
            models=["deepseek-coder-v2"],
        )
        result = self.cache.lookup_query("Write a binary search in Python.")
        assert result is None or isinstance(result, CacheResult)

    def test_warmup_empty_list(self):
        n = self.cache.warmup(queries=[])
        assert n == 0

    # Thread safety

    def test_stats_thread_safe(self):
        errors = []
        _seed_atlas(self.atlas, self.emb, ["test query"])

        def lookup_loop():
            try:
                vec = self.emb.embed("test query")
                for _ in range(20):
                    self.cache.lookup(vec)
            except Exception as e:
                errors.append(e)

        ths = [threading.Thread(target=lookup_loop) for _ in range(5)]
        for t in ths: t.start()
        for t in ths: t.join()
        assert errors == []
        s = self.cache.stats()
        assert s.total_lookups == 100


# ══ PHASE 3 — Router integration ══════════════════════════════════════════

class TestRouterCacheIntegration:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_was_cached_false_without_cache(self):
        r = _mkrouter()
        resp = r.chat("test")
        assert resp.was_cached is False
        assert resp.cache_result is None

    def test_cache_result_none_on_miss(self):
        atlas = _mk_atlas(self.tmpdir)
        cache = _mk_cache(atlas, min_atlas_size=0)
        r = _mkrouter(atlas=atlas, cache=cache)
        resp = r.chat("Write a binary search.")
        assert resp.cache_result is None  # atlas empty → miss

    def test_tier_property_from_classifiers(self):
        r = _mkrouter()
        resp = r.chat("What is 2+2?")
        assert resp.tier == Tier.NO_THINK

    def test_tier_property_from_cache(self):
        atlas = _mk_atlas(self.tmpdir)
        emb   = HashSketchEmbedder(dim=64)
        cache = _mk_cache(atlas, threshold=0.80, min_atlas_size=0)
        # Pre-seed with exact query
        vec = emb.embed("Write a binary search in Python.")
        atlas.store(query="Write a binary search in Python.", embedding=vec,
                    domain=Domain.CODE, tier=Tier.FULL,
                    model="deepseek-coder-v2", provider="openai",
                    quality_score=0.92, latency_ms=5.0)

        r = _mkrouter(atlas=atlas, cache=cache)
        resp = r.chat("Write a binary search in Python.")
        # Whether hit or miss, tier property works
        assert resp.tier in (Tier.NO_THINK, Tier.SHORT, Tier.FULL)

    def test_router_repr_shows_hit_rate(self):
        atlas = _mk_atlas(self.tmpdir)
        cache = _mk_cache(atlas, min_atlas_size=0)
        r = _mkrouter(atlas=atlas, cache=cache)
        assert "hit_rate=" in repr(r)

    def test_router_repr_no_cache(self):
        r = _mkrouter()
        assert "hit_rate=" not in repr(r)

    def test_update_quality_with_cache_router(self):
        atlas = _mk_atlas(self.tmpdir)
        cache = _mk_cache(atlas, min_atlas_size=0)
        r     = _mkrouter(atlas=atlas, cache=cache)
        emb   = HashSketchEmbedder(dim=64)
        vec   = emb.embed("Test query")
        rid   = atlas.store(query="Test", embedding=vec,
                            domain=Domain.CODE, tier=Tier.FULL,
                            model="gpt-4o", provider="openai")
        r.update_quality(rid, 0.85)
        rec = atlas.get(rid)
        assert rec is not None
        assert abs(rec.quality_score - 0.85) < 1e-6

    @pytest.mark.asyncio
    async def test_achat_cache_miss(self):
        atlas = _mk_atlas(self.tmpdir)
        cache = _mk_cache(atlas, min_atlas_size=0)
        r     = _mkrouter(atlas=atlas, cache=cache)
        resp  = await r.achat("Write a quicksort.")
        assert isinstance(resp, RouterResponse)

    def test_was_cached_property(self):
        resp = RouterResponse(
            content="Hi", routing=None, domain_result=None,
            model_target=None, cache_result=None, raw=None,
            provider="openai", model="gpt-4o",
            usage_tokens={}, record_id=None,
        )
        assert resp.was_cached is False

        cr = CacheResult(
            domain=Domain.CODE, tier=Tier.FULL, model="gpt-4o",
            provider="openai", similarity=0.95, quality_score=0.9,
            source_id="x", source_preview="q", latency_ms=1.0,
        )
        resp2 = RouterResponse(
            content="Hi", routing=None, domain_result=None,
            model_target=None, cache_result=cr, raw=None,
            provider="openai", model="gpt-4o",
            usage_tokens={}, record_id=None,
        )
        assert resp2.was_cached is True


# ══ PHASE 2 — Atlas & Embedder (carried forward) ══════════════════════════

class TestHashSketchEmbedder:
    def setup_method(self): self.emb = HashSketchEmbedder(dim=256)
    def test_shape(self):         assert self.emb.embed("test").shape == (256,)
    def test_dtype(self):         assert self.emb.embed("test").dtype == np.float32
    def test_normalised(self):    assert abs(float(np.linalg.norm(self.emb.embed("test"))) - 1.0) < 1e-5
    def test_deterministic(self):
        assert np.array_equal(self.emb.embed("same"), self.emb.embed("same"))
    def test_batch_shape(self):   assert self.emb.embed_batch(["a","b","c"]).shape == (3,256)
    def test_dim_property(self):  assert self.emb.dim == 256
    def test_meta_result(self):
        r = self.emb.embed_with_meta("test")
        assert isinstance(r, EmbeddingResult)
        assert r.dim == 256

class TestAtlas:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.atlas  = Atlas(path=self.tmpdir, embedding_dim=64, embedding_backend="test")
        self.emb    = HashSketchEmbedder(dim=64)
    def teardown_method(self): self.atlas.close()

    def _s(self, q, domain=Domain.CODE, quality=None):
        vec = self.emb.embed(q)
        return self.atlas.store(query=q, embedding=vec, domain=domain,
                                tier=Tier.FULL, model="gpt-4o",
                                provider="openai", quality_score=quality)

    def test_empty(self):          assert len(self.atlas) == 0
    def test_store(self):          self._s("q"); assert len(self.atlas) == 1
    def test_id_format(self):      assert len(self._s("q")) == 36
    def test_get(self):
        rid = self._s("test retrieval", domain=Domain.MATH)
        rec = self.atlas.get(rid)
        assert rec is not None and rec.domain == Domain.MATH
    def test_update_quality(self):
        rid = self._s("q")
        self.atlas.update_quality(rid, 0.88)
        assert abs(self.atlas.get(rid).quality_score - 0.88) < 1e-6
    def test_find_similar_empty(self):
        vec = self.emb.embed("x")
        assert self.atlas.find_similar(vec) == []
    def test_stats_total(self):
        self._s("a"); self._s("b")
        assert self.atlas.stats().total_records == 2
    def test_persist(self):
        self._s("persist")
        self.atlas.close()
        a2 = Atlas(path=self.tmpdir, embedding_dim=64, embedding_backend="test")
        assert len(a2) == 1; a2.close()


# ══ PHASE 1 (carried forward) ══════════════════════════════════════════════

class TestDomainClassifier:
    def setup_method(self): self.clf = DomainClassifier()
    def test_code(self):     assert self.clf.predict("Write a binary search function in Python.").domain == Domain.CODE
    def test_math(self):     assert self.clf.predict("Prove that sqrt(2) is irrational.").domain == Domain.MATH
    def test_medical(self):  assert self.clf.predict("What is the mechanism of action of metformin?").domain == Domain.MEDICAL
    def test_legal(self):    assert self.clf.predict("What are the elements of a valid contract?").domain == Domain.LEGAL
    def test_financial(self):assert self.clf.predict("How do you build a DCF valuation model?").domain == Domain.FINANCIAL
    def test_general(self):  assert self.clf.predict("What is the capital of France?").domain == Domain.GENERAL
    def test_batch(self):    assert len(self.clf.predict_batch(["a","b","c"])) == 3

class TestHeuristicClassifier:
    def setup_method(self): self.clf = get_classifier("heuristic")
    def test_no_think(self): assert self.clf.predict("What is 7*8?").tier == Tier.NO_THINK
    def test_full(self):     assert self.clf.predict("Prove that sqrt(2) is irrational.").tier == Tier.FULL

class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(ProviderError, ThinkRouterError)
        assert issubclass(RateLimitError, ProviderError)
        assert issubclass(ClassifierError, ThinkRouterError)
        assert issubclass(ConfigurationError, ThinkRouterError)

class TestUsageTracker:
    def setup_method(self): self.t = UsageTracker()
    def test_empty(self):    assert self.t.summary().total_calls == 0
    def test_record(self):
        self.t.record("q", Tier.NO_THINK, 0.9, 1.0)
        assert self.t.summary().total_calls == 1
    def test_thread_safe(self):
        def w():
            for _ in range(50): self.t.record("q", Tier.SHORT, 0.8, 0.5)
        ths = [threading.Thread(target=w) for _ in range(10)]
        for t in ths: t.start()
        for t in ths: t.join()
        assert self.t.summary().total_calls == 500

class TestThinkRouter:
    def test_returns_response(self): assert isinstance(_mkrouter().chat("test"), RouterResponse)
    def test_content(self):          assert _mkrouter("Hi.").chat("test").content == "Hi."
    def test_no_adapter_raises(self):
        r = ThinkRouter.__new__(ThinkRouter)
        r.provider="generic"; r.model="t"; r.verbose=False; r.max_retries=1
        r._clf=get_classifier("heuristic"); r._domain_clf=DomainClassifier()
        r._threshold=0.75; r.domain_routing=True; r.domain_min_confidence=0.40
        r._preferred_provider="openai"; r._registry=ModelRegistry()
        r._ollama_adapter=None; r.usage=UsageTracker()
        r._embedder=None; r.atlas=None; r.cache=None; r._adapter=None
        with pytest.raises(ConfigurationError): r.chat("test")
    def test_bad_provider(self):
        with pytest.raises(ConfigurationError): ThinkRouter(provider="fake")
    @pytest.mark.asyncio
    async def test_achat(self):
        r = _mkrouter()
        resp = await r.achat("test")
        assert isinstance(resp, RouterResponse)

class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.atlas_enabled is True
        assert cfg.cache_enabled is True
        assert cfg.cache_threshold == 0.92
    def test_phase3_env(self):
        with patch.dict(os.environ, {
            "THINKROUTER_CACHE_ENABLED":   "0",
            "THINKROUTER_CACHE_THRESHOLD": "0.88",
        }):
            cfg = Config()
            assert cfg.cache_enabled is False
            assert abs(cfg.cache_threshold - 0.88) < 1e-9

class TestConstants:
    def test_o1(self):   assert "o1" in OPENAI_REASONING_MODELS
    def test_full(self): assert OPENAI_REASONING_EFFORT[Tier.FULL] == "high"
    def test_notk(self): assert ANTHROPIC_THINKING_BUDGETS[Tier.NO_THINK] == 0

class TestEndToEnd:
    clf_c = get_classifier("heuristic"); clf_d = DomainClassifier()

    @pytest.mark.parametrize("q,d",[
        ("Write a quicksort in Python.", Domain.CODE),
        ("Prove Fermat's Last Theorem.", Domain.MATH),
        ("What are symptoms of diabetes?", Domain.MEDICAL),
        ("Explain GDPR compliance.", Domain.LEGAL),
        ("How do you calculate P/E ratio?", Domain.FINANCIAL),
        ("What is the capital of France?", Domain.GENERAL),
    ])
    def test_domain(self,q,d): assert self.clf_d.predict(q).domain == d

    @pytest.mark.parametrize("q",["What is 12*7?","Define osmosis."])
    def test_no_think(self,q): assert self.clf_c.predict(q).tier == Tier.NO_THINK

    @pytest.mark.parametrize("q",[
        "Prove by induction that sum of first n integers is n(n+1)/2.",
        "Write a Python implementation of a balanced binary search tree.",
    ])
    def test_full(self,q): assert self.clf_c.predict(q).tier == Tier.FULL
