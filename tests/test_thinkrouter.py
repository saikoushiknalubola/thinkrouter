"""tests/test_thinkrouter.py — v0.7.0  Phase 1+2+3+4"""
from __future__ import annotations

import os, tempfile, threading
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from thinkrouter import (
    Atlas, CacheResult, ClassifierResult, Config,
    ConfidenceResult, CostRecord, CostTracker, Domain,
    DomainClassifier, DomainResult, EmbeddingResult,
    FallbackChain, FallbackResult, HashSketchEmbedder,
    HeuristicConfidenceModel, ModelRegistry, ModelTarget,
    Recommendation, RouterResponse, SemanticCache,
    ThinkRouter, Tier, TIER_TOKEN_BUDGETS,
    UsageTracker, get_classifier, get_confidence_model, get_cost_usd,
)
from thinkrouter.exceptions import (
    ConfigurationError, ProviderError, RateLimitError,
    ThinkRouterError, ClassifierError,
)
from thinkrouter.constants import (
    OPENAI_REASONING_EFFORT, OPENAI_REASONING_MODELS,
    ANTHROPIC_THINKING_BUDGETS, ANTHROPIC_THINKING_MODELS,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _mkrouter(content="Test.", atlas=None, cache=None, with_confidence=True):
    r = ThinkRouter.__new__(ThinkRouter)
    r.provider="openai"; r.model="gpt-4o"; r.verbose=False; r.max_retries=1
    r.domain_routing=True; r.domain_min_confidence=0.40
    r._preferred_provider="openai"; r._escalation_model=None
    r._clf=get_classifier("heuristic"); r._domain_clf=DomainClassifier()
    r._registry=ModelRegistry(provider_priority=["openai"])
    r._threshold=0.75; r.usage=UsageTracker()
    r._ollama_adapter=None; r._fallback=None
    r._embedder=HashSketchEmbedder(dim=64) if (atlas or cache) else None
    r.atlas=atlas; r.cache=cache
    r.confidence_model=HeuristicConfidenceModel() if with_confidence else None
    r.cost_tracker=CostTracker()
    a=MagicMock()
    ret=("Test.",MagicMock(),{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150},None)
    a.call=MagicMock(return_value=ret); a.acall=AsyncMock(return_value=ret)
    r._adapter=a
    return r


# ══ PHASE 4 — Confidence Model ════════════════════════════════════════════

class TestHeuristicConfidenceModel:

    def setup_method(self):
        self.clf = HeuristicConfidenceModel()

    # Risk score range
    def test_risk_in_range(self):
        r = self.clf.predict("What happened in the news today?", "gpt-4o")
        assert 0.0 <= r.risk_score <= 1.0

    def test_result_type(self):
        r = self.clf.predict("What is photosynthesis?")
        assert isinstance(r, ConfidenceResult)

    # High-risk signals
    def test_recent_event_high_risk(self):
        r = self.clf.predict("What happened in the news today?", "gpt-4o")
        assert r.risk_score >= 0.55

    def test_citation_request_high_risk(self):
        r = self.clf.predict("Cite the peer-reviewed study on this topic.", "gpt-4o")
        assert r.risk_score >= 0.50

    def test_specific_price_high_risk(self):
        r = self.clf.predict("What is the current price of Tesla stock?", "gpt-4o")
        assert r.risk_score >= 0.45

    def test_medical_specific_high_risk(self):
        r = self.clf.predict(
            "What is the FDA-approved dosage of metformin and its drug interactions?",
            "gpt-4o"
        )
        assert r.risk_score >= 0.60

    # Low-risk signals
    def test_math_proof_low_risk(self):
        r = self.clf.predict("Prove that sqrt(2) is irrational.", "gpt-4o")
        assert r.risk_score < 0.40

    def test_code_generation_low_risk(self):
        r = self.clf.predict("Write a binary search function in Python.", "gpt-4o")
        assert r.risk_score < 0.40

    def test_historical_low_risk(self):
        r = self.clf.predict(
            "When was the Eiffel Tower built in the 1880s?", "gpt-4o"
        )
        assert r.risk_score < 0.45

    # Recommendation mapping
    def test_recommendation_is_enum(self):
        r = self.clf.predict("test", "gpt-4o")
        assert isinstance(r.recommendation, Recommendation)

    def test_high_risk_recommend_rag_or_escalate(self):
        r = self.clf.predict(
            "What happened in this week's G7 summit and cite your sources?",
            "gpt-4o"
        )
        assert r.recommendation in (
            Recommendation.RAG, Recommendation.ESCALATE, Recommendation.ABSTAIN
        )

    def test_safe_query_proceed(self):
        r = self.clf.predict("Prove by induction that n^2 > 2n for n > 2.", "gpt-4o")
        assert r.recommendation in (Recommendation.PROCEED, Recommendation.VERIFY)

    # Properties
    def test_is_high_risk_property(self):
        r = self.clf.predict("What happened in the news today?", "gpt-4o")
        assert r.is_high_risk == (r.risk_score >= 0.65)

    def test_is_safe_property(self):
        r = self.clf.predict("Prove Pythagoras theorem algebraically.", "gpt-4o")
        assert r.is_safe == (r.risk_score < 0.35)

    def test_signals_tuple(self):
        r = self.clf.predict("What happened today?", "gpt-4o")
        assert isinstance(r.signals, tuple)

    def test_latency_fast(self):
        r = self.clf.predict("test query", "gpt-4o")
        assert r.latency_ms < 100

    def test_backend_label(self):
        r = self.clf.predict("test", "gpt-4o")
        assert r.backend == "heuristic"

    def test_repr(self):
        r = self.clf.predict("test", "gpt-4o")
        assert "ConfidenceResult(" in repr(r)

    # Model modifier
    def test_smaller_model_higher_risk(self):
        r_large = self.clf.predict("What is the CEO of Apple?", "gpt-4o")
        r_small = self.clf.predict("What is the CEO of Apple?", "gpt-4o-mini")
        assert r_small.risk_score >= r_large.risk_score

    # Batch
    def test_batch(self):
        results = self.clf.predict_batch(
            ["test1", "test2", "test3"], model="gpt-4o"
        )
        assert len(results) == 3
        assert all(isinstance(r, ConfidenceResult) for r in results)

    # Factory
    def test_get_confidence_model_heuristic(self):
        m = get_confidence_model("heuristic")
        assert isinstance(m, HeuristicConfidenceModel)

    def test_get_confidence_model_bad(self):
        with pytest.raises(ValueError):
            get_confidence_model("nonexistent")


# ══ PHASE 4 — Cost Tracker ════════════════════════════════════════════════

class TestCostTracker:

    def setup_method(self):
        self.tracker = CostTracker()

    def test_empty_summary(self):
        s = self.tracker.summary()
        assert s.total_calls == 0
        assert s.total_cost_usd == 0.0
        assert s.savings_pct == 0.0

    def test_record_returns_cost_record(self):
        rec = self.tracker.record(
            model="gpt-4o", provider="openai",
            domain=Domain.CODE, tier=Tier.FULL,
            input_tokens=1000, output_tokens=500,
        )
        assert isinstance(rec, CostRecord)
        assert rec.cost_usd >= 0.0

    def test_ollama_is_free(self):
        rec = self.tracker.record(
            model="deepseek-coder-v2", provider="ollama",
            domain=Domain.CODE, tier=Tier.FULL,
            input_tokens=1000, output_tokens=500,
        )
        assert rec.cost_usd == 0.0
        assert rec.baseline_usd > 0.0
        assert rec.saved_usd > 0.0

    def test_savings_accumulate(self):
        for _ in range(5):
            self.tracker.record(
                model="deepseek-coder-v2", provider="ollama",
                domain=Domain.CODE, tier=Tier.FULL,
                input_tokens=500, output_tokens=200,
            )
        s = self.tracker.summary()
        assert s.total_calls == 5
        assert s.saved_usd > 0.0
        assert s.savings_pct > 0.0
        assert s.free_calls == 5

    def test_gpt4o_costs_money(self):
        rec = self.tracker.record(
            model="gpt-4o", provider="openai",
            domain=Domain.GENERAL, tier=Tier.FULL,
            input_tokens=1000, output_tokens=500,
        )
        assert rec.cost_usd > 0.0

    def test_cost_by_domain(self):
        self.tracker.record("gpt-4o","openai",Domain.CODE,Tier.FULL,100,50)
        self.tracker.record("gpt-4o","openai",Domain.MATH,Tier.FULL,100,50)
        s = self.tracker.summary()
        assert "code" in s.cost_by_domain
        assert "math" in s.cost_by_domain

    def test_cost_by_model(self):
        self.tracker.record("gpt-4o","openai",Domain.GENERAL,Tier.SHORT,100,50)
        s = self.tracker.summary()
        assert "gpt-4o" in s.cost_by_model

    def test_reset(self):
        self.tracker.record("gpt-4o","openai",Domain.GENERAL,Tier.FULL,500,200)
        self.tracker.reset()
        assert self.tracker.summary().total_calls == 0

    def test_thread_safe(self):
        errors = []
        def w():
            try:
                for _ in range(20):
                    self.tracker.record(
                        "gpt-4o","openai",Domain.CODE,Tier.FULL,100,50
                    )
            except Exception as e:
                errors.append(e)
        ths = [threading.Thread(target=w) for _ in range(5)]
        for t in ths: t.start()
        for t in ths: t.join()
        assert errors == []
        assert self.tracker.summary().total_calls == 100

    def test_summary_str(self):
        self.tracker.record("gpt-4o","openai",Domain.CODE,Tier.FULL,500,200)
        text = str(self.tracker.summary())
        assert "Cost Dashboard" in text
        assert "Actual spend" in text

    def test_daily_projection(self):
        self.tracker.record("gpt-4o","openai",Domain.GENERAL,Tier.FULL,1000,500)
        proj = self.tracker.summary().daily_projection(10_000)
        assert "month" in proj

    def test_get_cost_usd_gpt4o(self):
        cost = get_cost_usd("gpt-4o", 1_000_000, 0)
        assert abs(cost - 2.50) < 0.01

    def test_get_cost_usd_ollama(self):
        cost = get_cost_usd("deepseek-coder-v2", 1_000_000, 1_000_000)
        assert cost == 0.0

    def test_recent(self):
        for i in range(5):
            self.tracker.record("gpt-4o","openai",Domain.GENERAL,Tier.SHORT,100,50)
        assert len(self.tracker.recent(3)) == 3


# ══ PHASE 4 — FallbackChain ═══════════════════════════════════════════════

class TestFallbackChain:

    def _mock_adapter(self, content="OK", fail_with=None):
        a = MagicMock()
        if fail_with:
            a.call  = MagicMock(side_effect=fail_with)
            a.acall = AsyncMock(side_effect=fail_with)
        else:
            ret = (content, MagicMock(), {"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}, None)
            a.call  = MagicMock(return_value=ret)
            a.acall = AsyncMock(return_value=ret)
        return a

    def test_primary_success(self):
        a = self._mock_adapter("primary response")
        chain = FallbackChain([("openai",a)], ["gpt-4o"])
        content, raw, usage, xp, fb = chain.call(
            messages=[{"role":"user","content":"test"}],
            tier=Tier.SHORT,
        )
        assert content == "primary response"
        assert fb.fallback_used is False
        assert fb.succeeded == "openai"
        assert fb.attempts == 1

    def test_fallback_on_rate_limit(self):
        a1 = self._mock_adapter(fail_with=RateLimitError("429",429,"openai"))
        a2 = self._mock_adapter("fallback response")
        chain = FallbackChain(
            [("openai",a1),("anthropic",a2)],
            ["gpt-4o","claude-sonnet-4-6"],
            retry_delay=0.0,
        )
        content, _, _, _, fb = chain.call(
            messages=[{"role":"user","content":"test"}], tier=Tier.FULL,
        )
        assert content == "fallback response"
        assert fb.fallback_used is True
        assert fb.succeeded == "anthropic"
        assert fb.attempts == 2
        assert len(fb.errors) == 1

    def test_all_fail_raises(self):
        err = ProviderError("fail",500,"openai")
        a1  = self._mock_adapter(fail_with=err)
        a2  = self._mock_adapter(fail_with=err)
        chain = FallbackChain(
            [("openai",a1),("anthropic",a2)],
            ["gpt-4o","claude-sonnet-4-6"],
            retry_delay=0.0,
        )
        with pytest.raises(ProviderError):
            chain.call(messages=[{"role":"user","content":"test"}], tier=Tier.SHORT)

    def test_permanent_error_does_not_retry(self):
        a1 = self._mock_adapter(fail_with=ProviderError("auth",401,"openai"))
        a2 = self._mock_adapter("should not reach")
        chain = FallbackChain(
            [("openai",a1),("anthropic",a2)],
            ["gpt-4o","claude-sonnet-4-6"],
            retry_delay=0.0,
        )
        with pytest.raises(ProviderError):
            chain.call(messages=[{"role":"user","content":"test"}], tier=Tier.SHORT)
        a2.call.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_primary_success(self):
        a = self._mock_adapter("async response")
        chain = FallbackChain([("openai",a)], ["gpt-4o"])
        content, _, _, _, fb = await chain.acall(
            messages=[{"role":"user","content":"test"}], tier=Tier.SHORT,
        )
        assert content == "async response"
        assert fb.fallback_used is False

    @pytest.mark.asyncio
    async def test_async_fallback(self):
        a1 = self._mock_adapter(fail_with=RateLimitError("429",429,"openai"))
        a2 = self._mock_adapter("async fallback")
        chain = FallbackChain(
            [("openai",a1),("anthropic",a2)],
            ["gpt-4o","claude-sonnet-4-6"],
            retry_delay=0.0,
        )
        content, _, _, _, fb = await chain.acall(
            messages=[{"role":"user","content":"test"}], tier=Tier.FULL,
        )
        assert content == "async fallback"
        assert fb.fallback_used is True

    def test_empty_adapters_raises(self):
        with pytest.raises(ValueError):
            FallbackChain([], [])

    def test_mismatched_lengths_raises(self):
        a = self._mock_adapter()
        with pytest.raises(ValueError):
            FallbackChain([("openai",a)], ["gpt-4o","extra"])

    def test_repr(self):
        a = self._mock_adapter()
        chain = FallbackChain([("openai",a)], ["gpt-4o"])
        assert "FallbackChain(" in repr(chain)
        assert "openai:gpt-4o" in repr(chain)

    def test_fallback_result_fields(self):
        a = self._mock_adapter("ok")
        chain = FallbackChain([("openai",a)], ["gpt-4o"])
        _, _, _, _, fb = chain.call(
            messages=[{"role":"user","content":"test"}], tier=Tier.SHORT
        )
        assert isinstance(fb, FallbackResult)
        assert isinstance(fb.attempted, list)
        assert isinstance(fb.succeeded, str)
        assert isinstance(fb.total_latency_ms, float)


# ══ PHASE 4 — RouterResponse ══════════════════════════════════════════════

class TestRouterResponseV7:

    def _make_resp(self, cache=None, conf=None, cost=None, fb=None):
        return RouterResponse(
            content="hi", routing=None, domain_result=None,
            model_target=None, cache_result=cache,
            confidence_result=conf, cost_record=cost,
            fallback_result=fb, raw=None,
            provider="openai", model="gpt-4o",
            usage_tokens={"prompt_tokens":100,"completion_tokens":50,"total_tokens":150},
        )

    def test_was_cached_false(self):
        assert self._make_resp().was_cached is False

    def test_was_cached_true(self):
        cr = CacheResult(Domain.CODE,Tier.FULL,"gpt-4o","openai",0.95,0.9,"x","q",1.0)
        assert self._make_resp(cache=cr).was_cached is True

    def test_fallback_used_false(self):
        assert self._make_resp().fallback_used is False

    def test_is_high_risk_none(self):
        assert self._make_resp().is_high_risk is False

    def test_cost_usd_none(self):
        assert self._make_resp().cost_usd == 0.0

    def test_cost_usd_with_record(self):
        from thinkrouter.cost import CostRecord as CR
        from datetime import datetime, timezone
        rec = CR("gpt-4o","openai",Domain.CODE,Tier.FULL,100,50,0.001234,0.003,0.001766)
        resp = self._make_resp(cost=rec)
        assert resp.cost_usd == 0.001234


# ══ PHASE 4 — ThinkRouter integration ════════════════════════════════════

class TestThinkRouterV7:

    def test_confidence_result_in_response(self):
        r = _mkrouter(with_confidence=True)
        resp = r.chat("What happened in the news today?")
        assert resp.confidence_result is not None
        assert isinstance(resp.confidence_result, ConfidenceResult)

    def test_confidence_none_when_disabled(self):
        r = _mkrouter(with_confidence=False)
        resp = r.chat("test")
        assert resp.confidence_result is None

    def test_cost_tracked(self):
        r = _mkrouter()
        r.chat("Write a binary search in Python.")
        s = r.cost_tracker.summary()
        assert s.total_calls == 1

    def test_cost_record_in_response(self):
        r = _mkrouter()
        resp = r.chat("Write a binary search in Python.")
        assert resp.cost_record is not None
        assert resp.cost_record.cost_usd >= 0.0

    def test_assess_confidence_standalone(self):
        r = _mkrouter()
        result = r.assess_confidence("What happened today in the news?")
        assert isinstance(result, ConfidenceResult)
        assert result.risk_score >= 0.50

    def test_assess_confidence_disabled(self):
        r = _mkrouter(with_confidence=False)
        assert r.assess_confidence("test") is None

    def test_fallback_none_by_default(self):
        r = _mkrouter()
        assert r._fallback is None

    def test_repr_shows_saved(self):
        r = _mkrouter()
        r.chat("test")
        assert "saved=$" in repr(r)

    @pytest.mark.asyncio
    async def test_achat_confidence(self):
        r    = _mkrouter(with_confidence=True)
        resp = await r.achat("Cite the peer-reviewed study on this.")
        assert resp.confidence_result is not None

    @pytest.mark.asyncio
    async def test_achat_cost(self):
        r    = _mkrouter()
        resp = await r.achat("test")
        assert resp.cost_record is not None


# ══ CARRIED FORWARD — Phases 1,2,3 ═══════════════════════════════════════

class TestAtlas:
    def setup_method(self):
        self.tmpdir=tempfile.mkdtemp()
        self.atlas=Atlas(path=self.tmpdir,embedding_dim=64,embedding_backend="test")
        self.emb=HashSketchEmbedder(dim=64)
    def teardown_method(self): self.atlas.close()
    def _s(self,q,d=Domain.CODE,q_score=None):
        v=self.emb.embed(q)
        return self.atlas.store(q,v,d,Tier.FULL,"gpt-4o","openai",q_score,5.0)
    def test_empty(self):     assert len(self.atlas)==0
    def test_store(self):     self._s("q"); assert len(self.atlas)==1
    def test_get(self):
        rid=self._s("test",Domain.MATH); rec=self.atlas.get(rid)
        assert rec is not None and rec.domain==Domain.MATH
    def test_update_quality(self):
        rid=self._s("q"); self.atlas.update_quality(rid,0.88)
        assert abs(self.atlas.get(rid).quality_score-0.88)<1e-6
    def test_thread_safe(self):
        errors=[]
        def w():
            try:
                for i in range(10): self._s(f"q{threading.current_thread().name}{i}")
            except Exception as e: errors.append(e)
        ths=[threading.Thread(target=w) for _ in range(5)]
        for t in ths: t.start()
        for t in ths: t.join()
        assert errors==[]
        assert len(self.atlas)==50

class TestSemanticCache:
    def setup_method(self):
        self.tmpdir=tempfile.mkdtemp()
        self.atlas=Atlas(path=self.tmpdir,embedding_dim=64,embedding_backend="test")
        self.emb=HashSketchEmbedder(dim=64)
        self.cache=SemanticCache(self.atlas,self.emb,threshold=0.80,min_atlas_size=0)
    def teardown_method(self): self.atlas.close()
    def _seed(self,q,d=Domain.CODE):
        v=self.emb.embed(q)
        self.atlas.store(q,v,d,Tier.FULL,"deepseek-coder-v2","openai",0.90,5.0)
    def test_miss_empty(self):
        assert self.cache.lookup(self.emb.embed("test")) is None
    def test_hit_after_seed(self):
        self._seed("Write a binary search.")
        r=self.cache.lookup(self.emb.embed("Write a binary search."))
        assert r is None or isinstance(r,CacheResult)
    def test_stats(self):
        s=self.cache.stats(); assert s.total_lookups==0
    def test_warmup(self):
        n=self.cache.warmup(["Write a function."],domains=[Domain.CODE])
        assert n==1

class TestDomainClassifier:
    def setup_method(self): self.clf=DomainClassifier()
    def test_code(self):     assert self.clf.predict("Write a binary search in Python.").domain==Domain.CODE
    def test_math(self):     assert self.clf.predict("Prove sqrt(2) is irrational.").domain==Domain.MATH
    def test_medical(self):  assert self.clf.predict("What is the mechanism of action of metformin?").domain==Domain.MEDICAL
    def test_legal(self):    assert self.clf.predict("What are the elements of a valid contract?").domain==Domain.LEGAL
    def test_financial(self):assert self.clf.predict("How do you build a DCF valuation model?").domain==Domain.FINANCIAL
    def test_general(self):  assert self.clf.predict("What is the capital of France?").domain==Domain.GENERAL

class TestHeuristicClassifier:
    def setup_method(self): self.clf=get_classifier("heuristic")
    def test_no_think(self): assert self.clf.predict("What is 7*8?").tier==Tier.NO_THINK
    def test_full(self):     assert self.clf.predict("Prove sqrt(2) is irrational.").tier==Tier.FULL

class TestUsageTracker:
    def setup_method(self): self.t=UsageTracker()
    def test_empty(self):    assert self.t.summary().total_calls==0
    def test_record(self):
        self.t.record("q",Tier.NO_THINK,0.9,1.0); assert self.t.summary().total_calls==1
    def test_thread_safe(self):
        def w():
            for _ in range(50): self.t.record("q",Tier.SHORT,0.8,0.5)
        ths=[threading.Thread(target=w) for _ in range(10)]
        for t in ths: t.start()
        for t in ths: t.join()
        assert self.t.summary().total_calls==500

class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(ProviderError,ThinkRouterError)
        assert issubclass(RateLimitError,ProviderError)
        assert issubclass(ClassifierError,ThinkRouterError)
        assert issubclass(ConfigurationError,ThinkRouterError)

class TestConstants:
    def test_o1(self):   assert "o1" in OPENAI_REASONING_MODELS
    def test_full(self): assert OPENAI_REASONING_EFFORT[Tier.FULL]=="high"
    def test_notk(self): assert ANTHROPIC_THINKING_BUDGETS[Tier.NO_THINK]==0

class TestThinkRouter:
    def test_returns_response(self): assert isinstance(_mkrouter().chat("test"),RouterResponse)
    def test_no_adapter_raises(self):
        r=ThinkRouter.__new__(ThinkRouter)
        r.provider="generic"; r.model="t"; r.verbose=False; r.max_retries=1
        r._clf=get_classifier("heuristic"); r._domain_clf=DomainClassifier()
        r._threshold=0.75; r.domain_routing=True; r.domain_min_confidence=0.40
        r._preferred_provider="openai"; r._registry=ModelRegistry()
        r._ollama_adapter=None; r.usage=UsageTracker()
        r._embedder=None; r.atlas=None; r.cache=None; r._adapter=None
        r.confidence_model=None; r.cost_tracker=None; r._fallback=None
        r._escalation_model=None
        with pytest.raises(ConfigurationError): r.chat("test")
    def test_bad_provider(self):
        with pytest.raises(ConfigurationError): ThinkRouter(provider="fake")
    @pytest.mark.asyncio
    async def test_achat(self):
        r=_mkrouter(); resp=await r.achat("test")
        assert isinstance(resp,RouterResponse)

class TestEndToEnd:
    clf_c=get_classifier("heuristic"); clf_d=DomainClassifier()
    @pytest.mark.parametrize("q,d",[
        ("Write a quicksort in Python.",Domain.CODE),
        ("Prove Fermat Last Theorem.",Domain.MATH),
        ("What are symptoms of diabetes?",Domain.MEDICAL),
        ("Explain GDPR compliance.",Domain.LEGAL),
        ("How do you calculate P/E ratio?",Domain.FINANCIAL),
        ("What is the capital of France?",Domain.GENERAL),
    ])
    def test_domain(self,q,d): assert self.clf_d.predict(q).domain==d
    @pytest.mark.parametrize("q",["What is 12*7?","Define osmosis."])
    def test_no_think(self,q): assert self.clf_c.predict(q).tier==Tier.NO_THINK
    @pytest.mark.parametrize("q",[
        "Prove by induction that sum of first n integers is n(n+1)/2.",
        "Write a Python implementation of a balanced binary search tree.",
    ])
    def test_full(self,q): assert self.clf_c.predict(q).tier==Tier.FULL
