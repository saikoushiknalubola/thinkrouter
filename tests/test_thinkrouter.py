"""
tests/test_thinkrouter.py — v0.4.0 full test suite
Run:  pytest tests/ -v
"""
from __future__ import annotations

import os
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thinkrouter import (
    ClassifierResult, Config, Domain, DomainClassifier, DomainResult,
    HeuristicClassifier, ModelRegistry, ModelTarget, RouterResponse,
    ThinkRouter, Tier, TIER_TOKEN_BUDGETS, UsageTracker, get_classifier,
)
from thinkrouter.constants import (
    ANTHROPIC_THINKING_BUDGETS, ANTHROPIC_THINKING_MODELS,
    OPENAI_REASONING_EFFORT, OPENAI_REASONING_MODELS,
)
from thinkrouter.exceptions import (
    AuthenticationError, ClassifierError, ConfigurationError,
    ModelNotFoundError, ProviderError, RateLimitError, ThinkRouterError,
)


def _make_router(content: str = "Test.") -> ThinkRouter:
    r = ThinkRouter.__new__(ThinkRouter)
    r.provider              = "openai"
    r.model                 = "gpt-4o"
    r.verbose               = False
    r.max_retries           = 1
    r.domain_routing        = True
    r.domain_min_confidence = 0.40
    r._preferred_provider   = "openai"
    r._clf                  = HeuristicClassifier()
    r._domain_clf           = DomainClassifier()
    r._registry             = ModelRegistry(provider_priority=["openai"])
    r._threshold            = 0.75
    r.usage                 = UsageTracker()
    r._ollama_adapter       = None
    a = MagicMock()
    ret = (content, MagicMock(), {"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}, None)
    a.call  = MagicMock(return_value=ret)
    a.acall = AsyncMock(return_value=ret)
    r._adapter = a
    return r


# ── DomainClassifier ───────────────────────────────────────────────────────

class TestDomainClassifier:
    def setup_method(self): self.clf = DomainClassifier()

    def test_write_function_code(self):   assert self.clf.predict("Write a binary search function in Python.").domain == Domain.CODE
    def test_implement_algo_code(self):   assert self.clf.predict("Implement a red-black tree data structure.").domain == Domain.CODE
    def test_debug_code(self):            assert self.clf.predict("Debug the race condition in my Python threading code.").domain == Domain.CODE
    def test_sql_code(self):              assert self.clf.predict("Write a SQL query to join three tables.").domain == Domain.CODE
    def test_prove_math(self):            assert self.clf.predict("Prove that sqrt(2) is irrational.").domain == Domain.MATH
    def test_derivative_math(self):       assert self.clf.predict("Calculate the derivative of sin(x2).").domain == Domain.MATH
    def test_probability_math(self):      assert self.clf.predict("What is the probability of rolling two sixes?").domain == Domain.MATH
    def test_eigenvalue_math(self):       assert self.clf.predict("Find the eigenvalues of this 3x3 matrix.").domain == Domain.MATH
    def test_diabetes_medical(self):      assert self.clf.predict("What are the symptoms of type 2 diabetes?").domain == Domain.MEDICAL
    def test_drug_mechanism_medical(self):assert self.clf.predict("What is the mechanism of action of metformin?").domain == Domain.MEDICAL
    def test_contract_legal(self):        assert self.clf.predict("What are the elements of a valid contract?").domain == Domain.LEGAL
    def test_gdpr_legal(self):            assert self.clf.predict("Explain GDPR requirements for data processing.").domain == Domain.LEGAL
    def test_stock_financial(self):       assert self.clf.predict("Explain how to calculate the P/E ratio of a stock.").domain == Domain.FINANCIAL
    def test_dcf_financial(self):         assert self.clf.predict("How do you build a DCF valuation model?").domain == Domain.FINANCIAL
    def test_capital_general(self):       assert self.clf.predict("What is the capital of France?").domain == Domain.GENERAL
    def test_greeting_general(self):      assert self.clf.predict("Hello, how are you?").domain == Domain.GENERAL

    def test_confidence_range(self):
        assert 0.0 <= self.clf.predict("Write quicksort.").confidence <= 1.0

    def test_signals_tuple(self):
        r = self.clf.predict("Write a binary search function.")
        assert isinstance(r.signals, tuple)

    def test_backend_label(self):
        assert self.clf.predict("test").backend == "heuristic"

    def test_latency_fast(self):
        assert self.clf.predict("any query").latency_ms < 100.0

    def test_batch_length(self):
        assert len(self.clf.predict_batch(["code", "math", "general"])) == 3

    def test_batch_types(self):
        assert all(isinstance(r, DomainResult) for r in self.clf.predict_batch(["a","b"]))

    def test_scores_returns_dict(self):
        scores = self.clf.scores("Write a Python sorting algorithm.")
        assert isinstance(scores, dict)
        assert Domain.CODE in scores

    def test_strict_threshold_falls_back(self):
        clf = DomainClassifier(min_confidence=0.99)
        assert clf.predict("hello there").domain == Domain.GENERAL


# ── ModelRegistry ──────────────────────────────────────────────────────────

class TestModelRegistry:
    def setup_method(self): self.reg = ModelRegistry()

    def test_resolve_code_ollama(self):
        t = self.reg.resolve(Domain.CODE, preferred_provider="ollama")
        assert t.provider == "ollama" and t.domain == Domain.CODE

    def test_resolve_math_openai(self):
        t = self.reg.resolve(Domain.MATH, preferred_provider="openai")
        assert t.provider == "openai" and t.model == "gpt-4o"

    def test_all_domains_resolve(self):
        for domain in Domain:
            t = self.reg.resolve(domain)
            assert isinstance(t, ModelTarget)

    def test_best_for_sorted(self):
        targets = self.reg.best_for(Domain.CODE)
        scores  = [t.quality_score for t in targets]
        assert scores == sorted(scores, reverse=True)

    def test_register_custom(self):
        self.reg.register(Domain.CODE, "openai", "gpt-4-turbo",
                          quality_score=0.88, cost_relative=0.9)
        t = self.reg.resolve(Domain.CODE, preferred_provider="openai")
        assert t.model == "gpt-4-turbo"

    def test_summary_contains_domains(self):
        s = self.reg.summary()
        assert "CODE" in s and "MATH" in s and "ollama" in s


# ── Config ─────────────────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.classifier_backend == "heuristic"
        assert cfg.confidence_threshold == 0.75

    def test_env_override(self):
        with patch.dict(os.environ, {"THINKROUTER_VERBOSE":"1","THINKROUTER_THRESHOLD":"0.8"}):
            cfg = Config()
            assert cfg.verbose is True
            assert cfg.confidence_threshold == 0.8


# ── HeuristicClassifier ────────────────────────────────────────────────────

class TestHeuristicClassifier:
    def setup_method(self): self.clf = HeuristicClassifier()

    def test_arithmetic_no_think(self): assert self.clf.predict("What is 7 * 8?").tier == Tier.NO_THINK
    def test_capital_no_think(self):    assert self.clf.predict("What is the capital of Germany?").tier == Tier.NO_THINK
    def test_prove_full(self):          assert self.clf.predict("Prove that sqrt(2) is irrational.").tier == Tier.FULL
    def test_design_full(self):         assert self.clf.predict("Design a distributed database architecture.").tier == Tier.FULL
    def test_write_fn_full(self):       assert self.clf.predict("Write a Python function implementing quicksort.").tier == Tier.FULL
    def test_confidence_range(self):    assert 0.0 <= self.clf.predict("test").confidence <= 1.0
    def test_budget_correct(self):
        r = self.clf.predict("What is 2+2?")
        assert r.token_budget == TIER_TOKEN_BUDGETS[r.tier]
    def test_batch(self):               assert len(self.clf.predict_batch(["a","b","c"])) == 3


# ── Exceptions ─────────────────────────────────────────────────────────────

class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(ProviderError,      ThinkRouterError)
        assert issubclass(RateLimitError,     ProviderError)
        assert issubclass(AuthenticationError, ProviderError)
        assert issubclass(ModelNotFoundError,  ProviderError)
        assert issubclass(ClassifierError,     ThinkRouterError)
        assert issubclass(ConfigurationError,  ThinkRouterError)

    def test_provider_error_fields(self):
        exc = ProviderError("fail", 500, "openai")
        assert exc.status_code == 500 and exc.provider == "openai"


# ── UsageTracker ───────────────────────────────────────────────────────────

class TestUsageTracker:
    def setup_method(self): self.t = UsageTracker(max_records=100)

    def test_empty(self):           assert self.t.summary().total_calls == 0
    def test_record(self):
        self.t.record("q", Tier.NO_THINK, 0.9, 1.0)
        assert self.t.summary().total_calls == 1

    def test_savings_math(self):
        for _ in range(10): self.t.record("easy", Tier.NO_THINK, 0.9, 1.0)
        expected = 10 * (TIER_TOKEN_BUDGETS[Tier.FULL] - TIER_TOKEN_BUDGETS[Tier.NO_THINK])
        assert self.t.summary().total_tokens_saved == expected

    def test_thread_safety(self):
        def w():
            for _ in range(50): self.t.record("q", Tier.SHORT, 0.8, 0.5)
        ths = [threading.Thread(target=w) for _ in range(10)]
        for th in ths: th.start()
        for th in ths: th.join()
        assert self.t.summary().total_calls == 500


# ── ThinkRouter ────────────────────────────────────────────────────────────

class TestThinkRouter:

    def test_returns_response(self):
        assert isinstance(_make_router().chat("test"), RouterResponse)

    def test_content(self):
        assert _make_router("Hello.").chat("test").content == "Hello."

    def test_domain_result_populated(self):
        resp = _make_router().chat("Write a Python function to sort a list.")
        assert resp.domain_result is not None
        assert isinstance(resp.domain_result.domain, Domain)

    def test_code_domain_detected(self):
        resp = _make_router().chat("Write a binary search implementation in Python.")
        assert resp.domain_result is not None
        assert resp.domain_result.domain == Domain.CODE

    def test_math_domain_detected(self):
        resp = _make_router().chat("Prove that sqrt(2) is irrational.")
        assert resp.domain_result is not None
        assert resp.domain_result.domain == Domain.MATH

    def test_usage_tracked(self):
        r = _make_router()
        r.chat("test")
        assert r.usage.summary().total_calls == 1

    def test_classify_no_api(self):
        r = _make_router()
        r.classify("test")
        r._adapter.call.assert_not_called()

    def test_classify_domain_no_api(self):
        r = _make_router()
        result = r.classify_domain("Write a Python function.")
        r._adapter.call.assert_not_called()
        assert result.domain == Domain.CODE

    def test_classify_full_returns_both(self):
        r = _make_router()
        complexity, domain = r.classify_full("Prove that sqrt(2) is irrational.")
        assert isinstance(complexity, ClassifierResult)
        assert isinstance(domain, DomainResult)
        assert domain.domain == Domain.MATH

    def test_domain_routing_false_skips_domain(self):
        r = _make_router()
        r.domain_routing = False
        resp = r.chat("Write a binary search in Python.")
        assert resp.domain_result is None

    def test_no_adapter_raises(self):
        r = ThinkRouter.__new__(ThinkRouter)
        r.provider="generic"; r.model="t"; r.verbose=False; r.max_retries=1
        r._clf=HeuristicClassifier(); r._domain_clf=DomainClassifier()
        r._threshold=0.75; r.domain_routing=True; r.domain_min_confidence=0.40
        r._preferred_provider="openai"; r._registry=ModelRegistry()
        r._ollama_adapter=None; r.usage=UsageTracker(); r._adapter=None
        with pytest.raises(ConfigurationError):
            r.chat("test")

    def test_bad_provider_raises(self):
        with pytest.raises(ConfigurationError):
            ThinkRouter(provider="fake")  # type: ignore

    def test_no_openai_key_raises(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY":""}, clear=False):
            with pytest.raises(ConfigurationError):
                ThinkRouter(provider="openai", api_key="")

    def test_savings_accumulate(self):
        r = _make_router()
        for _ in range(5): r.chat("What is 2+2?")
        assert r.usage.summary().total_calls == 5
        assert r.usage.summary().savings_pct > 0

    def test_repr_domain_routing(self):
        assert "domain_routing=True" in repr(_make_router())

    @pytest.mark.asyncio
    async def test_achat_returns_response(self):
        r = _make_router()
        resp = await r.achat("test")
        assert isinstance(resp, RouterResponse)
        assert resp.content == "Test."

    @pytest.mark.asyncio
    async def test_achat_domain_populated(self):
        r = _make_router()
        resp = await r.achat("Write a binary search tree in Python.")
        assert resp.domain_result is not None
        assert resp.domain_result.domain == Domain.CODE


# ── Constants ──────────────────────────────────────────────────────────────

class TestConstants:
    def test_o1_reasoning(self):        assert "o1" in OPENAI_REASONING_MODELS
    def test_gpt4o_not_reasoning(self): assert "gpt-4o" not in OPENAI_REASONING_MODELS
    def test_full_effort_high(self):    assert OPENAI_REASONING_EFFORT[Tier.FULL] == "high"
    def test_notk_budget_zero(self):    assert ANTHROPIC_THINKING_BUDGETS[Tier.NO_THINK] == 0
    def test_claude_in_thinking(self):  assert "claude-opus-4-6" in ANTHROPIC_THINKING_MODELS


# ── End-to-end ────────────────────────────────────────────────────────────

class TestEndToEnd:
    clf_c = HeuristicClassifier()
    clf_d = DomainClassifier()

    @pytest.mark.parametrize("q,domain",[
        ("Write a quicksort in Python.",              Domain.CODE),
        ("Implement a linked list.",                  Domain.CODE),
        ("Prove Fermat's Last Theorem.",              Domain.MATH),
        ("Calculate the integral of e^x.",            Domain.MATH),
        ("What is the treatment for hypertension?",   Domain.MEDICAL),
        ("Explain GDPR data processing requirements.",Domain.LEGAL),
        ("How do you value a company using DCF?",     Domain.FINANCIAL),
        ("What is the capital of France?",            Domain.GENERAL),
    ])
    def test_domain(self, q, domain):
        assert self.clf_d.predict(q).domain == domain, f"Failed: {q!r}"

    @pytest.mark.parametrize("q",[
        "What is 12 * 7?","Define osmosis.","What is the capital of Japan?",
    ])
    def test_no_think(self, q):
        assert self.clf_c.predict(q).tier == Tier.NO_THINK

    @pytest.mark.parametrize("q",[
        "Prove by induction that sum of first n integers is n(n+1)/2.",
        "Write a Python implementation of a balanced binary search tree.",
        "Design a fault-tolerant distributed message queue system.",
    ])
    def test_full(self, q):
        assert self.clf_c.predict(q).tier == Tier.FULL


# ── CLI ────────────────────────────────────────────────────────────────────

class TestCLI:
    def test_classify_no_think(self, capsys):
        from thinkrouter.cli import cmd_classify
        cmd_classify("What is 7 * 8?", "heuristic")
        assert "NO_THINK" in capsys.readouterr().out

    def test_demo_runs(self, capsys):
        from thinkrouter.cli import cmd_demo
        cmd_demo()
        out = capsys.readouterr().out
        assert "ThinkRouter" in out and "Compute savings" in out
