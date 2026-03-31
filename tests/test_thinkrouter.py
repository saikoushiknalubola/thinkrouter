"""
tests/test_thinkrouter.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Complete test suite. Run:  pytest tests/ -v
"""
from __future__ import annotations

import os
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thinkrouter import (
    ClassifierResult,
    Config,
    HeuristicClassifier,
    RouterResponse,
    ThinkRouter,
    Tier,
    TIER_TOKEN_BUDGETS,
    UsageTracker,
    get_classifier,
)
from thinkrouter.constants import (
    ANTHROPIC_THINKING_BUDGETS,
    ANTHROPIC_THINKING_MODELS,
    OPENAI_REASONING_EFFORT,
    OPENAI_REASONING_MODELS,
)
from thinkrouter.exceptions import (
    AuthenticationError,
    ClassifierError,
    ConfigurationError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    ThinkRouterError,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_router(content: str = "Test.") -> ThinkRouter:
    r = ThinkRouter.__new__(ThinkRouter)
    r.provider    = "openai"
    r.model       = "gpt-4o"
    r.verbose     = False
    r.max_retries = 1
    r._clf        = HeuristicClassifier()
    r._threshold  = 0.75
    r.usage       = UsageTracker()
    a             = MagicMock()
    ret           = (content, MagicMock(), {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, None)
    a.call        = MagicMock(return_value=ret)
    a.acall       = AsyncMock(return_value=ret)
    r._adapter    = a
    return r


# ── Config ─────────────────────────────────────────────────────────────────

class TestConfig:

    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=False):
            cfg = Config()
            assert cfg.classifier_backend   == "heuristic"
            assert cfg.confidence_threshold == 0.75
            assert cfg.max_retries          == 3
            assert cfg.verbose              is False

    def test_env_override(self):
        env = {
            "THINKROUTER_BACKEND":   "distilbert",
            "THINKROUTER_THRESHOLD": "0.80",
            "THINKROUTER_VERBOSE":   "1",
        }
        with patch.dict(os.environ, env):
            cfg = Config()
            assert cfg.classifier_backend   == "distilbert"
            assert cfg.confidence_threshold == 0.80
            assert cfg.verbose              is True

    def test_repr(self):
        assert "heuristic" in repr(Config())


# ── HeuristicClassifier ────────────────────────────────────────────────────

class TestHeuristicClassifier:

    def setup_method(self):
        self.clf = HeuristicClassifier()

    # NO_THINK
    def test_arithmetic(self):      assert self.clf.predict("What is 7 * 8?").tier == Tier.NO_THINK
    def test_capital(self):         assert self.clf.predict("What is the capital of Germany?").tier == Tier.NO_THINK
    def test_define(self):          assert self.clf.predict("Define photosynthesis.").tier == Tier.NO_THINK
    def test_translate(self):       assert self.clf.predict("Translate hello to French").tier == Tier.NO_THINK
    def test_calculate(self):       assert self.clf.predict("Calculate 100 / 4.").tier == Tier.NO_THINK
    def test_days_year(self):       assert self.clf.predict("How many days are in a leap year?").tier == Tier.NO_THINK
    def test_who_invented(self):    assert self.clf.predict("Who invented the telephone?").tier == Tier.NO_THINK
    def test_greeting(self):        assert self.clf.predict("Hello!").tier == Tier.NO_THINK

    # FULL
    def test_prove_short(self):     assert self.clf.predict("Prove that sqrt(2) is irrational.").tier == Tier.FULL
    def test_prove_long(self):      assert self.clf.predict("Prove by induction that sum of first n integers is n(n+1)/2.").tier == Tier.FULL
    def test_design_system(self):   assert self.clf.predict("Design a distributed database architecture for a global e-commerce platform.").tier == Tier.FULL
    def test_write_fn(self):        assert self.clf.predict("Write a Python function that implements quicksort with randomised pivot selection.").tier == Tier.FULL
    def test_write_impl(self):      assert self.clf.predict("Write a Python implementation of a balanced binary search tree.").tier == Tier.FULL
    def test_debug_deadlock(self):  assert self.clf.predict("Debug and fix the deadlock bug in this multithreaded Python code.").tier == Tier.FULL
    def test_explain_detail(self):  assert self.clf.predict("Explain in detail how TCP congestion control works.").tier == Tier.FULL
    def test_fault_tolerant(self):  assert self.clf.predict("Design a fault-tolerant distributed message queue system.").tier == Tier.FULL
    def test_primes(self):          assert self.clf.predict("Prove that there are infinitely many prime numbers.").tier == Tier.FULL
    def test_dijkstra(self):        assert self.clf.predict("Implement Dijkstra's algorithm with a priority queue in Python.").tier == Tier.FULL
    def test_refactor(self):        assert self.clf.predict("Refactor this class to improve code quality and reduce memory usage.").tier == Tier.FULL
    def test_step_by_step(self):    assert self.clf.predict("Walk me through how to design a REST API step by step.").tier == Tier.FULL

    # Contract
    def test_confidence_range(self):
        assert 0.0 <= self.clf.predict("anything").confidence <= 1.0

    def test_budget_matches_tier(self):
        r = self.clf.predict("What is 2+2?")
        assert r.token_budget == TIER_TOKEN_BUDGETS[r.tier]

    def test_latency_fast(self):
        assert self.clf.predict("test").latency_ms < 100.0

    def test_backend_label(self):
        assert self.clf.predict("test").backend == "heuristic"

    def test_result_frozen(self):
        r = self.clf.predict("test")
        with pytest.raises(Exception):
            r.tier = Tier.FULL  # type: ignore

    def test_batch_length(self):
        assert len(self.clf.predict_batch(["a", "b", "c"])) == 3

    def test_batch_all_results(self):
        results = self.clf.predict_batch(["x", "y"])
        assert all(isinstance(r, ClassifierResult) for r in results)

    def test_batch_consistent_with_single(self):
        queries = ["What is 2+2?", "Design a distributed system."]
        single  = [self.clf.predict(q).tier for q in queries]
        batch   = [r.tier for r in self.clf.predict_batch(queries)]
        assert single == batch


# ── ClassifierResult ───────────────────────────────────────────────────────

class TestClassifierResult:

    @pytest.mark.parametrize("tier", list(Tier))
    def test_budget(self, tier):
        r = ClassifierResult(tier=tier, confidence=0.9, latency_ms=1.0, backend="t")
        assert r.token_budget == TIER_TOKEN_BUDGETS[tier]

    def test_repr_tier(self):
        assert "FULL" in repr(ClassifierResult(tier=Tier.FULL, confidence=0.95, latency_ms=2.0, backend="h"))

    def test_repr_confidence(self):
        assert "0.880" in repr(ClassifierResult(tier=Tier.NO_THINK, confidence=0.88, latency_ms=1.0, backend="h"))


# ── get_classifier ─────────────────────────────────────────────────────────

class TestGetClassifier:

    def test_heuristic(self):
        assert isinstance(get_classifier("heuristic"), HeuristicClassifier)

    def test_bad_backend(self):
        with pytest.raises(ValueError, match="Unknown classifier backend"):
            get_classifier("bogus")

    def test_distilbert_no_deps(self):
        import sys
        with patch.dict(sys.modules, {"transformers": None, "torch": None}):
            from thinkrouter.classifier import DistilBertClassifier
            clf = DistilBertClassifier()
            with pytest.raises(ClassifierError):
                clf.predict("test")


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
        assert exc.status_code == 500
        assert exc.provider    == "openai"

    def test_catchable_as_base(self):
        with pytest.raises(ThinkRouterError):
            raise RateLimitError("too many requests", 429, "openai")


# ── UsageTracker ───────────────────────────────────────────────────────────

class TestUsageTracker:

    def setup_method(self):
        self.t = UsageTracker(max_records=100)

    def test_empty(self):
        s = self.t.summary()
        assert s.total_calls == 0
        assert s.savings_pct == 0.0
        assert s.since is None

    def test_record(self):
        self.t.record("q", Tier.NO_THINK, 0.9, 1.0)
        assert self.t.summary().total_calls == 1

    def test_savings_math(self):
        for _ in range(10):
            self.t.record("easy", Tier.NO_THINK, 0.9, 1.0)
        s        = self.t.summary()
        expected = 10 * (TIER_TOKEN_BUDGETS[Tier.FULL] - TIER_TOKEN_BUDGETS[Tier.NO_THINK])
        assert s.total_tokens_saved == expected
        assert s.savings_pct > 0.0

    def test_full_saves_nothing(self):
        self.t.record("hard", Tier.FULL, 0.95, 2.0)
        assert self.t.summary().total_tokens_saved == 0

    def test_tier_breakdown(self):
        self.t.record("a", Tier.NO_THINK, 0.9, 1.0)
        self.t.record("b", Tier.SHORT,    0.7, 1.0)
        self.t.record("c", Tier.FULL,     0.9, 1.0)
        s = self.t.summary()
        assert s.tier_breakdown[Tier.NO_THINK] == 1
        assert s.tier_breakdown[Tier.SHORT]    == 1
        assert s.tier_breakdown[Tier.FULL]     == 1

    def test_max_records(self):
        t = UsageTracker(max_records=5)
        for i in range(10):
            t.record(f"q{i}", Tier.SHORT, 0.7, 1.0)
        assert len(t._records) == 5

    def test_recent(self):
        for i in range(30):
            self.t.record(f"q{i}", Tier.NO_THINK, 0.8, 0.5)
        assert len(self.t.recent(n=10)) == 10

    def test_reset(self):
        self.t.record("q", Tier.FULL, 0.9, 1.0)
        self.t.reset()
        assert self.t.summary().total_calls == 0

    def test_thread_safety(self):
        def w():
            for _ in range(50):
                self.t.record("q", Tier.SHORT, 0.8, 0.5)
        ths = [threading.Thread(target=w) for _ in range(10)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        assert self.t.summary().total_calls == 500

    def test_dashboard_str(self):
        self.t.record("x", Tier.NO_THINK, 0.9, 1.0)
        text = str(self.t.summary())
        assert "ThinkRouter" in text
        assert "Compute savings" in text
        assert "no_think" in text

    def test_record_with_model_and_provider(self):
        rec = self.t.record("q", Tier.FULL, 0.9, 2.0, model="gpt-4o", provider="openai")
        assert rec.model    == "gpt-4o"
        assert rec.provider == "openai"


# ── ThinkRouter ────────────────────────────────────────────────────────────

class TestThinkRouter:

    def test_returns_RouterResponse(self):
        assert isinstance(_make_router().chat("test"), RouterResponse)

    def test_content(self):
        assert _make_router("Hello.").chat("test").content == "Hello."

    def test_routing_field(self):
        assert isinstance(_make_router().chat("test").routing, ClassifierResult)

    def test_provider_field(self):
        assert _make_router().chat("test").provider == "openai"

    def test_no_think_tier_passed(self):
        r = _make_router()
        r.chat("What is 2+2?")
        assert r._adapter.call.call_args.kwargs["tier"] == Tier.NO_THINK

    def test_full_tier_passed(self):
        r = _make_router()
        r.chat("Design a fault-tolerant distributed caching system with detailed tradeoffs.")
        assert r._adapter.call.call_args.kwargs["tier"] == Tier.FULL

    def test_usage_tracked(self):
        r = _make_router()
        r.chat("test")
        assert r.usage.summary().total_calls == 1

    def test_classify_no_api(self):
        r = _make_router()
        r.classify("test")
        r._adapter.call.assert_not_called()

    def test_classify_batch(self):
        assert len(_make_router().classify_batch(["a", "b", "c"])) == 3

    def test_five_calls_savings(self):
        r = _make_router()
        for _ in range(5):
            r.chat("What is 2+2?")
        s = r.usage.summary()
        assert s.total_calls == 5
        assert s.savings_pct > 0.0

    def test_repr(self):
        assert "openai" in repr(_make_router())

    def test_no_adapter_raises(self):
        r = ThinkRouter.__new__(ThinkRouter)
        r.provider = "generic"; r.model = "t"; r.verbose = False
        r.max_retries = 1; r._clf = HeuristicClassifier()
        r._threshold = 0.75; r.usage = UsageTracker(); r._adapter = None
        with pytest.raises(ConfigurationError):
            r.chat("test")

    def test_bad_provider_raises(self):
        with pytest.raises(ConfigurationError):
            ThinkRouter(provider="fake")  # type: ignore

    def test_no_key_openai_raises(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            with pytest.raises(ConfigurationError, match="No OpenAI API key"):
                ThinkRouter(provider="openai", api_key="")

    @pytest.mark.asyncio
    async def test_achat(self):
        r    = _make_router()
        resp = await r.achat("test")
        assert isinstance(resp, RouterResponse)
        assert resp.content == "Test."

    @pytest.mark.asyncio
    async def test_achat_tracks_usage(self):
        r = _make_router()
        await r.achat("test")
        assert r.usage.summary().total_calls == 1


# ── Constants ──────────────────────────────────────────────────────────────

class TestConstants:

    def test_o1_in_reasoning_models(self):
        assert "o1" in OPENAI_REASONING_MODELS

    def test_o3_in_reasoning_models(self):
        assert "o3" in OPENAI_REASONING_MODELS

    def test_gpt4o_not_reasoning(self):
        assert "gpt-4o" not in OPENAI_REASONING_MODELS

    def test_full_effort_high(self):
        assert OPENAI_REASONING_EFFORT[Tier.FULL] == "high"

    def test_no_think_effort_low(self):
        assert OPENAI_REASONING_EFFORT[Tier.NO_THINK] == "low"

    def test_anthropic_no_think_zero(self):
        assert ANTHROPIC_THINKING_BUDGETS[Tier.NO_THINK] == 0

    def test_anthropic_full_large(self):
        assert ANTHROPIC_THINKING_BUDGETS[Tier.FULL] >= 8_000

    def test_claude_opus_in_thinking(self):
        assert "claude-opus-4-6" in ANTHROPIC_THINKING_MODELS


# ── CLI ────────────────────────────────────────────────────────────────────

class TestCLI:

    def test_classify_no_think(self, capsys):
        from thinkrouter.cli import cmd_classify
        cmd_classify("What is 7 * 8?", "heuristic")
        out = capsys.readouterr().out
        assert "NO_THINK" in out
        assert "50" in out

    def test_classify_full(self, capsys):
        from thinkrouter.cli import cmd_classify
        cmd_classify("Design a distributed system.", "heuristic")
        assert "FULL" in capsys.readouterr().out

    def test_demo(self, capsys):
        from thinkrouter.cli import cmd_demo
        cmd_demo()
        out = capsys.readouterr().out
        assert "ThinkRouter" in out
        assert "Compute savings" in out
        assert "queries/day" in out


# ── End-to-end routing ────────────────────────────────────────────────────

class TestEndToEndRouting:

    clf = HeuristicClassifier()

    NO_THINK_QUERIES = [
        "What is 12 * 7?",
        "Define osmosis.",
        "What is the capital of Japan?",
        "Calculate 100 / 4.",
        "How many days are in a leap year?",
        "Who invented the telephone?",
        "Translate goodbye to Spanish.",
        "Hello, how are you?",
    ]

    FULL_QUERIES = [
        "Prove by induction that sum of first n integers is n(n+1)/2.",
        "Write a Python implementation of a balanced binary search tree.",
        "Design a system architecture for a real-time collaborative document editor.",
        "Prove that there are infinitely many prime numbers.",
        "Explain in detail how garbage collection works in CPython.",
        "Write a Python function that implements merge sort recursively.",
        "Debug and fix the deadlock bug in this multithreaded Python code.",
        "Design a fault-tolerant distributed message queue system.",
        "Implement Dijkstra's algorithm with a priority queue in Python.",
        "Analyse the time complexity of this dynamic programming solution.",
    ]

    @pytest.mark.parametrize("query", NO_THINK_QUERIES)
    def test_no_think(self, query):
        assert self.clf.predict(query).tier == Tier.NO_THINK, f"Expected NO_THINK: {query!r}"

    @pytest.mark.parametrize("query", FULL_QUERIES)
    def test_full(self, query):
        assert self.clf.predict(query).tier == Tier.FULL, f"Expected FULL: {query!r}"
