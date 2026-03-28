"""
tests/test_thinkrouter.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Full test suite for the ThinkRouter library.
Run with:  pytest tests/ -v
"""

import threading
from unittest.mock import MagicMock

import pytest

from thinkrouter import (
    ClassifierResult,
    HeuristicClassifier,
    RouterResponse,
    ThinkRouter,
    Tier,
    TIER_TOKEN_BUDGETS,
    UsageTracker,
    get_classifier,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mock_router(content: str = "Test response.") -> ThinkRouter:
    """Return a ThinkRouter with a mocked provider adapter."""
    r = ThinkRouter.__new__(ThinkRouter)
    r.provider    = "openai"
    r.model       = "gpt-4o"
    r.verbose     = False
    r._classifier = HeuristicClassifier()
    r._threshold  = 0.75
    r.usage       = UsageTracker()

    a = MagicMock()
    a.call = MagicMock(return_value=(
        content,
        MagicMock(),
        {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    ))
    r._adapter = a
    return r


# ─── HeuristicClassifier ──────────────────────────────────────────────────────

class TestHeuristicClassifier:

    def setup_method(self):
        self.clf = HeuristicClassifier()

    # NO_THINK tier

    def test_arithmetic_is_no_think(self):
        assert self.clf.predict("What is 7 * 8?").tier == Tier.NO_THINK

    def test_capital_city_is_no_think(self):
        assert self.clf.predict("What is the capital of Germany?").tier == Tier.NO_THINK

    def test_define_word_is_no_think(self):
        assert self.clf.predict("Define photosynthesis.").tier == Tier.NO_THINK

    def test_translate_is_no_think(self):
        assert self.clf.predict("Translate hello to French").tier == Tier.NO_THINK

    def test_calculate_is_no_think(self):
        assert self.clf.predict("Calculate 100 / 4.").tier == Tier.NO_THINK

    def test_days_in_leap_year_is_no_think(self):
        assert self.clf.predict("How many days are in a leap year?").tier == Tier.NO_THINK

    # FULL tier

    def test_prove_short_is_full(self):
        assert self.clf.predict("Prove that sqrt(2) is irrational.").tier == Tier.FULL

    def test_prove_induction_is_full(self):
        assert self.clf.predict(
            "Prove by induction that sum of first n integers is n(n+1)/2."
        ).tier == Tier.FULL

    def test_system_design_is_full(self):
        assert self.clf.predict(
            "Design a distributed database architecture for a global e-commerce platform."
        ).tier == Tier.FULL

    def test_write_python_function_is_full(self):
        assert self.clf.predict(
            "Write a Python function that implements quicksort with randomised pivot selection."
        ).tier == Tier.FULL

    def test_write_python_implementation_is_full(self):
        assert self.clf.predict(
            "Write a Python implementation of a balanced binary search tree."
        ).tier == Tier.FULL

    def test_debug_code_is_full(self):
        assert self.clf.predict(
            "Debug and fix the race condition bug in this concurrent queue implementation."
        ).tier == Tier.FULL

    def test_explain_detail_is_full(self):
        assert self.clf.predict(
            "Explain in detail how TCP congestion control works."
        ).tier == Tier.FULL

    # ClassifierResult contract

    def test_confidence_in_range(self):
        r = self.clf.predict("test query")
        assert 0.0 <= r.confidence <= 1.0

    def test_token_budget_matches_tier(self):
        r = self.clf.predict("What is 2+2?")
        assert r.token_budget == TIER_TOKEN_BUDGETS[r.tier]

    def test_latency_under_100ms(self):
        assert self.clf.predict("simple test").latency_ms < 100.0

    def test_backend_is_heuristic(self):
        assert self.clf.predict("test").backend == "heuristic"

    def test_result_is_frozen(self):
        r = self.clf.predict("test")
        with pytest.raises(Exception):
            r.tier = Tier.FULL  # type: ignore

    # Batch

    def test_predict_batch_length(self):
        queries = ["q1", "q2", "q3", "q4"]
        assert len(self.clf.predict_batch(queries)) == len(queries)

    def test_predict_batch_all_results(self):
        results = self.clf.predict_batch(["easy", "hard design system"])
        assert all(isinstance(r, ClassifierResult) for r in results)


# ─── ClassifierResult ─────────────────────────────────────────────────────────

class TestClassifierResult:

    @pytest.mark.parametrize("tier", list(Tier))
    def test_budget_matches_constant(self, tier):
        r = ClassifierResult(tier=tier, confidence=0.9, latency_ms=1.0, backend="test")
        assert r.token_budget == TIER_TOKEN_BUDGETS[tier]

    def test_repr_contains_tier_name(self):
        r = ClassifierResult(tier=Tier.FULL, confidence=0.95, latency_ms=2.0, backend="h")
        assert "FULL" in repr(r)

    def test_repr_contains_confidence(self):
        r = ClassifierResult(tier=Tier.NO_THINK, confidence=0.88, latency_ms=1.0, backend="h")
        assert "0.880" in repr(r)


# ─── get_classifier factory ───────────────────────────────────────────────────

class TestGetClassifier:

    def test_heuristic_returns_HeuristicClassifier(self):
        from thinkrouter.classifier import HeuristicClassifier as HC
        assert isinstance(get_classifier("heuristic"), HC)

    def test_unknown_backend_raises_ValueError(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_classifier("nonexistent")

    def test_distilbert_raises_ImportError_without_deps(self):
        import sys
        from unittest.mock import patch
        with patch.dict(sys.modules, {"transformers": None, "torch": None}):
            from thinkrouter.classifier import DistilBertClassifier
            clf = DistilBertClassifier()
            with pytest.raises(ImportError, match="classifier"):
                clf.predict("test")


# ─── UsageTracker ─────────────────────────────────────────────────────────────

class TestUsageTracker:

    def setup_method(self):
        self.tracker = UsageTracker(max_records=100)

    def test_empty_summary_defaults(self):
        s = self.tracker.summary()
        assert s.total_calls == 0
        assert s.savings_pct == 0.0
        assert s.since is None

    def test_record_increments_total_calls(self):
        self.tracker.record("q", Tier.NO_THINK, 0.9, 1.0)
        assert self.tracker.summary().total_calls == 1

    def test_no_think_savings_correct(self):
        for _ in range(10):
            self.tracker.record("easy", Tier.NO_THINK, 0.9, 1.0)
        s = self.tracker.summary()
        expected = 10 * (TIER_TOKEN_BUDGETS[Tier.FULL] - TIER_TOKEN_BUDGETS[Tier.NO_THINK])
        assert s.total_tokens_saved == expected
        assert s.savings_pct > 0.0

    def test_full_tier_saves_nothing(self):
        self.tracker.record("hard", Tier.FULL, 0.95, 2.0)
        assert self.tracker.summary().total_tokens_saved == 0

    def test_mixed_tier_breakdown(self):
        self.tracker.record("easy",   Tier.NO_THINK, 0.88, 0.5)
        self.tracker.record("medium", Tier.SHORT,    0.72, 0.8)
        self.tracker.record("hard",   Tier.FULL,     0.91, 1.2)
        s = self.tracker.summary()
        assert s.total_calls == 3
        assert s.tier_breakdown[Tier.NO_THINK] == 1
        assert s.tier_breakdown[Tier.SHORT]    == 1
        assert s.tier_breakdown[Tier.FULL]     == 1

    def test_max_records_enforced(self):
        t = UsageTracker(max_records=5)
        for i in range(10):
            t.record(f"q{i}", Tier.SHORT, 0.7, 1.0)
        assert len(t._records) == 5

    def test_recent_returns_n_records(self):
        for i in range(30):
            self.tracker.record(f"q{i}", Tier.NO_THINK, 0.8, 0.5)
        assert len(self.tracker.recent(n=10)) == 10

    def test_reset_clears_all(self):
        self.tracker.record("q", Tier.FULL, 0.9, 1.0)
        self.tracker.reset()
        s = self.tracker.summary()
        assert s.total_calls == 0
        assert s.total_tokens_saved == 0

    def test_dashboard_string_contains_key_fields(self):
        self.tracker.record("test", Tier.NO_THINK, 0.9, 1.0)
        text = str(self.tracker.summary())
        assert "Total calls" in text
        assert "Compute savings" in text
        assert "no_think" in text

    def test_thread_safety(self):
        def worker():
            for _ in range(50):
                self.tracker.record("q", Tier.SHORT, 0.8, 0.5)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert self.tracker.summary().total_calls == 500


# ─── ThinkRouter ──────────────────────────────────────────────────────────────

class TestThinkRouter:

    def test_chat_returns_RouterResponse(self):
        r = _mock_router()
        assert isinstance(r.chat("What is 2+2?"), RouterResponse)

    def test_chat_content_correct(self):
        r = _mock_router("Hello world.")
        assert r.chat("test").content == "Hello world."

    def test_chat_routing_is_ClassifierResult(self):
        r = _mock_router()
        assert isinstance(r.chat("test").routing, ClassifierResult)

    def test_chat_provider_field(self):
        r = _mock_router()
        assert r.chat("test").provider == "openai"

    def test_no_think_query_uses_small_budget(self):
        r = _mock_router()
        r.chat("What is 2+2?")
        kw = r._adapter.call.call_args.kwargs
        assert kw["max_tokens"] == TIER_TOKEN_BUDGETS[Tier.NO_THINK]

    def test_full_query_uses_full_budget(self):
        r = _mock_router()
        r.chat("Design a distributed caching system with all tradeoffs explained in detail.")
        kw = r._adapter.call.call_args.kwargs
        assert kw["max_tokens"] == TIER_TOKEN_BUDGETS[Tier.FULL]

    def test_chat_records_in_usage_tracker(self):
        r = _mock_router()
        r.chat("Simple?")
        assert r.usage.summary().total_calls == 1

    def test_classify_does_not_call_api(self):
        r = _mock_router()
        result = r.classify("test query")
        r._adapter.call.assert_not_called()
        assert isinstance(result, ClassifierResult)

    def test_classify_batch_does_not_call_api(self):
        r = _mock_router()
        results = r.classify_batch(["q1", "q2", "q3"])
        r._adapter.call.assert_not_called()
        assert len(results) == 3

    def test_multiple_calls_accumulate_savings(self):
        r = _mock_router()
        for _ in range(5):
            r.chat("What is 2+2?")
        s = r.usage.summary()
        assert s.total_calls == 5
        assert s.savings_pct > 0.0

    def test_repr_contains_provider(self):
        r = _mock_router()
        assert "openai" in repr(r)

    def test_no_adapter_raises_RuntimeError(self):
        r = ThinkRouter.__new__(ThinkRouter)
        r.provider    = "generic"
        r.model       = "test"
        r.verbose     = False
        r._classifier = HeuristicClassifier()
        r._threshold  = 0.75
        r.usage       = UsageTracker()
        r._adapter    = None
        with pytest.raises(RuntimeError, match="No provider"):
            r.chat("test")

    def test_invalid_provider_raises_ValueError(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            ThinkRouter(provider="fakeprovider")  # type: ignore


# ─── End-to-end routing correctness ──────────────────────────────────────────

class TestEndToEndRouting:
    """
    Tests that verify correct routing decisions on real query strings
    using only the heuristic classifier — no API calls.
    """

    def setup_method(self):
        self.clf = HeuristicClassifier()

    NO_THINK_QUERIES = [
        "What is 12 * 7?",
        "Define osmosis.",
        "What is the capital of Japan?",
        "Calculate 100 / 4.",
        "How many days are in a leap year?",
        "Who invented the telephone?",
        "Translate goodbye to Spanish.",
    ]

    FULL_QUERIES = [
        "Prove by induction that sum of first n integers is n(n+1)/2.",
        "Write a Python implementation of a balanced binary search tree.",
        "Design a system architecture for a real-time collaborative document editor.",
        "Prove that there are infinitely many prime numbers.",
        "Explain in detail how garbage collection works in CPython.",
        "Write a comprehensive algorithm for detecting cycles in a directed graph.",
        "Write a Python function that implements merge sort recursively.",
        "Debug and fix the deadlock in this multithreaded Python application.",
        "Design a fault-tolerant distributed message queue system.",
        "Implement Dijkstra's algorithm in Python with a priority queue.",
    ]

    @pytest.mark.parametrize("query", NO_THINK_QUERIES)
    def test_no_think_query(self, query):
        assert self.clf.predict(query).tier == Tier.NO_THINK, (
            f"Expected NO_THINK for: {query!r}"
        )

    @pytest.mark.parametrize("query", FULL_QUERIES)
    def test_full_query(self, query):
        assert self.clf.predict(query).tier == Tier.FULL, (
            f"Expected FULL for: {query!r}"
        )

    def test_savings_dashboard_prints(self, capsys):
        r = ThinkRouter.__new__(ThinkRouter)
        r.provider    = "generic"
        r.model       = "test"
        r.verbose     = False
        r._classifier = HeuristicClassifier()
        r._threshold  = 0.75
        r.usage       = UsageTracker()
        r._adapter    = None

        for q in self.NO_THINK_QUERIES:
            result = r.classify(q)
            r.usage.record(q, result.tier, result.confidence, result.latency_ms)

        r.usage.print_dashboard()
        captured = capsys.readouterr()
        assert "ThinkRouter" in captured.out
        assert "Compute savings" in captured.out
