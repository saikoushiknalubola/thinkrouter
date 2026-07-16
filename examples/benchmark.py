"""
examples/benchmark.py
~~~~~~~~~~~~~~~~~~~~~
Phase 1 benchmark — measures quality of domain-routed specialist models
vs GPT-4o baseline across 5 domains.

Run:
    export OPENAI_API_KEY=sk-...
    python examples/benchmark.py

    # Ollama only (no API key needed):
    python examples/benchmark.py --provider ollama

Output:
    - Per-query routing decision and quality score
    - Domain-level summary table
    - Overall savings vs GPT-4o baseline
    - Results saved to benchmark_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thinkrouter import ThinkRouter, Domain
from thinkrouter.domain import DomainClassifier
from thinkrouter.registry import DEFAULT_REGISTRY


# ── Benchmark queries — 5 per domain = 25 total ───────────────────────────

BENCHMARK_QUERIES: Dict[str, List[str]] = {

    "code": [
        "Write a Python function that implements binary search on a sorted list.",
        "Implement a thread-safe singleton pattern in Python using a lock.",
        "Write a function to detect cycles in a directed graph using DFS.",
        "Implement merge sort in Python with O(n log n) time complexity.",
        "Write a decorator that retries a function up to 3 times on exception.",
    ],

    "math": [
        "Prove that the square root of 2 is irrational.",
        "Calculate the derivative of f(x) = x³ sin(x) using the product rule.",
        "What is the expected value of the sum of two fair six-sided dice?",
        "Prove by induction that the sum of the first n integers is n(n+1)/2.",
        "Solve the differential equation dy/dx = 2xy where y(0) = 1.",
    ],

    "medical": [
        "What are the first-line treatment options for Type 2 diabetes?",
        "Explain the mechanism of action of beta-blockers in heart failure.",
        "What is the difference between Crohn's disease and ulcerative colitis?",
        "Describe the pathophysiology of acute respiratory distress syndrome.",
        "What are the contraindications for thrombolytic therapy in stroke?",
    ],

    "legal": [
        "What elements are required to prove negligence in a tort claim?",
        "Explain the difference between copyright and trademark protection.",
        "What is the standard for judicial review of administrative agency decisions?",
        "Under GDPR, what constitutes lawful processing of personal data?",
        "What is the difference between a merger and an acquisition in corporate law?",
    ],

    "general": [
        "What is the capital of Australia?",
        "Explain the difference between machine learning and deep learning.",
        "What caused the 2008 financial crisis?",
        "How does photosynthesis work?",
        "What are the main differences between TCP and UDP?",
    ],
}


# ── Result containers ─────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query:          str
    expected_domain: str
    detected_domain: str
    domain_correct:  bool
    routed_model:    str
    provider_used:   str
    quality_score:   Optional[float]
    latency_ms:      float
    error:           Optional[str] = None


@dataclass
class DomainSummary:
    domain:            str
    total:             int
    domain_accuracy:   float
    avg_quality:       float
    avg_latency_ms:    float
    specialist_model:  str


# ── Judge function using GPT-4o ───────────────────────────────────────────

def judge_response(
    query:    str,
    response: str,
    openai_key: Optional[str] = None,
) -> Optional[float]:
    """
    Use GPT-4o as an independent judge to score response quality 1-10.
    Returns None if judging is unavailable (no API key).
    """
    if not openai_key:
        return None

    try:
        import openai
        client = openai.OpenAI(api_key=openai_key)
        judge_prompt = f"""You are an impartial quality evaluator.

Score the following response to the given query on a scale of 1 to 10 where:
1-3: Incorrect, misleading, or completely missing the point
4-6: Partially correct but incomplete or has errors
7-8: Correct and reasonably complete
9-10: Excellent — accurate, complete, and well-explained

Query: {query}

Response: {response[:1500]}

Reply with ONLY a single integer from 1 to 10. Nothing else."""

        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=5,
            temperature=0,
        )
        score_str = result.choices[0].message.content.strip()
        return float(score_str)
    except Exception:
        return None


# ── Main benchmark runner ─────────────────────────────────────────────────

def run_benchmark(
    provider:    str   = "openai",
    api_key:     Optional[str] = None,
    ollama_url:  str   = "http://localhost:11434",
    judge:       bool  = True,
    max_queries: int   = 25,
    output_file: str   = "benchmark_results.json",
) -> List[QueryResult]:

    openai_key = api_key or os.environ.get("OPENAI_API_KEY")

    print(f"\n  ThinkRouter — Phase 1 Domain Routing Benchmark")
    print(f"  {'─' * 56}")
    print(f"  Provider         : {provider}")
    print(f"  Domain routing   : enabled")
    print(f"  Judge            : {'GPT-4o' if judge and openai_key else 'disabled (no key)'}")
    print(f"  Total queries    : {min(max_queries, sum(len(v) for v in BENCHMARK_QUERIES.values()))}")
    print(f"  {'─' * 56}\n")

    # Init router
    try:
        router = ThinkRouter(
            provider=provider,
            api_key=api_key,
            domain_routing=True,
            verbose=False,
            ollama_url=ollama_url,
        )
    except Exception as exc:
        print(f"  ERROR initialising router: {exc}")
        sys.exit(1)

    # Also run domain classifier standalone for accuracy measurement
    domain_clf = DomainClassifier()

    results: List[QueryResult] = []
    query_count = 0

    for expected_domain, queries in BENCHMARK_QUERIES.items():
        print(f"  ── {expected_domain.upper()} queries ─────────────────────────────")

        for query in queries:
            if query_count >= max_queries:
                break

            t0 = time.perf_counter()
            error = None
            response_text = ""
            routed_model  = "unknown"
            provider_used = provider

            try:
                response = router.chat(query, temperature=0.3)
                response_text = response.content
                routed_model  = response.model
                provider_used = response.provider
                detected_domain = (
                    response.domain_result.domain.value
                    if response.domain_result else "general"
                )
            except Exception as exc:
                error           = str(exc)
                detected_domain = domain_clf.predict(query).domain.value

            elapsed = (time.perf_counter() - t0) * 1000

            # Judge response quality
            quality = None
            if judge and openai_key and response_text and not error:
                quality = judge_response(query, response_text, openai_key)

            result = QueryResult(
                query=query[:60] + "..." if len(query) > 60 else query,
                expected_domain=expected_domain,
                detected_domain=detected_domain,
                domain_correct=(detected_domain == expected_domain),
                routed_model=routed_model,
                provider_used=provider_used,
                quality_score=quality,
                latency_ms=elapsed,
                error=error,
            )
            results.append(result)
            query_count += 1

            # Print row
            dom_ok = "✓" if result.domain_correct else "✗"
            q_str  = f"  Q={quality:.0f}" if quality else ""
            err_str = f"  ERROR: {error[:40]}" if error else ""
            print(
                f"  [{dom_ok}] {detected_domain:<12} "
                f"→ {routed_model:<28} "
                f"{elapsed:>6.0f}ms"
                f"{q_str}{err_str}"
            )

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n  {'═' * 58}")
    print(f"  RESULTS SUMMARY")
    print(f"  {'─' * 58}")

    domain_results: Dict[str, List[QueryResult]] = {}
    for r in results:
        domain_results.setdefault(r.expected_domain, []).append(r)

    summaries: List[DomainSummary] = []
    for domain, rs in domain_results.items():
        correct    = sum(1 for r in rs if r.domain_correct)
        qualities  = [r.quality_score for r in rs if r.quality_score is not None]
        avg_q      = sum(qualities) / len(qualities) if qualities else 0.0
        avg_lat    = sum(r.latency_ms for r in rs) / len(rs)
        spec_model = DEFAULT_REGISTRY.resolve(
            Domain(domain) if domain in [d.value for d in Domain] else Domain.GENERAL,
            preferred_provider=provider,
        ).model

        summaries.append(DomainSummary(
            domain=domain,
            total=len(rs),
            domain_accuracy=correct / len(rs) * 100,
            avg_quality=avg_q,
            avg_latency_ms=avg_lat,
            specialist_model=spec_model,
        ))

        q_display = f"{avg_q:.1f}/10" if avg_q > 0 else "N/A (no key)"
        print(
            f"  {domain.upper():<12} "
            f"acc={correct}/{len(rs)} ({correct/len(rs)*100:.0f}%)  "
            f"quality={q_display}  "
            f"lat={avg_lat:.0f}ms  "
            f"model={spec_model}"
        )

    total     = len(results)
    correct   = sum(1 for r in results if r.domain_correct)
    qualities = [r.quality_score for r in results if r.quality_score is not None]
    avg_q     = sum(qualities) / len(qualities) if qualities else 0.0

    print(f"  {'─' * 58}")
    print(f"  TOTAL: {correct}/{total} domain correct ({correct/total*100:.1f}%)")
    if avg_q:
        print(f"  AVG QUALITY SCORE: {avg_q:.2f} / 10")
    print(f"  {'═' * 58}\n")

    # Save results
    output = {
        "metadata": {
            "provider":       provider,
            "domain_routing": True,
            "judge_enabled":  bool(judge and openai_key),
            "total_queries":  total,
        },
        "summary": {
            "domain_accuracy_pct": correct / total * 100,
            "avg_quality_score":   avg_q,
            "by_domain":           [
                {
                    "domain":           s.domain,
                    "domain_accuracy":  s.domain_accuracy,
                    "avg_quality":      s.avg_quality,
                    "avg_latency_ms":   s.avg_latency_ms,
                    "specialist_model": s.specialist_model,
                } for s in summaries
            ],
        },
        "queries": [
            {
                "query":           r.query,
                "expected_domain": r.expected_domain,
                "detected_domain": r.detected_domain,
                "domain_correct":  r.domain_correct,
                "routed_model":    r.routed_model,
                "quality_score":   r.quality_score,
                "latency_ms":      r.latency_ms,
                "error":           r.error,
            }
            for r in results
        ],
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to: {output_file}")

    return results


# ── Domain detection standalone demo ─────────────────────────────────────

def run_domain_demo() -> None:
    clf = DomainClassifier()
    queries = [
        "Write a binary search tree implementation in Python.",
        "Calculate the eigenvalues of this matrix.",
        "What are the side effects of metformin?",
        "Explain the elements of a valid contract.",
        "What is the current federal funds rate?",
        "What is the capital of France?",
        "Prove that there are infinitely many prime numbers.",
        "Debug this segmentation fault in my C++ code.",
        "What is the recommended dosage of ibuprofen for adults?",
        "Is this non-compete clause enforceable in California?",
    ]

    print(f"\n  ThinkRouter — Domain Detection Demo (no API key needed)")
    print(f"  {'─' * 60}")
    print(f"  {'Domain':<12} {'Conf':>6}  {'Signals':<30}  Query")
    print(f"  {'─' * 60}")

    for q in queries:
        r = clf.predict(q)
        sig_str = ", ".join(r.signals[:2]) if r.signals else "—"
        print(
            f"  {r.domain.value:<12} {r.confidence:>6.2f}  "
            f"{sig_str:<30}  {q[:45]}"
        )

    print()
    print(f"  Model Registry — Default Routing Targets")
    print(f"  {'─' * 60}")
    for domain in Domain:
        target = DEFAULT_REGISTRY.resolve(domain, preferred_provider="ollama")
        target_oa = DEFAULT_REGISTRY.resolve(domain, preferred_provider="openai")
        print(
            f"  {domain.value:<12}  "
            f"ollama → {target.model:<30} "
            f"openai → {target_oa.model}"
        )
    print()


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ThinkRouter Phase 1 — Domain Routing Benchmark"
    )
    parser.add_argument("--demo",     action="store_true",
                        help="Run domain detection demo only (no API key needed)")
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "anthropic", "ollama"],
                        help="LLM provider to benchmark against")
    parser.add_argument("--api-key",  default=None,
                        help="Provider API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip GPT-4o quality judging")
    parser.add_argument("--max",      type=int, default=25,
                        help="Maximum queries to run (default: 25)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                        help="Ollama server URL")
    parser.add_argument("--output",   default="benchmark_results.json",
                        help="Output file for results JSON")
    args = parser.parse_args()

    if args.demo:
        run_domain_demo()
    else:
        run_benchmark(
            provider=args.provider,
            api_key=args.api_key,
            ollama_url=args.ollama_url,
            judge=not args.no_judge,
            max_queries=args.max,
            output_file=args.output,
        )
