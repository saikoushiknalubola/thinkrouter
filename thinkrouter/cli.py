"""
thinkrouter.cli
~~~~~~~~~~~~~~~
Command-line interface.

Usage:
    thinkrouter classify "What is 7 * 8?"
    thinkrouter classify "Prove sqrt(2) is irrational." --backend heuristic
    thinkrouter demo
    thinkrouter --version
"""
from __future__ import annotations

import argparse
import sys


def cmd_classify(query: str, backend: str) -> None:
    from thinkrouter.classifier import get_classifier
    clf    = get_classifier(backend)
    result = clf.predict(query)
    print(f"\n  Query      : {query}")
    print(f"  Tier       : {result.tier.name}")
    print(f"  Budget     : {result.token_budget:,} thinking tokens")
    print(f"  Confidence : {result.confidence:.3f}")
    print(f"  Latency    : {result.latency_ms:.2f} ms")
    print(f"  Backend    : {result.backend}\n")


def cmd_demo() -> None:
    from thinkrouter import ThinkRouter

    client  = ThinkRouter(provider="generic")
    queries = [
        "What is 7 * 8?",
        "Define entropy.",
        "Calculate 256 / 16.",
        "What is the capital of France?",
        "How many days are in a leap year?",
        "Who invented the telephone?",
        "Prove that sqrt(2) is irrational.",
        "Write a Python function to implement merge sort.",
        "Design a fault-tolerant distributed caching system.",
        "Explain in detail how TCP congestion control works.",
        "Implement Dijkstra's algorithm with a priority queue.",
        "Debug and fix the deadlock bug in this multithreaded code.",
    ]
    print(f"\n  {'Tier':<12} {'Budget':>8} {'Conf':>6}  Query")
    print("  " + "─" * 65)
    for q in queries:
        r = client.classify(q)
        client.usage.record(q, r.tier, r.confidence, r.latency_ms)
        print(f"  {r.tier.name:<12} {r.token_budget:>6} tok {r.confidence:>6.2f}  {q[:48]}")
    print()
    client.usage.print_dashboard()

    # Cost projection
    s   = client.usage.summary()
    cpm = 15.0   # USD per million reasoning tokens (approximate o1 rate)
    print("  Cost projection at scale")
    print("  " + "─" * 46)
    for vol, label in [
        (10_000,     "10k queries/day   "),
        (100_000,    "100k queries/day  "),
        (1_000_000,  "1M queries/day    "),
    ]:
        saved = vol * 8_000 / 1_000_000 * cpm * (s.savings_pct / 100)
        print(f"  {label}  save ${saved:>10,.2f}/day  (${saved*30:>12,.0f}/month)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="thinkrouter",
        description="ThinkRouter — pre-inference LLM query routing",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.3.0")
    sub = parser.add_subparsers(dest="command")

    p_clf = sub.add_parser("classify", help="Classify a single query")
    p_clf.add_argument("query", type=str)
    p_clf.add_argument(
        "--backend", default="heuristic",
        choices=["heuristic", "distilbert"],
    )
    sub.add_parser("demo", help="Run the routing demo")

    args = parser.parse_args()
    if args.command == "classify":
        cmd_classify(args.query, args.backend)
    elif args.command == "demo":
        cmd_demo()
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
