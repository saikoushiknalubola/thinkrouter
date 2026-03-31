"""
thinkrouter.cli
~~~~~~~~~~~~~~~
Command-line interface for ThinkRouter.

Usage:
    thinkrouter classify "What is 7 * 8?"
    thinkrouter classify "Prove that sqrt(2) is irrational."
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
    print(f"\n  Query    : {query}")
    print(f"  Tier     : {result.tier.name}")
    print(f"  Budget   : {result.token_budget:,} thinking tokens")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Latency  : {result.latency_ms:.2f} ms")
    print(f"  Backend  : {result.backend}\n")


def cmd_demo() -> None:
    from thinkrouter import ThinkRouter

    client = ThinkRouter(provider="generic")
    queries = [
        "What is 7 * 8?",
        "Define entropy.",
        "Calculate 256 / 16.",
        "What is the capital of France?",
        "Prove that sqrt(2) is irrational.",
        "Write a Python function to implement merge sort.",
        "Design a distributed caching system.",
        "Explain in detail how TCP congestion control works.",
    ]
    print(f"\n  {'Tier':<12} {'Budget':>8} {'Conf':>6}  Query")
    print("  " + "─" * 62)
    for q in queries:
        r = client.classify(q)
        client.usage.record(q, r.tier, r.confidence, r.latency_ms)
        print(f"  {r.tier.name:<12} {r.token_budget:>6} tok {r.confidence:>6.2f}  {q[:45]}")
    print()
    client.usage.print_dashboard()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="thinkrouter",
        description="ThinkRouter — pre-inference LLM query routing",
    )
    parser.add_argument(
        "--version", action="version",
        version="%(prog)s 0.2.0",
    )
    sub = parser.add_subparsers(dest="command")

    p_classify = sub.add_parser("classify", help="Classify a single query")
    p_classify.add_argument("query", type=str, help="Query string to classify")
    p_classify.add_argument(
        "--backend", default="heuristic",
        choices=["heuristic", "distilbert"],
        help="Classifier backend (default: heuristic)",
    )

    sub.add_parser("demo", help="Run the built-in routing demo")

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
