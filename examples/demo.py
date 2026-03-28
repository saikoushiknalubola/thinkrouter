"""
examples/demo.py
~~~~~~~~~~~~~~~~

ThinkRouter demo — works in Google Colab with zero API keys.

Run in Colab:
    !pip install git+https://github.com/YOUR_USERNAME/thinkrouter.git -q
    exec(open("examples/demo.py").read())

Or copy-paste into a Colab cell after the pip install.
"""

from thinkrouter import ThinkRouter, Tier

# ── Initialise with "generic" provider — no API key needed ────────────────────
client = ThinkRouter(provider="generic")

queries = [
    # These should route to NO_THINK (50 tokens)
    "What is 7 * 8?",
    "Define entropy.",
    "Calculate 256 / 16.",
    "What is the capital of France?",
    "How many days are in a leap year?",
    "Who invented the telephone?",
    "Translate goodbye to Spanish.",
    # These should route to FULL (8,000 tokens)
    "Prove that sqrt(2) is irrational.",
    "Write a Python function to sort a list using merge sort.",
    "Design a distributed caching system with fault tolerance.",
    "Explain in detail how TCP congestion control works.",
    "Implement Dijkstra's algorithm with a priority queue.",
    "Debug and fix this deadlock in the multithreaded code.",
]

tier_icons = {Tier.NO_THINK: "[ 0 ]", Tier.SHORT: "[ 1 ]", Tier.FULL: "[ 2 ]"}

print()
print("  ThinkRouter — Routing Demo")
print("  " + "─" * 60)
print(f"  {'Tier':<12} {'Budget':>8} {'Conf':>6}  Query")
print("  " + "─" * 60)

for q in queries:
    r = client.classify(q)
    client.usage.record(q, r.tier, r.confidence, r.latency_ms)
    icon = tier_icons[r.tier]
    print(
        f"  {icon} {r.tier.name:<10} {r.token_budget:>6} tok "
        f"{r.confidence:>6.2f}  {q[:48]}"
    )

print()
client.usage.print_dashboard()

# ── Savings projection ────────────────────────────────────────────────────────
savings_pct = client.usage.summary().savings_pct
cost_per_m  = 15.0   # USD per million reasoning tokens (approximate o1 rate)

print("  Cost projection at scale")
print("  " + "─" * 44)
for vol, label in [(10_000, "10k queries/day"), (100_000, "100k queries/day"),
                   (1_000_000, "1M queries/day")]:
    baseline_cost = vol * 8_000 / 1_000_000 * cost_per_m
    saved_cost    = baseline_cost * (savings_pct / 100)
    print(f"  {label:<20}  save ${saved_cost:,.2f}/day  (${saved_cost*30:,.0f}/month)")
print()
