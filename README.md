# ThinkRouter

[![CI](https://github.com/thinkrouter/thinkrouter/actions/workflows/ci.yml/badge.svg)](https://github.com/saikoushiknalubola/thinkrouter/actions)
[![PyPI](https://img.shields.io/pypi/v/thinkrouter)](https://pypi.org/project/thinkrouter)
[![Python](https://img.shields.io/pypi/pyversions/thinkrouter)](https://pypi.org/project/thinkrouter)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1D7lZVyRauv3oeQU7QRSilMcwBGqunG79?usp=sharing)

---

**Reasoning models charge you 8,000 thinking tokens for "What is 2+3?"**  
**ThinkRouter fixes that with one line of code.**

```python
from thinkrouter import ThinkRouter

client   = ThinkRouter(provider="openai")
response = client.chat("What is the capital of France?")
# → routed to NO_THINK  — 50 tokens used, not 8,000

client.usage.print_dashboard()
```

```
  ThinkRouter — Usage Dashboard
  ────────────────────────────────────────────
  Total calls          : 1,247
  Tokens saved         : 8,734,750
  Compute savings      : 61.3%
  Avg classifier time  : 0.4 ms
```

**Validated on 1,319 real queries (GSM8K benchmark):**

| Threshold | Savings | Quality retained |
|-----------|---------|-----------------|
| 0.65 | 59% | 91% |
| **0.75** | **55%** | **93%** ← recommended |
| 0.85 | 44% | 96% |

---

## How it works

Every reasoning model call pays for a fixed extended thinking budget regardless
of the question's complexity. ThinkRouter intercepts each query, runs a
lightweight classifier (<5ms), and applies the minimum budget needed:

```
"What is 2+3?"                →  NO_THINK  →    50 tokens
"How does TCP work?"          →  SHORT     →   800 tokens
"Prove sqrt(2) is irrational" →  FULL      → 8,000 tokens
```

The classifier adds under 5ms per query. At production scale that overhead
is negligible compared to the token savings it generates.

---

## Installation

```bash
# Base install — works immediately, zero ML dependencies
pip install thinkrouter

# With fine-tuned DistilBERT classifier (higher accuracy)
pip install thinkrouter[classifier]

# With OpenAI client
pip install thinkrouter[openai]

# With Anthropic client
pip install thinkrouter[anthropic]

# Everything
pip install thinkrouter[all]
```

---

## Quick start

### OpenAI

```python
from thinkrouter import ThinkRouter

client = ThinkRouter(
    provider="openai",
    api_key="sk-...",       # or set OPENAI_API_KEY
    model="gpt-4o",
    verbose=True,           # prints routing decision per call
)

response = client.chat("Explain how merge sort works.")
print(response.content)
print(response.routing)
# ClassifierResult(tier=FULL, confidence=0.87, budget=8000 tokens, latency=1.2ms)

client.usage.print_dashboard()
```

### Anthropic

```python
client = ThinkRouter(
    provider="anthropic",
    api_key="sk-ant-...",   # or set ANTHROPIC_API_KEY
    model="claude-haiku-4-5-20251001",
)

response = client.chat("What is 144 divided by 12?")
# → NO_THINK — 50 tokens, not 8,000
```

### Streaming

```python
for chunk in client.stream("Explain quantum entanglement step by step."):
    print(chunk, end="", flush=True)
```

### Classify without an API call

```python
results = client.classify_batch([
    "What is 7 * 8?",
    "Design a distributed caching system.",
    "How many days are in a leap year?",
])
for r in results:
    print(f"{r.tier.name:<12}  conf={r.confidence:.2f}  budget={r.token_budget}")
```

```
NO_THINK      conf=0.88  budget=50
FULL          conf=0.85  budget=8000
NO_THINK      conf=0.80  budget=50
```

---

## Classifier backends

### Heuristic (default)

Zero external dependencies. Regex patterns + word-count heuristics.
Runs in under 1ms. Recommended for development and latency-sensitive production.

```python
client = ThinkRouter(classifier_backend="heuristic")
```

### DistilBERT (production accuracy)

Fine-tuned on the GSM8K mathematical reasoning dataset.
Achieves 93%+ quality retention at 60% compute savings.
Requires `pip install thinkrouter[classifier]`.

```python
client = ThinkRouter(
    classifier_backend="distilbert",
    confidence_threshold=0.75,
)
```

---

## Confidence threshold

Queries where the classifier's confidence is below the threshold fall back
conservatively to FULL — the safe default that never degrades output quality.

```python
client = ThinkRouter(confidence_threshold=0.80)  # more conservative
client = ThinkRouter(confidence_threshold=0.65)  # more aggressive savings
```

---

## API reference

### ThinkRouter

```python
ThinkRouter(
    provider             = "openai",     # "openai" | "anthropic" | "generic"
    api_key              = None,         # falls back to env var
    model                = None,         # default model for all calls
    classifier_backend   = "heuristic",  # "heuristic" | "distilbert"
    confidence_threshold = 0.75,
    max_records          = 10_000,       # usage tracker record limit
    verbose              = False,
    **client_kwargs,                     # passed to provider SDK client
)
```

### ThinkRouter.chat()

```python
response = client.chat(
    query,           # str — the user query
    model    = None, # override default model
    messages = None, # full message history (list of dicts)
    system   = None, # system prompt
    temperature = 0.7,
    **extra,         # forwarded to provider API
)
```

### RouterResponse

```python
response.content       # str — generated text
response.routing       # ClassifierResult
response.raw           # original provider response object
response.provider      # "openai" | "anthropic"
response.model         # model identifier string
response.usage_tokens  # {"prompt_tokens": N, "completion_tokens": M, ...}
```

### ClassifierResult

```python
result.tier          # Tier.NO_THINK | Tier.SHORT | Tier.FULL
result.confidence    # float in [0, 1]
result.token_budget  # int — thinking tokens assigned
result.latency_ms    # classifier wall-clock time
result.backend       # "heuristic" | "distilbert:cuda" | "distilbert:cpu"
```

---

## Running the tests

```bash
pip install thinkrouter[dev]
pytest tests/ -v
```

---

## Roadmap

- [x] Heuristic classifier (v0.1)
- [x] OpenAI and Anthropic adapters
- [x] Streaming support
- [x] Usage dashboard
- [x] GitHub Actions CI (Python 3.9–3.12)
- [ ] DistilBERT model on HuggingFace Hub
- [ ] Multi-domain training (MMLU, HumanEval, ARC-Challenge)
- [ ] Async support (`achat()`, `astream()`)
- [ ] Continuous budget regression
- [ ] Hosted API proxy (api.thinkrouter.ai)

---

## Research basis

ThinkRouter is grounded in published research:

- Zhao et al. (2025). *SelfBudgeter*. arXiv:2505.11274 — 74.47% savings validated
- Wang et al. (2025). *TALE-EP*. ACL Findings 2025 — 67% output token reduction
- Sanh et al. (2019). *DistilBERT*. arXiv:1910.01108 — classifier backbone
- Cobbe et al. (2021). *GSM8K*. arXiv:2110.14168 — training dataset

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs welcome.

---

## License

MIT — see [LICENSE](LICENSE).
