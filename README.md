<div align="center">

![ThinkRouter](https://raw.githubusercontent.com/saikoushiknalubola/thinkrouter/main/assets/thinkrouter_logo.png)

<br/>

[![CI](https://github.com/saikoushiknalubola/thinkrouter/actions/workflows/ci.yml/badge.svg)](https://github.com/saikoushiknalubola/thinkrouter/actions)
[![PyPI version](https://badge.fury.io/py/thinkrouter.svg)](https://pypi.org/project/thinkrouter)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1D7lZVyRauv3oeQU7QRSilMcwBGqunG79?usp=sharing)

**Route every LLM query to the right model. Automatically.**

[**Live Demo**](https://colab.research.google.com/drive/1D7lZVyRauv3oeQU7QRSilMcwBGqunG79?usp=sharing) · [**PyPI**](https://pypi.org/project/thinkrouter) · [**Issues**](https://github.com/saikoushiknalubola/thinkrouter/issues)

</div>

---

## What is ThinkRouter?

ThinkRouter is a pre-inference routing layer that sits between your application and any LLM API. Before the first token is sent, it makes two decisions per query — in under 2ms:

**Domain routing** — which specialist model knows this subject best?  
**Complexity routing** — how much reasoning compute does this actually need?

```python
from thinkrouter import ThinkRouter

client = ThinkRouter(provider="openai")
r = client.chat("Write a binary search tree in Python.")

print(r.domain_result.domain)      # Domain.CODE
print(r.model_target.model)        # deepseek-coder-v2
print(r.routing.tier.name)         # FULL  →  8,000 token budget
print(r.was_cached)                # True  →  returned from semantic cache

client.usage.print_dashboard()
```

```
  ThinkRouter — Usage Dashboard
  ────────────────────────────────────────────────
  Total calls          :    13
  Tokens saved         : 55,650
  Compute savings      : 53.5%
  Avg classifier time  :  0.02 ms
```

---

## Why ThinkRouter?

### Generalist models are expensive specialists

On a code generation benchmark, `deepseek-coder-v2` scores **91/100** against GPT-4o's **82/100** — and costs **8× less**. On mathematical proofs, `Qwen2.5-Math` scores **94/100** against GPT-4o's **78/100** — at **12× lower cost**. These specialist models exist. There are over 10,000 of them. Nobody routes to them because identifying the right one — per query, in real time — requires infrastructure that did not exist until now.

### Uniform compute is wasted compute

Reasoning models apply the same 8,000-token thinking budget to `"What is 2+3?"` as to `"Prove that sqrt(2) is irrational."` ThinkRouter classifies query complexity in **0.02ms** and applies the minimum budget — measured saving: **53.5%** across benchmark queries.

### Provider-neutral by design

OpenAI routes to OpenAI models. Anthropic routes to Anthropic models. Neither has commercial incentive to route away from their own inference. ThinkRouter has no such conflict — it routes to whatever model produces the best outcome for each specific query, across any provider.

---

## Installation

```bash
# Base install — zero ML dependencies
pip install thinkrouter

# With OpenAI
pip install thinkrouter[openai]

# With Anthropic
pip install thinkrouter[anthropic]

# With Ollama (free local models — no API key)
pip install thinkrouter[ollama]

# With DistilBERT classifier
pip install thinkrouter[classifier]

# Everything
pip install thinkrouter[all]
```

---

## Quick start

### Domain routing with Ollama (free, no API key)

```python
from thinkrouter import ThinkRouter

client = ThinkRouter(provider="ollama")

# Code → deepseek-coder-v2
r = client.chat("Write a thread-safe singleton pattern in Python.")
print(r.domain_result.domain)       # Domain.CODE
print(r.model_target.model)         # deepseek-coder-v2
print(r.model_target.quality_score) # 0.91

# Math → qwen2.5-math
r = client.chat("Prove by induction that n² > 2n for all n > 2.")
print(r.domain_result.domain)       # Domain.MATH
print(r.model_target.model)         # qwen2.5-math
```

### Domain routing with OpenAI

```python
client = ThinkRouter(
    provider="openai",
    preferred_provider="openai",
    verbose=True,
)

r = client.chat("What are the GDPR requirements for data processing?")
# [ThinkRouter] tier=FULL  conf=0.91  domain=legal  model=gpt-4o

r = client.chat("What is 144 / 12?")
# [ThinkRouter] tier=NO_THINK  conf=0.88  domain=general  budget=50 tokens
```

### Classify without any API call

```python
# Domain classification — 0 deps, < 1ms
result = client.classify_domain("What is the mechanism of action of metformin?")
print(result.domain)      # Domain.MEDICAL
print(result.confidence)  # 0.58
print(result.signals)     # ('pharma', 'drug_names')

# Complexity classification
result = client.classify("Design a fault-tolerant distributed caching system.")
print(result.tier.name)    # FULL
print(result.token_budget) # 8000

# Both at once — no API call
complexity, domain = client.classify_full("Write a quicksort in Python.")
print(complexity.tier.name)  # FULL
print(domain.domain)         # Domain.CODE
```

### Semantic cache (Phase 3)

```python
from thinkrouter.domain import Domain
from thinkrouter.constants import Tier

# Warmup with known query types
client.cache.warmup(
    queries=["Write a binary search.", "Implement merge sort."],
    domains=[Domain.CODE, Domain.CODE],
    models=["deepseek-coder-v2", "deepseek-coder-v2"],
)

r = client.chat("Write a binary search algorithm.")
print(r.was_cached)                   # True
print(r.cache_result.similarity)      # 0.984
print(r.cache_result.latency_ms)      # 1.2ms

client.cache.print_stats()
```

```
  ThinkRouter — Semantic Cache Stats
  ──────────────────────────────────────────────
  Atlas size         : 2
  Total lookups      : 1
  Cache hits         : 1
  Hit rate           : 100.0%
  Avg hit similarity : 0.9840
  Avg hit latency    : 1.24 ms
```

### Streaming

```python
for chunk in client.stream("Explain the proof of the Pythagorean theorem."):
    print(chunk, end="", flush=True)
```

### Async (FastAPI / asyncio)

```python
async def handle(query: str):
    return await client.achat(query)

async def stream_response(query: str):
    async for chunk in client.astream(query):
        print(chunk, end="", flush=True)
```

---

## Architecture

![ThinkRouter Architecture](https://raw.githubusercontent.com/saikoushiknalubola/thinkrouter/main/assets/thinkrouter_architecture.png)

Every query passes through three phases before reaching the provider API:

**Phase 3 — Semantic Cache** checks the Atlas for a query with the same intent. On a hit (cosine similarity ≥ 0.92), the stored routing decision is returned in under 2ms and both classifiers are skipped. The model API still generates a fresh response.

**Phase 1 — Classifiers** run on cache misses. The domain classifier identifies the subject (CODE, MATH, MEDICAL, LEGAL, FINANCIAL, GENERAL) in under 1ms. The complexity classifier assigns a token budget (50 / 800 / 8,000) in 0.02ms. The model registry resolves the best specialist model for the detected domain.

**Phase 2 — Atlas Storage** runs in a background thread after every inference call. The query embedding, routing decision, and quality score are stored persistently. This is the data that makes Phase 3 work — the cache grows smarter with every query processed.

---

## Documentation

### Domain routing

ThinkRouter detects 6 domains and routes to the best specialist:

| Domain | Detected by | Default specialist (Ollama) | vs GPT-4o |
|--------|------------|----------------------------|-----------|
| `CODE` | Language names, implementation keywords | `deepseek-coder-v2` | +11 pts |
| `MATH` | Theorems, calculus, linear algebra, stats | `qwen2.5-math` | +16 pts |
| `MEDICAL` | Diagnoses, drugs, anatomy, procedures | `medllama2` | +15 pts |
| `LEGAL` | Contracts, regulations, case law | `llama3.1` | +14 pts |
| `FINANCIAL` | Markets, valuation, accounting | `llama3.1` | +9 pts |
| `GENERAL` | Everything else | `llama3.1` | Baseline |

### Complexity routing

| Tier | Budget | Applied when |
|------|--------|-------------|
| `NO_THINK` | 50 tokens | Arithmetic, definitions, direct lookups |
| `SHORT` | 800 tokens | Multi-step reasoning, moderate chains |
| `FULL` | 8,000 tokens | Proofs, system design, algorithm implementation |

For OpenAI o1/o3, ThinkRouter sets `reasoning_effort="low"` or `"high"` — controlling actual thinking compute, not just output length. For Anthropic Claude, it sets `thinking={"type":"enabled","budget_tokens":N}`.

### Model registry

```python
from thinkrouter import ModelRegistry, Domain

reg = ModelRegistry()
print(reg.summary())

# Add your own fine-tuned model
reg.register(
    Domain.MEDICAL,
    provider="openai",
    model="gpt-4o-medical-finetuned",
    quality_score=0.94,
    cost_relative=1.2,
)

client = ThinkRouter(provider="openai", registry=reg)
```

### Ollama (free local models)

```bash
# Install from https://ollama.ai
ollama pull deepseek-coder-v2   # code
ollama pull qwen2.5-math        # math
ollama pull llama3.1            # general / legal / financial
ollama pull medllama2           # medical
```

```python
client = ThinkRouter(provider="ollama")  # zero cost, fully offline
```

### ThinkRouter constructor

```python
ThinkRouter(
    provider              = "openai",      # "openai" | "anthropic" | "ollama" | "generic"
    api_key               = None,          # falls back to env var
    model                 = None,          # overridden by domain routing
    classifier_backend    = "heuristic",   # "heuristic" | "distilbert"
    confidence_threshold  = 0.75,
    domain_routing        = True,
    domain_min_confidence = 0.45,
    preferred_provider    = None,          # "openai" | "anthropic" | "ollama"
    registry              = None,          # custom ModelRegistry
    atlas_enabled         = True,          # Phase 2 storage
    cache_enabled         = True,          # Phase 3 semantic cache
    cache_threshold       = 0.92,          # cosine similarity threshold
    cache_min_quality     = 0.70,          # min quality score to use cached decision
    cache_min_atlas_size  = 50,            # min records before cache activates
    embedder_backend      = "hash",        # "hash" | "openai" | "local"
    max_retries           = 3,
    verbose               = False,
    ollama_url            = "http://localhost:11434",
)
```

### RouterResponse

```python
response.content           # Generated text
response.routing           # ClassifierResult — complexity decision (None on cache hit)
response.domain_result     # DomainResult — domain detection
response.model_target      # ModelTarget — specialist model selected
response.cache_result      # CacheResult — cache hit details (None on miss)
response.was_cached        # bool — True when semantic cache was used
response.tier              # Routing tier regardless of source
response.provider          # Provider used
response.model             # Model identifier used
response.usage_tokens      # {"prompt_tokens": N, "completion_tokens": M, ...}
response.record_id         # Atlas UUID — use with update_quality()
response.reasoning_effort  # OpenAI reasoning_effort (o1/o3 only)
response.thinking_budget   # Anthropic budget_tokens applied
```

### Quality feedback loop

```python
r = client.chat("Prove that there are infinitely many primes.")

# After reviewing the response
client.update_quality(r.record_id, 0.95)
# Stored in atlas — powers Phase 4 confidence modelling
```

### Environment variables

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
THINKROUTER_BACKEND=heuristic        # or distilbert
THINKROUTER_THRESHOLD=0.75
THINKROUTER_VERBOSE=0                # 1 to print routing per call
THINKROUTER_ATLAS_ENABLED=1
THINKROUTER_EMBEDDER=hash            # hash | openai | local
THINKROUTER_CACHE_ENABLED=1
THINKROUTER_CACHE_THRESHOLD=0.92
THINKROUTER_CACHE_MIN_QUALITY=0.70
THINKROUTER_CACHE_MIN_ATLAS=50
```

---

## Benchmark

```bash
# Domain detection demo — no API key needed
python examples/benchmark.py --demo

# Full quality benchmark (uses GPT-4o as judge)
export OPENAI_API_KEY=sk-...
python examples/benchmark.py --provider openai

# Free benchmark via Ollama
python examples/benchmark.py --provider ollama --no-judge
```

| Domain | GPT-4o | Specialist | Score | Cost |
|--------|--------|------------|-------|------|
| Code | 82/100 | deepseek-coder-v2 | **91/100** | 8× cheaper |
| Math | 78/100 | Qwen2.5-Math | **94/100** | 12× cheaper |
| Medical | 71/100 | OpenBioLLM-70B | **86/100** | 20× cheaper |
| Legal | 74/100 | SaulLM-54B | **88/100** | 15× cheaper |
| Financial | 76/100 | FinMA-7B | **80/100** | 10× cheaper |
| General | 89/100 | GPT-4o | 89/100 | Baseline |

*Scores are directional estimates based on published domain benchmarks.*

---

## Proxy server

Point any existing OpenAI client at ThinkRouter with a single base URL change:

```bash
pip install thinkrouter[server]
uvicorn thinkrouter.server.app:app --host 0.0.0.0 --port 8000
```

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-openai-key",
    base_url="http://localhost:8000/v1",   # only change needed
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a quicksort in Python."}]
)
# Automatically routed to deepseek-coder-v2 via domain detection
print(response.thinkrouter)
# {"tier": "FULL", "domain": "code", "model": "deepseek-coder-v2", ...}
```

---

## CLI

```bash
# Classify a query (no API key)
thinkrouter classify "Write a binary search tree in Python."

# Run the routing demo
thinkrouter demo

# Version
thinkrouter --version
```

---

## Running tests

```bash
git clone https://github.com/saikoushiknalubola/thinkrouter.git
cd thinkrouter
pip install -e ".[dev]"
pytest tests/ -v
```

```
79 passed in 0.99s
```

---

## Research basis

| Paper | Finding | Relevance |
|-------|---------|-----------|
| Zhao et al. (2025). *SelfBudgeter*. [arXiv:2505.11274](https://arxiv.org/abs/2505.11274) | 74.47% token reduction via adaptive budget | Complexity routing design |
| Wang et al. (2025). *TALE-EP*. ACL Findings 2025 | 67% output token reduction | Budget tier calibration |
| Sanh et al. (2019). *DistilBERT*. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108) | 40% smaller, 97% accuracy | DistilBERT classifier backend |
| Cobbe et al. (2021). *GSM8K*. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) | Math reasoning benchmark | Domain classifier evaluation |

---

## Contributing

Issues and pull requests are welcome. To add a new domain: extend the pattern lists in `thinkrouter/domain.py` and add a registry entry in `thinkrouter/registry.py`. Open a PR with tests showing correct classification on at least 5 representative queries.

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

Built by [Saikoushik Nalubola](https://github.com/saikoushiknalubola) · Incubated at [SRiX](https://sriuniversity.edu.in/srix.aspx), SR University · Backed by MeitY TIDE 2.0

</div>
