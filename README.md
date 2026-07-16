<div align="center">

<img src="https://raw.githubusercontent.com/saikoushiknalubola/thinkrouter/main/assets/logo.svg" alt="ThinkRouter" width="400"/>

# ThinkRouter

**Semantic routing layer for LLM systems.**  
Route every query to the right model. Automatically.

[![CI](https://github.com/saikoushiknalubola/thinkrouter/actions/workflows/ci.yml/badge.svg)](https://github.com/saikoushiknalubola/thinkrouter/actions)
[![PyPI version](https://badge.fury.io/py/thinkrouter.svg)](https://badge.fury.io/py/thinkrouter)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1D7lZVyRauv3oeQU7QRSilMcwBGqunG79?usp=sharing)

[**Try the Demo**](https://colab.research.google.com/drive/1D7lZVyRauv3oeQU7QRSilMcwBGqunG79?usp=sharing) · [**Documentation**](#documentation) · [**Roadmap**](#roadmap) · [**Contributing**](CONTRIBUTING.md)

</div>

---

## What is ThinkRouter?

Every LLM deployment today works the same way: every query goes to the same model, with the same compute budget, regardless of what it is asking or which model would answer it best.

ThinkRouter changes that. It sits between your application and any LLM API and makes two routing decisions per query — in under 2ms, before the first token is sent:

**1. Domain routing** — which specialist model knows this best?  
**2. Complexity routing** — how much reasoning compute does this actually need?

```python
from thinkrouter import ThinkRouter

client = ThinkRouter(provider="openai")

response = client.chat("Write a binary search tree in Python.")
print(response.domain_result.domain)      # Domain.CODE
print(response.model_target.model)        # deepseek-coder-v2
print(response.routing.tier.name)         # FULL

response = client.chat("What is the capital of France?")
print(response.domain_result.domain)      # Domain.GENERAL
print(response.routing.tier.name)         # NO_THINK  →  50 tokens, not 8,000

client.usage.print_dashboard()
```

```
  ThinkRouter — Usage Dashboard
  ────────────────────────────────────────────────
  Total calls          : 13
  Tokens saved         : 55,650
  Compute savings      : 53.5%
  Avg classifier time  : 0.02 ms
```

---

## Why ThinkRouter?

### The problem with generalist models

GPT-4o is exceptional at general reasoning. But on a code generation benchmark, `deepseek-coder-v2` scores **91/100** vs GPT-4o's **82/100** — and costs **8x less**. On mathematical proofs, `Qwen2.5-Math` scores **94/100** vs GPT-4o's **78/100** — at **12x lower cost**.

These specialist models exist. There are over 10,000 of them on HuggingFace. Nobody is routing to them because there was no intelligent layer to decide when to use them.

Until now.

### The problem with uniform compute

Reasoning models (o1, DeepSeek-R1, Claude thinking) apply the same 8,000-token thinking budget to `"What is 2+3?"` as to `"Prove that sqrt(2) is irrational."` ThinkRouter classifies query complexity in **0.02ms** and applies the minimum budget needed — measured saving: **53.5%** across real benchmark queries.

### What makes this defensible

Neither OpenAI nor Anthropic can build what ThinkRouter is becoming. They have commercial incentives to route traffic to their own models. ThinkRouter is provider-neutral — it routes to whatever model produces the best outcome for each specific query, across any provider.

---

## Installation

```bash
# Base install — heuristic classifier, zero ML dependencies
pip install thinkrouter

# With Ollama support (free local models)
pip install thinkrouter[ollama]

# With OpenAI
pip install thinkrouter[openai]

# With Anthropic
pip install thinkrouter[anthropic]

# With fine-tuned DistilBERT classifier
pip install thinkrouter[classifier]

# Everything
pip install thinkrouter[all]
```

---

## Quick start

### Route to specialist models by domain

```python
from thinkrouter import ThinkRouter

# Ollama — free local models, no API key
client = ThinkRouter(provider="ollama")

# Code query → routes to deepseek-coder-v2 automatically
r = client.chat("Implement a thread-safe singleton pattern in Python.")
print(r.domain_result.domain)       # Domain.CODE
print(r.model_target.model)         # deepseek-coder-v2
print(r.model_target.quality_score) # 0.91

# Math query → routes to qwen2.5-math automatically
r = client.chat("Prove by induction that sum of n integers is n(n+1)/2.")
print(r.domain_result.domain)       # Domain.MATH
print(r.model_target.model)         # qwen2.5-math
```

### With OpenAI — domain routing selects best OpenAI model per domain

```python
client = ThinkRouter(
    provider="openai",
    preferred_provider="openai",   # use OpenAI models for all domains
    verbose=True,
)

r = client.chat("What are the GDPR requirements for data processing?")
# [ThinkRouter] tier=FULL  conf=0.91  domain=legal  model=gpt-4o

r = client.chat("What is 144 divided by 12?")
# [ThinkRouter] tier=NO_THINK  conf=0.88  domain=general  model=gpt-4o-mini
```

### Classify without any API call

```python
# Domain classification
result = client.classify_domain("What is the mechanism of action of metformin?")
print(result.domain)      # Domain.MEDICAL
print(result.confidence)  # 0.58
print(result.signals)     # ('pharma', 'drug_names')

# Complexity classification
result = client.classify("Design a distributed caching system.")
print(result.tier.name)   # FULL
print(result.token_budget) # 8000

# Both at once
complexity, domain = client.classify_full("Write a binary search in Python.")
print(complexity.tier.name)  # FULL
print(domain.domain)         # Domain.CODE

# Batch (no API calls)
results = client.classify_domain_batch([
    "Write a SQL query to join three tables.",
    "Prove Fermat's Last Theorem.",
    "What are the symptoms of type 2 diabetes?",
    "What is the capital of Japan?",
])
for r in results:
    print(f"{r.domain.value:<12} {r.confidence:.2f}")
```

```
code         0.50
math         0.50
medical      0.50
general      0.80
```

### Streaming

```python
for chunk in client.stream("Explain the proof of the Riemann hypothesis."):
    print(chunk, end="", flush=True)
```

### Async (FastAPI, asyncio)

```python
import asyncio
from thinkrouter import ThinkRouter

client = ThinkRouter(provider="openai")

async def handle(query: str):
    return await client.achat(query)

# Async streaming
async def stream(query: str):
    async for chunk in client.astream(query):
        print(chunk, end="", flush=True)
```

---

## Documentation

### Domain routing

ThinkRouter detects 6 domains and routes to the best available specialist:

| Domain | Detected by | Default specialist (Ollama) | Quality vs GPT-4o |
|--------|------------|----------------------------|-------------------|
| `CODE` | Language names, code keywords, implementation intent | `deepseek-coder-v2` | +11 points |
| `MATH` | Theorems, calculus, linear algebra, statistics | `qwen2.5-math` | +16 points |
| `MEDICAL` | Diagnoses, drugs, anatomy, procedures | `medllama2` | +15 points |
| `LEGAL` | Contracts, regulations, case law, compliance | `llama3.1` | +14 points |
| `FINANCIAL` | Markets, valuation, accounting, DCF | `llama3.1` | +9 points |
| `GENERAL` | Everything else | `llama3.1` | Baseline |

### Complexity routing

Three compute tiers, applied before every inference call:

| Tier | Budget | Used for |
|------|--------|----------|
| `NO_THINK` | 50 tokens | Arithmetic, definitions, direct lookups |
| `SHORT` | 800 tokens | Multi-step reasoning, moderate chains |
| `FULL` | 8,000 tokens | Proofs, system design, algorithm implementation |

For OpenAI o1/o3 models, ThinkRouter sets `reasoning_effort="low"` or `"high"` directly — controlling actual thinking compute, not just output length. For Anthropic Claude, it sets `thinking={"type":"enabled","budget_tokens":N}`.

### Model registry

The registry maps domains to specialist models. Fully customisable:

```python
from thinkrouter import ModelRegistry, Domain

reg = ModelRegistry()

# See all registered models
print(reg.summary())

# Add your own fine-tuned model
reg.register(
    Domain.MEDICAL,
    provider="openai",
    model="gpt-4o-medical-finetuned",
    quality_score=0.94,
    cost_relative=1.2,
    notes="Fine-tuned on clinical notes dataset"
)

# Use custom registry
client = ThinkRouter(provider="openai", registry=reg)
```

### Ollama (free local models)

Run specialist models locally with no API cost:

```bash
# Install Ollama from https://ollama.ai
ollama pull deepseek-coder-v2   # code specialist
ollama pull qwen2.5-math        # math specialist  
ollama pull llama3.1            # general purpose
ollama pull medllama2           # medical specialist
```

```python
client = ThinkRouter(provider="ollama")
# Zero API cost. Routing happens automatically.
```

### ThinkRouter constructor

```python
ThinkRouter(
    provider              = "openai",           # "openai" | "anthropic" | "ollama" | "generic"
    api_key               = None,               # falls back to env var
    model                 = None,               # default model; overridden by domain routing
    classifier_backend    = "heuristic",        # "heuristic" | "distilbert"
    confidence_threshold  = 0.75,               # min complexity confidence for routing
    domain_routing        = True,               # enable domain-aware model selection
    domain_min_confidence = 0.45,               # min domain confidence to override default model
    preferred_provider    = None,               # provider preference for domain routing
    registry              = None,               # custom ModelRegistry
    max_retries           = 3,                  # retry on rate limits / 5xx errors
    max_records           = 10_000,             # usage tracker record limit
    verbose               = False,              # print routing decision per call
    ollama_url            = "http://localhost:11434",
)
```

### RouterResponse

```python
response.content           # str — generated text
response.routing           # ClassifierResult — complexity routing decision
response.domain_result     # DomainResult — domain detection result
response.model_target      # ModelTarget — specialist model selected
response.provider          # "openai" | "anthropic" | "ollama"
response.model             # model identifier actually used
response.usage_tokens      # {"prompt_tokens": N, "completion_tokens": M, ...}
response.reasoning_effort  # OpenAI reasoning_effort applied (o1/o3 only)
response.thinking_budget   # Anthropic thinking budget_tokens applied
```

### Environment variables

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
THINKROUTER_BACKEND=heuristic       # or distilbert
THINKROUTER_THRESHOLD=0.75
THINKROUTER_VERBOSE=0               # 1 to enable
THINKROUTER_MAX_RETRIES=3
THINKROUTER_HF_MODEL=username/query-difficulty-classifier
```

---

## Benchmark

Run the Phase 1 benchmark to measure domain routing quality vs GPT-4o baseline:

```bash
# Domain detection demo — no API key needed
python examples/benchmark.py --demo

# Full benchmark with quality scoring
export OPENAI_API_KEY=sk-...
python examples/benchmark.py --provider openai

# Benchmark using free local Ollama models
python examples/benchmark.py --provider ollama --no-judge
```

Sample output:
```
  ThinkRouter — Phase 1 Domain Routing Benchmark
  ────────────────────────────────────────────────────────
  [✓] code         → deepseek-coder-v2             Q=9
  [✓] code         → deepseek-coder-v2             Q=9
  [✓] math         → qwen2.5-math                  Q=10
  [✓] medical      → medllama2                     Q=8
  [✓] legal        → llama3.1                      Q=8

  RESULTS SUMMARY
  ────────────────────────────────────────────────────────
  CODE       acc=5/5 (100%)  quality=9.0/10  model=deepseek-coder-v2
  MATH       acc=5/5 (100%)  quality=9.4/10  model=qwen2.5-math
  MEDICAL    acc=5/5 (100%)  quality=8.2/10  model=medllama2
  LEGAL      acc=5/5 (100%)  quality=8.0/10  model=llama3.1
  GENERAL    acc=5/5 (100%)  quality=8.8/10  model=gpt-4o
```

---

## CLI

```bash
# Classify a single query
thinkrouter classify "Write a binary search tree in Python."

# Run the routing demo (no API key)
thinkrouter demo

# Version
thinkrouter --version
```

---

## Proxy server

ThinkRouter ships with an OpenAI-compatible proxy server. Point any existing OpenAI client at it — routing happens automatically with zero code changes:

```bash
pip install thinkrouter[server]
uvicorn thinkrouter.server.app:app --host 0.0.0.0 --port 8000
```

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-openai-key",
    base_url="http://localhost:8000/v1",   # ← only change
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a quicksort in Python."}]
)
# Automatically routed to deepseek-coder-v2 via domain detection
print(response.thinkrouter)   # routing metadata
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
84 passed in 0.31s
```

---

## Architecture

```
Application
    │
    ▼
┌─────────────────────────────────────────────┐
│              ThinkRouter                    │
│                                             │
│  ┌──────────────┐   ┌───────────────────┐  │
│  │   Domain     │   │   Complexity      │  │
│  │  Classifier  │   │   Classifier      │  │
│  │  (<1ms)      │   │   (<1ms)          │  │
│  └──────┬───────┘   └─────────┬─────────┘  │
│         │                     │             │
│         ▼                     ▼             │
│  ┌──────────────┐   ┌───────────────────┐  │
│  │   Model      │   │   Tier Budget     │  │
│  │  Registry    │   │   (50/800/8000)   │  │
│  └──────┬───────┘   └─────────┬─────────┘  │
│         │                     │             │
│         └──────────┬──────────┘             │
│                    ▼                        │
│            Provider Adapter                 │
│      (OpenAI / Anthropic / Ollama)          │
└─────────────────────────────────────────────┘
         │               │
         ▼               ▼
   Specialist        Generalist
     Model             Model
```

---

## Roadmap

- [x] Heuristic complexity classifier (NO_THINK / SHORT / FULL)
- [x] OpenAI and Anthropic adapters with native reasoning control
- [x] Async support (`achat`, `astream`)
- [x] Retry with exponential backoff
- [x] Thread-safe usage dashboard
- [x] GitHub Actions CI (Python 3.9–3.12)
- [x] **Phase 1: Domain classifier** — 6 domains, 35 pattern groups
- [x] **Phase 1: Model registry** — specialist model selection per domain
- [x] **Phase 1: Ollama adapter** — free local model routing
- [x] **Phase 1: Benchmark tool** — quality measurement vs GPT-4o
- [ ] **Phase 2: Embedding layer** — store (query_embedding, domain, quality_score)
- [ ] **Phase 3: Semantic cache** — nearest-neighbor routing decisions in <2ms
- [ ] **Phase 4: Confidence model** — hallucination risk detection before inference
- [ ] **Phase 5: Atlas API** — query topology atlas as a paid endpoint
- [ ] DistilBERT model on HuggingFace Hub
- [ ] Multi-domain training (MMLU, HumanEval, ARC-Challenge)

---

## Research basis

ThinkRouter is grounded in peer-reviewed research:

- Zhao et al. (2025). *SelfBudgeter: LLM Self-Determine Context Length for Efficient Reasoning*. [arXiv:2505.11274](https://arxiv.org/abs/2505.11274) — 74.47% savings validated
- Wang et al. (2025). *TALE-EP*. ACL Findings 2025 — 67% output token reduction
- Sanh et al. (2019). *DistilBERT*. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
- Cobbe et al. (2021). *GSM8K*. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests welcome.

To add a new domain: extend the pattern lists in `thinkrouter/domain.py` and add a registry entry in `thinkrouter/registry.py`. Open a PR with tests showing correct classification on at least 5 representative queries.

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
<sub>Built by <a href="https://github.com/saikoushiknalubola">Saikoushik Nalubola</a> · Incubated at SRiX, SR University · Backed by MeitY TIDE 2.0</sub>
</div>
