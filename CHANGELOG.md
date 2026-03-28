# Changelog

All notable changes to ThinkRouter are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] — 2025-03-28

### Added
- `ThinkRouter` — core routing class with OpenAI and Anthropic adapters
- `HeuristicClassifier` — zero-dependency rule-based difficulty classifier (<1ms)
- `DistilBertClassifier` — fine-tuned DistilBERT backend (requires `thinkrouter[classifier]`)
- `UsageTracker` — thread-safe in-memory savings dashboard
- `RouterResponse` — unified response container across providers
- Three-tier difficulty system: NO_THINK (50 tokens), SHORT (800), FULL (8,000)
- Streaming support via `ThinkRouter.stream()`
- Batch classification via `ThinkRouter.classify_batch()`
- Confidence threshold routing with conservative FULL fallback
- Full test suite (69 tests, 0 failures)
- GitHub Actions CI across Python 3.9–3.12
- MIT License

### Research basis
- Zhao et al. (2025). SelfBudgeter. arXiv:2505.11274 — 74% savings validated
- Wang et al. (2025). TALE-EP. ACL Findings 2025 — 67% token reduction
- Sanh et al. (2019). DistilBERT. arXiv:1910.01108 — classifier backbone
