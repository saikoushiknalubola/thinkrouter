"""
thinkrouter.registry
~~~~~~~~~~~~~~~~~~~~
Model registry — maps (Domain, provider_preference) to the best
available model for that domain.

The registry is the source of truth for domain routing decisions.
It is consulted after domain classification and before the API call.

Usage::

    from thinkrouter.registry import ModelRegistry
    from thinkrouter.domain import Domain

    reg    = ModelRegistry()
    target = reg.resolve(Domain.CODE, preferred_provider="ollama")
    # ModelTarget(model="deepseek-coder-v2", provider="ollama", ...)

Customisation::

    reg = ModelRegistry()
    reg.register(Domain.CODE, "ollama", "codellama:13b", quality_score=0.85)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .domain import Domain


# ── Model target ───────────────────────────────────────────────────────────

@dataclass
class ModelTarget:
    """
    A resolved routing target after domain classification.

    Attributes
    ----------
    model           : Model identifier to use for the API call.
    provider        : Provider name: "openai", "anthropic", "ollama", "huggingface".
    domain          : Domain that triggered this routing decision.
    quality_score   : Estimated quality score on domain benchmarks [0, 1].
    cost_relative   : Relative cost vs GPT-4o (1.0 = same, 0.05 = 20x cheaper).
    notes           : Human-readable note on why this model was selected.
    """
    model:          str
    provider:       str
    domain:         Domain
    quality_score:  float
    cost_relative:  float
    notes:          str = ""

    def __repr__(self) -> str:
        return (
            f"ModelTarget("
            f"model={self.model!r}, "
            f"provider={self.provider!r}, "
            f"domain={self.domain.value}, "
            f"quality={self.quality_score:.2f}, "
            f"cost={self.cost_relative:.2f}x)"
        )


# ── Default registry entries ───────────────────────────────────────────────

# Structure: domain → list of entries ordered by quality score (desc)
# Each entry: (provider, model, quality_score, cost_relative, notes)
_DEFAULT_ENTRIES: Dict[Domain, List[tuple]] = {

    Domain.CODE: [
        ("ollama",      "deepseek-coder-v2",                   0.91, 0.00,
         "Best open-source code model, free via Ollama. Beats GPT-4o on HumanEval."),
        ("ollama",      "codellama:13b",                       0.84, 0.00,
         "Meta CodeLlama 13B — good fallback if deepseek-coder unavailable."),
        ("huggingface", "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", 0.89, 0.02,
         "HuggingFace hosted version — requires thinkrouter[classifier]."),
        ("openai",      "gpt-4o",                              0.82, 1.00,
         "Generalist fallback — lower code accuracy than specialists."),
        ("anthropic",   "claude-sonnet-4-6",                   0.84, 0.80,
         "Strong at code explanation and review, good fallback."),
    ],

    Domain.MATH: [
        ("ollama",      "qwen2.5-math",                        0.88, 0.00,
         "Qwen2.5-Math via Ollama — highest open-source MATH benchmark score."),
        ("ollama",      "llama3.1",                            0.74, 0.00,
         "Llama 3.1 — reasonable math, good free fallback."),
        ("huggingface", "Qwen/Qwen2.5-Math-7B-Instruct",      0.90, 0.02,
         "HuggingFace hosted Qwen Math — best accuracy for pure math."),
        ("openai",      "gpt-4o",                              0.78, 1.00,
         "Generalist fallback — decent but not specialist-level."),
        ("anthropic",   "claude-opus-4-6",                     0.82, 2.00,
         "Claude Opus — strong at mathematical reasoning and proofs."),
    ],

    Domain.MEDICAL: [
        ("ollama",      "medllama2",                           0.80, 0.00,
         "MedLLaMA2 via Ollama — fine-tuned on medical literature."),
        ("huggingface", "axiong/PMC_LLaMA_13B",               0.82, 0.02,
         "PMC-LLaMA trained on PubMed Central biomedical papers."),
        ("openai",      "gpt-4o",                              0.71, 1.00,
         "GPT-4o generalist — adequate but not specialist-level."),
        ("anthropic",   "claude-opus-4-6",                     0.78, 2.00,
         "Claude Opus — best generalist for sensitive medical queries."),
    ],

    Domain.LEGAL: [
        ("ollama",      "llama3.1",                            0.76, 0.00,
         "Llama 3.1 8B — best free option for legal text understanding."),
        ("huggingface", "AdaptLLM/law-LLM",                   0.82, 0.02,
         "Law-LLM fine-tuned on legal corpora and bar exam questions."),
        ("openai",      "gpt-4o",                              0.74, 1.00,
         "GPT-4o — strong legal reasoning as a generalist."),
        ("anthropic",   "claude-opus-4-6",                     0.80, 2.00,
         "Claude Opus — excellent at long-form legal document analysis."),
    ],

    Domain.FINANCIAL: [
        ("ollama",      "llama3.1",                            0.72, 0.00,
         "Llama 3.1 — decent financial reasoning, best free option."),
        ("huggingface", "TheFinAI/finma-7b-full",             0.80, 0.02,
         "FinMA fine-tuned on financial instruction datasets."),
        ("openai",      "gpt-4o",                              0.76, 1.00,
         "GPT-4o — strong financial analysis as a generalist."),
        ("anthropic",   "claude-sonnet-4-6",                   0.78, 0.80,
         "Claude Sonnet — good for financial modeling and analysis."),
    ],

    Domain.GENERAL: [
        ("ollama",      "llama3.1",                            0.74, 0.00,
         "Llama 3.1 8B — best open-source general-purpose model."),
        ("openai",      "gpt-4o",                              0.89, 1.00,
         "GPT-4o — best generalist for mixed or unclear domain."),
        ("anthropic",   "claude-sonnet-4-6",                   0.87, 0.80,
         "Claude Sonnet — strong generalist, good for complex queries."),
    ],
}


# ── Registry class ─────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Registry of domain → model mappings.

    Resolves the best available model for a given domain and provider
    preference. Falls back gracefully when a preferred provider has
    no entry for the requested domain.

    Parameters
    ----------
    provider_priority : list[str]
        Ordered list of provider preferences. The first provider that
        has an entry for the requested domain wins.
        Default: ["ollama", "openai", "anthropic", "huggingface"]
    """

    def __init__(
        self,
        provider_priority: Optional[List[str]] = None,
    ) -> None:
        # Deep copy so mutations don't affect the module-level defaults.
        self._entries: Dict[Domain, List[tuple]] = {
            domain: list(entries)
            for domain, entries in _DEFAULT_ENTRIES.items()
        }
        self._priority = provider_priority or ["ollama", "openai", "anthropic", "huggingface"]

    def resolve(
        self,
        domain:             Domain,
        preferred_provider: Optional[str] = None,
    ) -> ModelTarget:
        """
        Return the best ModelTarget for a given domain.

        Parameters
        ----------
        domain             : Detected query domain.
        preferred_provider : Override the registry's provider priority for
                             this one call (e.g. "openai" if Ollama is down).

        Returns
        -------
        ModelTarget
        """
        entries = self._entries.get(domain, self._entries[Domain.GENERAL])

        # Build priority list for this call
        if preferred_provider:
            priority = [preferred_provider] + [p for p in self._priority if p != preferred_provider]
        else:
            priority = self._priority

        # Find best entry matching provider priority
        for provider in priority:
            for prov, model, quality, cost, notes in entries:
                if prov == provider:
                    return ModelTarget(
                        model=model, provider=provider,
                        domain=domain, quality_score=quality,
                        cost_relative=cost, notes=notes,
                    )

        # Absolute fallback: first entry regardless of provider
        prov, model, quality, cost, notes = entries[0]
        return ModelTarget(
            model=model, provider=prov,
            domain=domain, quality_score=quality,
            cost_relative=cost, notes=notes,
        )

    def best_for(self, domain: Domain) -> List[ModelTarget]:
        """
        Return all registered models for a domain, sorted by quality score.
        """
        entries = self._entries.get(domain, [])
        targets = [
            ModelTarget(model=m, provider=p, domain=domain,
                        quality_score=q, cost_relative=c, notes=n)
            for p, m, q, c, n in entries
        ]
        return sorted(targets, key=lambda t: t.quality_score, reverse=True)

    def register(
        self,
        domain:         Domain,
        provider:       str,
        model:          str,
        quality_score:  float = 0.75,
        cost_relative:  float = 1.0,
        notes:          str   = "",
    ) -> None:
        """
        Add or update a model entry in the registry.

        The new entry is inserted at position 0 (highest priority) for
        its provider.

        Parameters
        ----------
        domain         : Domain this model specialises in.
        provider       : Provider name.
        model          : Model identifier.
        quality_score  : Estimated quality on domain benchmarks [0, 1].
        cost_relative  : Cost relative to GPT-4o (0.05 = 20x cheaper).
        notes          : Human-readable note.
        """
        entry = (provider, model, quality_score, cost_relative, notes)
        if domain not in self._entries:
            self._entries[domain] = []
        # Remove existing entry for same provider+model if present
        self._entries[domain] = [
            e for e in self._entries[domain]
            if not (e[0] == provider and e[1] == model)
        ]
        self._entries[domain].insert(0, entry)

    def summary(self) -> str:
        """Print a formatted summary of all registry entries."""
        lines = ["\n  ThinkRouter — Model Registry", "  " + "─" * 52]
        for domain in Domain:
            entries = self._entries.get(domain, [])
            lines.append(f"\n  {domain.value.upper()}")
            for prov, model, quality, cost, _ in entries:
                cost_str = "free" if cost == 0.0 else f"{cost:.1f}x"
                lines.append(f"    [{prov:<12}] {model:<42} q={quality:.2f}  cost={cost_str}")
        lines.append("")
        return "\n".join(lines)


# ── Module-level default instance ─────────────────────────────────────────

DEFAULT_REGISTRY = ModelRegistry()
