"""
thinkrouter.domain
~~~~~~~~~~~~~~~~~~
Domain classifier — Phase 1 of the semantic routing roadmap.

Detects the subject domain of a query so ThinkRouter can route to
the best specialist model rather than always using a generalist.

Domains: CODE · MATH · MEDICAL · LEGAL · FINANCIAL · GENERAL

Usage::

    from thinkrouter.domain import DomainClassifier, Domain

    clf = DomainClassifier()
    clf.predict("Write a binary search function in Python.")
    # DomainResult(domain=CODE, confidence=0.85, latency=0.3ms)

    clf.predict("Prove that sqrt(2) is irrational.")
    # DomainResult(domain=MATH, confidence=0.82, latency=0.2ms)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


# ── Domain enum ────────────────────────────────────────────────────────────

class Domain(str, Enum):
    CODE      = "code"
    MATH      = "math"
    MEDICAL   = "medical"
    LEGAL     = "legal"
    FINANCIAL = "financial"
    GENERAL   = "general"


DOMAIN_DESCRIPTIONS: Dict[Domain, str] = {
    Domain.CODE:      "Code generation, debugging, algorithms, software engineering",
    Domain.MATH:      "Mathematical proofs, equations, statistics, numerical reasoning",
    Domain.MEDICAL:   "Clinical medicine, pharmacology, anatomy, diagnostics, health",
    Domain.LEGAL:     "Contracts, case law, compliance, regulatory analysis",
    Domain.FINANCIAL: "Markets, accounting, valuation, trading, financial analysis",
    Domain.GENERAL:   "General knowledge, mixed domain, or undetermined",
}

DOMAIN_DEFAULT_MODELS: Dict[Domain, Dict[str, str]] = {
    Domain.CODE: {
        "ollama":      "deepseek-coder-v2",
        "huggingface": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "openai":      "gpt-4o",
        "anthropic":   "claude-sonnet-4-6",
    },
    Domain.MATH: {
        "ollama":      "qwen2.5-math",
        "huggingface": "Qwen/Qwen2.5-Math-7B-Instruct",
        "openai":      "gpt-4o",
        "anthropic":   "claude-opus-4-6",
    },
    Domain.MEDICAL: {
        "ollama":      "medllama2",
        "huggingface": "axiong/PMC_LLaMA_13B",
        "openai":      "gpt-4o",
        "anthropic":   "claude-opus-4-6",
    },
    Domain.LEGAL: {
        "ollama":      "llama3.1",
        "huggingface": "AdaptLLM/law-LLM",
        "openai":      "gpt-4o",
        "anthropic":   "claude-opus-4-6",
    },
    Domain.FINANCIAL: {
        "ollama":      "llama3.1",
        "huggingface": "TheFinAI/finma-7b-full",
        "openai":      "gpt-4o",
        "anthropic":   "claude-sonnet-4-6",
    },
    Domain.GENERAL: {
        "ollama":      "llama3.1",
        "huggingface": "meta-llama/Llama-3.1-8B-Instruct",
        "openai":      "gpt-4o",
        "anthropic":   "claude-sonnet-4-6",
    },
}


# ── Result container ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class DomainResult:
    """
    Output of a single domain classification call.

    Attributes
    ----------
    domain      : Detected subject domain.
    confidence  : Classifier confidence in [0, 1].
    latency_ms  : Wall-clock classification time.
    signals     : Pattern names that fired (for debugging).
    backend     : Classifier backend used.
    """
    domain:     Domain
    confidence: float
    latency_ms: float
    signals:    Tuple[str, ...]
    backend:    str

    def __repr__(self) -> str:
        return (
            f"DomainResult("
            f"domain={self.domain.value}, "
            f"confidence={self.confidence:.3f}, "
            f"latency={self.latency_ms:.2f}ms)"
        )


# ── Patterns ────────────────────────────────────────────────────────────────
# Design principle: broad > narrow.
# A domain classifier should be inclusive — one clear signal is enough.
# Precision is the job of the downstream specialist model, not the classifier.

_F = re.I  # shorthand

# ── CODE ──────────────────────────────────────────────────────────────────
_CODE: List[Tuple[str, re.Pattern]] = [
    # Intent verbs: write / implement / build / create + code artifact
    ("write_code",
     re.compile(r"\b(write|create|build|make)\b.{0,50}\b(function|class|method|program|script|module|library|api|endpoint|service|component|widget)\b", _F)),

    # Implement keyword (common in CS)
    ("implement",
     re.compile(r"\bimplement\b", _F)),

    # Explicit programming language mentions
    ("language",
     re.compile(r"\b(python|javascript|typescript|java|golang|rust|c\+\+|c#|ruby|swift|kotlin|scala|php|bash|shell|powershell|sql|html|css|react|vue|django|flask|fastapi|nodejs|numpy|pandas)\b", _F)),

    # Core CS / programming keywords
    ("cs_concept",
     re.compile(r"\b(algorithm|recursion|iteration|loop|array|linked\s+list|stack|queue|heap|tree|graph|hash\s*map|pointer|memory|thread|async|coroutine|mutex|semaphore|deadlock)\b", _F)),

    # Debugging / refactoring
    ("debug",
     re.compile(r"\b(debug|fix|refactor|optimize|profile|benchmark)\b.{0,40}\b(code|function|script|program|bug|error|crash|exception|memory\s+leak|race\s+condition|deadlock)\b", _F)),

    # SQL / database queries
    ("database",
     re.compile(r"\b(sql\s+(query|statement)|select\s+.{0,20}\s+from|join\s+(three|two|multiple)\s+table|stored\s+procedure|orm|migration|index\s+(on|for))\b", _F)),

    # Git / DevOps
    ("devops",
     re.compile(r"\b(git\s+(commit|push|pull|merge|rebase)|docker|kubernetes|ci\/cd|github\s+actions|unit\s+test|pytest|jest|junit)\b", _F)),

    # "In Python/JavaScript/etc" at end of sentence
    ("lang_in",
     re.compile(r"\bin\s+(python|javascript|typescript|java|go|rust|c\+\+|c#|ruby|swift|kotlin|scala|sql)\b", _F)),
]

# ── MATH ──────────────────────────────────────────────────────────────────
_MATH: List[Tuple[str, re.Pattern]] = [
    # Prove / derive / show that
    ("prove",
     re.compile(r"\b(prove|proof|disprove|show\s+that|demonstrate\s+that|verify\s+that|derive)\b", _F)),

    # Calculus
    ("calculus",
     re.compile(r"\b(derivative|integral|differentiate|integrate|limit\s+(of|as)|gradient|divergence|curl|laplacian|taylor\s+series|fourier)\b", _F)),

    # Linear algebra
    ("linear_algebra",
     re.compile(r"\b(matrix|matrices|eigenvalue|eigenvector|determinant|vector\s+space|dot\s+product|cross\s+product|svd|singular\s+value|orthogonal)\b", _F)),

    # Statistics & probability
    ("stats",
     re.compile(r"\b(probability|expected\s+value|variance|standard\s+deviation|regression|hypothesis\s+test|p.value|confidence\s+interval|bayesian|markov|distribution|random\s+variable)\b", _F)),

    # Number theory / discrete math
    ("number_theory",
     re.compile(r"\b(prime\s+(number|factorization)|fibonacci|factorial|combinatorics|permutation|combination|modular|gcd|lcm|induction|pigeonhole)\b", _F)),

    # Equation / formula / theorem
    ("theorem",
     re.compile(r"\b(theorem|lemma|corollary|axiom|equation|formula|series|sequence|convergence|divergence)\b", _F)),

    # Explicit math nouns
    ("math_noun",
     re.compile(r"\b(irrational|rational|integer|real\s+number|complex\s+number|polynomial|quadratic|cubic|logarithm|exponential|trigonometric|sine|cosine|tangent)\b", _F)),
]

# ── MEDICAL ───────────────────────────────────────────────────────────────
_MEDICAL: List[Tuple[str, re.Pattern]] = [
    # Clinical actions
    ("clinical",
     re.compile(r"\b(diagnos|treatment|symptom|prognosi|patholog|clinical|therapeut)\w*\b", _F)),

    # Pharmacology
    ("pharma",
     re.compile(r"\b(drug|medication|pharmacolog|mechanism\s+of\s+action|dosage|dose|pharmacokinetic|side\s+effect|contraindication|prescription|antibiotic|vaccine|antiviral|chemotherapy)\b", _F)),

    # Drug names (common ones)
    ("drug_names",
     re.compile(r"\b(metformin|insulin|aspirin|ibuprofen|acetaminophen|paracetamol|lisinopril|atorvastatin|amoxicillin|penicillin|warfarin|heparin|morphine|metoprolol|omeprazole)\b", _F)),

    # Anatomy
    ("anatomy",
     re.compile(r"\b(anatomical|organ|tissue|neuron|artery|vein|cardiac|pulmonary|hepatic|renal|cerebral|cortex|neurotransmitter|receptor|cell\s+membrane|dna|rna|protein\s+synthesis)\b", _F)),

    # Procedures
    ("procedures",
     re.compile(r"\b(surgery|surgical|biopsy|endoscopy|mri|ct\s+scan|x.ray|ultrasound|ecg|eeg|blood\s+test|lumbar\s+puncture|colonoscopy)\b", _F)),

    # Disease / condition names
    ("diseases",
     re.compile(r"\b(hypertension|diabetes|cancer|tumor|carcinoma|inflammation|autoimmune|stroke|myocardial|sepsis|pneumonia|alzheimer|parkinson|epilepsy|asthma|copd|hiv|hepatitis)\b", _F)),

    # Medical specialties
    ("specialty",
     re.compile(r"\b(oncolog|cardiolog|neurolog|pediatric|psychiatr|orthopedic|dermatolog|radiolog|anesthesiolog|endocrinolog|gastroenterolog)\w*\b", _F)),
]

# ── LEGAL ─────────────────────────────────────────────────────────────────
_LEGAL: List[Tuple[str, re.Pattern]] = [
    # Core legal terms
    ("legal_core",
     re.compile(r"\b(contract|agreement|clause|provision|liability|warranty|breach|jurisdiction|arbitration|indemnif)\w*\b", _F)),

    # Court / litigation
    ("court",
     re.compile(r"\b(lawsuit|litigation|plaintiff|defendant|court|judge|verdict|appeal|precedent|case\s+law|common\s+law|tort|negligence|damages|injunction)\b", _F)),

    # Statute / regulation
    ("regulation",
     re.compile(r"\b(statute|regulation|compliance|regulatory|gdpr|ccpa|hipaa|ferpa|sec|ftc|cfpb|enforceable|lawful|unlawful|legal\s+(requirement|obligation|right|duty))\b", _F)),

    # IP
    ("ip",
     re.compile(r"\b(patent|trademark|copyright|intellectual\s+property|trade\s+secret|licensing|fair\s+use|infringement)\b", _F)),

    # Corporate law
    ("corporate",
     re.compile(r"\b(due\s+diligence|fiduciary|shareholder|corporate\s+governance|sec\s+filing|ipo|merger\s+(and\s+)?acquisition|non.compete|non.disclosure|nda)\b", _F)),

    # Legal question phrasing
    ("legal_question",
     re.compile(r"\b(is\s+it\s+(legal|illegal)|legally\s+(required|allowed|binding|enforceable)|what\s+does\s+the\s+law|under\s+(the\s+law|gdpr|ccpa|hipaa)|elements\s+of\s+a)\b", _F)),

    # Legal terminology
    ("legal_terms",
     re.compile(r"\b(attorney|lawyer|counsel|subpoena|deposition|affidavit|habeas|mens\s+rea|actus\s+reus|promissory\s+note|easement|lien|mortgage\s+deed)\b", _F)),
]

# ── FINANCIAL ─────────────────────────────────────────────────────────────
_FINANCIAL: List[Tuple[str, re.Pattern]] = [
    # Markets
    ("market",
     re.compile(r"\b(stock|equity|bond|option|futures|derivative|portfolio|hedge\s+fund|etf|mutual\s+fund|volatility|beta|alpha|short\s+sell|market\s+cap)\b", _F)),

    # Valuation / analysis
    ("valuation",
     re.compile(r"\b(valuation|dcf|discounted\s+cash\s+flow|p\/e\s+ratio|ebitda|earnings|revenue|profit\s+margin|return\s+on\s+(equity|assets|investment)|npv|irr)\b", _F)),

    # Banking / credit
    ("banking",
     re.compile(r"\b(interest\s+rate|credit\s+(rating|score|risk)|loan|mortgage|yield\s+(curve)?|bond\s+price|duration|central\s+bank|monetary\s+policy|quantitative\s+easing)\b", _F)),

    # Crypto
    ("crypto",
     re.compile(r"\b(cryptocurrency|bitcoin|ethereum|blockchain|defi|smart\s+contract|nft|tokenomics|staking|crypto\s+(wallet|exchange))\b", _F)),

    # Accounting
    ("accounting",
     re.compile(r"\b(accounting|gaap|ifrs|balance\s+sheet|income\s+statement|cash\s+flow|depreciation|amortization|accrual|accounts\s+(receivable|payable)|working\s+capital|audit)\b", _F)),

    # Financial terms
    ("fin_terms",
     re.compile(r"\b(financial\s+(model|analysis|statement|forecast|projection|ratio)|capital\s+(structure|allocation|expenditure)|leverage|liquidity|solvency)\b", _F)),
]

# Domain → (pattern list, base_confidence_per_hit)
# Higher base_confidence = more aggressive classification
# 0.50 means: 1 hit → 50% confidence (above the 0.25 min_confidence threshold)
_DOMAIN_PATTERNS: Dict[Domain, Tuple[List[Tuple[str, re.Pattern]], float]] = {
    Domain.CODE:      (_CODE,      0.50),
    Domain.MATH:      (_MATH,      0.50),
    Domain.MEDICAL:   (_MEDICAL,   0.50),
    Domain.LEGAL:     (_LEGAL,     0.50),
    Domain.FINANCIAL: (_FINANCIAL, 0.50),
}


# ── Classifier ─────────────────────────────────────────────────────────────

class DomainClassifier:
    """
    Heuristic domain classifier using curated regex pattern libraries.

    Runs in under 1ms. Zero external dependencies.

    Parameters
    ----------
    min_confidence : float
        Minimum score to assign a non-GENERAL domain.
        Default: 0.25 — one clear domain signal is enough.
    """

    def __init__(self, min_confidence: float = 0.25) -> None:
        self.min_confidence = min_confidence

    def predict(self, query: str) -> DomainResult:
        t0 = time.perf_counter()
        q  = query.strip()

        scores:  Dict[Domain, float]     = {}
        signals: Dict[Domain, List[str]] = {}

        for domain, (patterns, base_conf) in _DOMAIN_PATTERNS.items():
            hits = []
            for name, pat in patterns:
                if pat.search(q):
                    hits.append(name)
            if hits:
                # First hit: base_conf. Each additional hit adds 0.08 (diminishing).
                conf = min(base_conf + 0.08 * (len(hits) - 1), 0.97)
                scores[domain]  = conf
                signals[domain] = hits

        ms = (time.perf_counter() - t0) * 1000

        if not scores:
            return DomainResult(
                domain=Domain.GENERAL, confidence=0.80,
                latency_ms=ms, signals=(), backend="heuristic",
            )

        best_domain = max(scores, key=scores.__getitem__)
        best_score  = scores[best_domain]

        # Reduce confidence when two domains are ambiguous (gap < 0.08)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < 0.08:
            best_score = max(best_score - 0.10, self.min_confidence)

        if best_score < self.min_confidence:
            return DomainResult(
                domain=Domain.GENERAL, confidence=0.75,
                latency_ms=ms, signals=(), backend="heuristic",
            )

        return DomainResult(
            domain=best_domain,
            confidence=round(best_score, 4),
            latency_ms=ms,
            signals=tuple(signals.get(best_domain, [])),
            backend="heuristic",
        )

    def predict_batch(self, queries: List[str]) -> List[DomainResult]:
        return [self.predict(q) for q in queries]

    def scores(self, query: str) -> Dict[Domain, float]:
        """Return confidence scores for all domains — useful for debugging."""
        q = query.strip()
        result: Dict[Domain, float] = {}
        for domain, (patterns, base_conf) in _DOMAIN_PATTERNS.items():
            hits = sum(1 for _, pat in patterns if pat.search(q))
            if hits:
                result[domain] = min(base_conf + 0.08 * (hits - 1), 0.97)
        if not result:
            result[Domain.GENERAL] = 0.80
        elif Domain.GENERAL not in result:
            result[Domain.GENERAL] = max(0.20, 0.80 - max(result.values()))
        return result
