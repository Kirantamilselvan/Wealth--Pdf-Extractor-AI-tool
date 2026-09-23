"""
Risk-flag extraction using ProsusAI/finbert for sentiment/tone classification,
combined with a lightweight keyword/rule layer to surface candidate risk
sentences from "Item 1A Risk Factors" style text.

FinBERT gives you positive/negative/neutral tone per sentence; the rule layer
narrows to sentences that are actually about risk (not just negative tone).
"""
import re
from functools import lru_cache

from transformers import pipeline

MODEL_NAME = "ProsusAI/finbert"

RISK_KEYWORDS = [
    "risk", "may adversely", "could materially", "uncertain", "disruption",
    "litigation", "regulatory", "volatility", "impairment", "default",
    "non-compliance", "cybersecurity", "breach", "shortage", "competition",
    "liquidity", "downturn", "exposure",
]
RISK_PATTERN = re.compile("|".join(re.escape(k) for k in RISK_KEYWORDS), re.IGNORECASE)


@lru_cache(maxsize=1)
def _get_pipeline():
    return pipeline("text-classification", model=MODEL_NAME)


def _split_sentences(text: str) -> list[str]:
    # simple sentence splitter -- good enough for filing prose
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def extract_risk_flags(text: str, max_flags: int = 20) -> list[dict]:
    """
    Return a list of {sentence, sentiment, score} for sentences that look like
    genuine risk disclosures, ranked by FinBERT negative-sentiment score.
    """
    sentences = _split_sentences(text)
    candidates = [s for s in sentences if RISK_PATTERN.search(s)]
    if not candidates:
        return []

    classifier = _get_pipeline()
    results = classifier(candidates, truncation=True)

    flags = [
        {"sentence": sent, "sentiment": res["label"], "score": round(res["score"], 4)}
        for sent, res in zip(candidates, results)
    ]
    # prioritize negative-tone risk sentences first
    flags.sort(key=lambda f: (f["sentiment"] != "negative", -f["score"]))
    return flags[:max_flags]


def extract_risk_flags_for_document(chunks: list[dict], max_flags: int = 20) -> list[dict]:
    risk_chunks = [c for c in chunks if "risk" in c["section"].lower()] or chunks
    combined_text = " ".join(c["text"] for c in risk_chunks)
    return extract_risk_flags(combined_text, max_flags=max_flags)


if __name__ == "__main__":
    sample = (
        "The Company faces significant risk from supply chain disruption. "
        "Revenue grew 12% year over year. Regulatory changes could materially "
        "impact our operations in the European Union."
    )
    for flag in extract_risk_flags(sample):
        print(flag)
