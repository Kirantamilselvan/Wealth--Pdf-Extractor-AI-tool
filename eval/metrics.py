"""
Faithfulness / factual consistency / hallucination metrics using an NLI
entailment model (facebook/bart-large-mnli) as the judge: for each sentence
in a generated summary, check whether the source text entails it.

This is the same family of approach as SummaC / FactCC-style factuality
checkers, implemented from scratch here so the eval harness has no
opaque dependencies.
"""
import re
from functools import lru_cache

from transformers import pipeline

NLI_MODEL = "facebook/bart-large-mnli"
ENTAILMENT_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _get_nli_pipeline():
    return pipeline("zero-shot-classification", model=NLI_MODEL)


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def sentence_entailment_score(premise: str, hypothesis: str) -> float:
    """
    Returns P(entailment) of `hypothesis` given `premise`, using the NLI
    model in zero-shot mode: candidate_labels=["entailment","contradiction","neutral"].
    """
    nli = _get_nli_pipeline()
    result = nli(premise[:2000], candidate_labels=["entailment", "contradiction", "neutral"],
                 hypothesis_template=f"This example is consistent with: {hypothesis}")
    scores = dict(zip(result["labels"], result["scores"]))
    return scores.get("entailment", 0.0)


def faithfulness_score(source_text: str, generated_text: str) -> dict:
    """
    For each sentence in generated_text, find the max entailment score against
    the source (checked as a whole, since chunk-level source is usually short
    enough for the NLI model's context window).
    Returns per-sentence scores + an aggregate faithfulness score (mean) and
    hallucination rate (fraction of sentences below ENTAILMENT_THRESHOLD).
    """
    sentences = _split_sentences(generated_text)
    if not sentences:
        return {"faithfulness": 1.0, "hallucination_rate": 0.0, "sentence_scores": []}

    sentence_scores = []
    for sent in sentences:
        score = sentence_entailment_score(source_text, sent)
        sentence_scores.append({"sentence": sent, "entailment_score": round(score, 4)})

    scores = [s["entailment_score"] for s in sentence_scores]
    faithfulness = sum(scores) / len(scores)
    hallucination_rate = sum(1 for s in scores if s < ENTAILMENT_THRESHOLD) / len(scores)

    return {
        "faithfulness": round(faithfulness, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "sentence_scores": sentence_scores,
    }


def factual_consistency_vs_reference(extracted: list[str], reference: list[str]) -> dict:
    """
    Precision/recall of extracted risk-flag sentences against a reference set
    (e.g. the actual Item 1A section sentences). Uses simple substring/fuzzy
    overlap -- swap in a stronger matcher if you want tighter numbers.
    """
    if not extracted or not reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def _norm(s: str) -> str:
        return re.sub(r"\W+", " ", s.lower()).strip()

    ref_norm = [_norm(r) for r in reference]
    matched = 0
    for e in extracted:
        e_norm = _norm(e)
        if any(e_norm[:50] in r or r[:50] in e_norm for r in ref_norm):
            matched += 1

    precision = matched / len(extracted)
    recall = matched / len(reference)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
