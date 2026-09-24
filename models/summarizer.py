"""
Summarization wrapper around Hugging Face facebook/bart-large-cnn.

This is the off-the-shelf baseline. If you fine-tune on financial filings later,
point MODEL_NAME at your fine-tuned checkpoint and note that explicitly in your
resume/portfolio writeup -- don't let this default silently become a claim of
fine-tuning you didn't do.
"""
from functools import lru_cache
from typing import Optional

from transformers import pipeline

MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_CHARS = 4000  # BART has a ~1024 token limit; keep inputs conservative


@lru_cache(maxsize=1)
def _get_pipeline():
    return pipeline("summarization", model=MODEL_NAME)


def summarize(text: str, max_length: int = 180, min_length: int = 60) -> str:
    """Summarize a single chunk or section of text."""
    text = text[:MAX_INPUT_CHARS]
    if len(text.strip()) < 50:
        return text.strip()

    summarizer = _get_pipeline()
    result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return result[0]["summary_text"]


def summarize_document(chunks: list[dict], focus_section: Optional[str] = None) -> str:
    """
    Summarize a document by summarizing each relevant chunk, then summarizing
    the concatenation of chunk summaries (map-reduce style).
    """
    relevant = [c for c in chunks if focus_section is None or focus_section in c["section"]]
    if not relevant:
        relevant = chunks

    chunk_summaries = [summarize(c["text"]) for c in relevant]
    combined = " ".join(chunk_summaries)
    if len(combined) > MAX_INPUT_CHARS:
        return summarize(combined)
    return combined if len(chunk_summaries) <= 1 else summarize(combined)


if __name__ == "__main__":
    sample = (
        "The Company faces risks related to supply chain disruptions, foreign currency "
        "fluctuations, and increasing competition in the smartphone market. Component "
        "shortages in fiscal 2024 led to production delays for several product lines."
    )
    print(summarize(sample))
