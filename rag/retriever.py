"""
Retrieval + citation-grounded answer generation.

Retrieval: FAISS top-k over sentence-transformer embeddings.
Generation: BART summarizer repurposed as an extractive-leaning "answer from
context" generator -- kept intentionally simple/swappable. Every answer
returns the chunk_ids it drew from so results are traceable back to source text.
"""
import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from models.summarizer import summarize

load_dotenv()

INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "rag/index"))
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_index = None
_metadata = None
_embedder = None


def _load():
    global _index, _metadata, _embedder
    if _index is None:
        _index = faiss.read_index(str(INDEX_DIR / "chunks.faiss"))
        _metadata = json.loads((INDEX_DIR / "metadata.json").read_text())
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _index, _metadata, _embedder


def retrieve(query: str, k: int = 5) -> list[dict]:
    index, metadata, embedder = _load()
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        chunk = metadata[idx]
        results.append({**chunk, "score": float(score)})
    return results


def answer_query(question: str, k: int = 5) -> dict:
    """
    Retrieve top-k chunks for the question, then produce a grounded answer
    with explicit citations back to chunk_ids.
    """
    hits = retrieve(question, k=k)
    if not hits:
        return {"answer": "No relevant context found in the indexed corpus.", "citations": []}

    context = "\n\n".join(f"[{h['chunk_id']}] {h['text']}" for h in hits)
    prompt_text = f"Question: {question}\n\nRelevant context:\n{context}"

    # Simple approach: summarize the concatenated, question-prefixed context.
    # For a stronger generative QA step, swap this for an instruction-tuned
    # HF model (e.g. a small Flan-T5) prompted with the question + context.
    answer = summarize(prompt_text, max_length=200, min_length=40)

    return {
        "answer": answer,
        "citations": [
            {"chunk_id": h["chunk_id"], "ticker": h["ticker"], "section": h["section"], "score": h["score"]}
            for h in hits
        ],
    }


if __name__ == "__main__":
    result = answer_query("What supply chain risks were disclosed?")
    print(json.dumps(result, indent=2))
