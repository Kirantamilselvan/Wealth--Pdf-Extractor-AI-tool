"""
Build a FAISS index over all processed document chunks, using
sentence-transformers embeddings. Also writes a parallel metadata file
so retrieved vector IDs can be mapped back to source chunk text + citation info.

Usage:
    python rag/build_index.py
"""
import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
INDEX_DIR = Path(os.getenv("FAISS_INDEX_DIR", "rag/index"))
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_all_chunks() -> list[dict]:
    all_chunks = []
    for doc_path in PROCESSED_DIR.glob("*.json"):
        doc = json.loads(doc_path.read_text())
        for chunk in doc["chunks"]:
            all_chunks.append(
                {
                    **chunk,
                    "ticker": doc["ticker"],
                    "form": doc.get("form"),
                    "filing_date": doc.get("filing_date"),
                    "company_name": doc.get("company_name"),
                    "source_file": doc["source_file"],
                }
            )
    return all_chunks


def build_index(chunks: list[dict]):
    print(f"Loading embedding model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)  # cosine similarity via inner product

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "chunks.faiss"))
    (INDEX_DIR / "metadata.json").write_text(json.dumps(chunks, indent=2))
    print(f"Wrote index with {index.ntotal} vectors to {INDEX_DIR}")


def main():
    chunks = load_all_chunks()
    if not chunks:
        print(f"No processed chunks found under {PROCESSED_DIR}. Run ingestion/parse_and_chunk.py first.")
        return
    build_index(chunks)


if __name__ == "__main__":
    main()
