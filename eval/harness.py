"""
Evaluation harness: runs summarization + risk extraction across N chunks from
the processed corpus, scores faithfulness/hallucination for summaries, and
writes an aggregate report to eval/results/eval_report.json.

Usage:
    python eval/harness.py --n 500
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.summarizer import summarize
from eval.metrics import faithfulness_score

PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_eval_chunks(n: int) -> list[dict]:
    chunks = []
    for doc_path in sorted(PROCESSED_DIR.glob("*.json")):
        doc = json.loads(doc_path.read_text())
        for chunk in doc["chunks"]:
            if len(chunk["text"]) > 200:  # skip trivially short chunks
                chunks.append(chunk)
            if len(chunks) >= n:
                return chunks
    return chunks


def run_eval(n: int) -> dict:
    chunks = load_eval_chunks(n)
    if not chunks:
        raise SystemExit(
            f"No processed chunks found under {PROCESSED_DIR}. "
            "Run ingestion/edgar_fetch.py and ingestion/parse_and_chunk.py first."
        )

    print(f"Evaluating {len(chunks)} chunks...")
    per_chunk_results = []
    start = time.time()

    for i, chunk in enumerate(chunks):
        summary = summarize(chunk["text"])
        scores = faithfulness_score(chunk["text"], summary)
        per_chunk_results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "section": chunk["section"],
                "faithfulness": scores["faithfulness"],
                "hallucination_rate": scores["hallucination_rate"],
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(chunks)} evaluated...")

    elapsed = time.time() - start
    avg_faithfulness = sum(r["faithfulness"] for r in per_chunk_results) / len(per_chunk_results)
    avg_hallucination = sum(r["hallucination_rate"] for r in per_chunk_results) / len(per_chunk_results)

    report = {
        "num_chunks_evaluated": len(per_chunk_results),
        "avg_faithfulness": round(avg_faithfulness, 4),
        "avg_hallucination_rate": round(avg_hallucination, 4),
        "eval_runtime_seconds": round(elapsed, 1),
        "per_chunk_results": per_chunk_results,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="number of chunks to evaluate")
    args = parser.parse_args()

    report = run_eval(args.n)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print("\n=== Evaluation summary ===")
    print(f"Chunks evaluated:      {report['num_chunks_evaluated']}")
    print(f"Avg faithfulness:      {report['avg_faithfulness']}")
    print(f"Avg hallucination rate:{report['avg_hallucination_rate']}")
    print(f"Runtime:               {report['eval_runtime_seconds']}s")
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
