# Wealth Document Intelligence

An end-to-end pipeline that ingests financial filings (SEC 10-K / 10-Q), summarizes them,
extracts risk flags, and answers grounded, cited questions over the corpus using RAG.
Includes an evaluation harness that scores faithfulness, factual consistency, and
hallucination rate — so model quality claims are backed by numbers, not vibes.

## Stack

- **Ingestion**: SEC EDGAR full-text search API
- **Summarization**: Hugging Face `facebook/bart-large-cnn`
- **Risk / sentiment tagging**: Hugging Face `ProsusAI/finbert`
- **Orchestration**: LangGraph multi-agent routing (summarize / extract-risk / answer-query)
- **Retrieval**: FAISS + `sentence-transformers/all-MiniLM-L6-v2` embeddings
- **Evaluation**: NLI-based faithfulness / hallucination scoring (BART-MNLI entailment)
- **Deployment**: AWS (S3 + Lambda/ECS), local Streamlit demo for quick testing
- **Demo UI**: Streamlit

## Repo layout

```
wealth-doc-intelligence/
├── ingestion/          # EDGAR scraping, HTML parsing, section-aware chunking
├── models/             # summarization + risk classification wrappers
├── graph/              # LangGraph orchestration (router + specialist agents)
├── rag/                # FAISS index build + retrieval + citation-grounded QA
├── eval/               # faithfulness/hallucination harness + results
├── infra/              # AWS deployment notes/templates
├── app/                # Streamlit demo frontend
├── data/               # raw + processed filings (gitignored except .gitkeep)
└── notebooks/          # exploration / eval analysis
```

## Quickstart (local)

```bash
git clone <your-repo-url>
cd wealth-doc-intelligence
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # fill in EDGAR user agent + (optional) AWS creds
```

### 1. Pull filings from EDGAR

```bash
python ingestion/edgar_fetch.py --tickers AAPL MSFT JPM --forms 10-K --limit 5
```

### 2. Parse + chunk

```bash
python ingestion/parse_and_chunk.py
```

### 3. Build the FAISS index

```bash
python rag/build_index.py
```

### 4. Run the pipeline via LangGraph (CLI test)

```bash
python graph/orchestrator.py --mode summarize --doc data/processed/AAPL_10-K_2024-11-01.json
python graph/orchestrator.py --mode risk --doc data/processed/AAPL_10-K_2024-11-01.json
python graph/orchestrator.py --mode query --question "What supply chain risks did Apple disclose?"
```

### 5. Launch the demo UI

```bash
streamlit run app/streamlit_app.py
```

### 6. Run the evaluation harness

```bash
python eval/harness.py --n 500
```

Results (faithfulness score and hallucination rate) are written to
`eval/results/eval_report.json` and printed as a summary table.

## AWS deployment

See `infra/README.md` for the S3 + Lambda/ECS deployment path. The pipeline runs
identically locally or on AWS — only the storage/compute backend for the FAISS index
and document store changes (local disk vs. S3-backed).

## Notes on honesty of claims

This repo is a real, runnable scaffold — not a mock. Model choices are off-the-shelf
Hugging Face checkpoints unless you explicitly fine-tune them (see `models/README.md`
for notes on what's baseline vs. fine-tuned). Any performance numbers you cite (e.g.
review-time reduction, hallucination rate) should come from `eval/results/eval_report.json`
on your own corpus, not from this README.
