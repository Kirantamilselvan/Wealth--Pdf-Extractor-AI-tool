"""
Streamlit demo UI for the Wealth Document Intelligence pipeline.

Run with:
    streamlit run app/streamlit_app.py
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv()

from models.summarizer import summarize_document
from models.risk_classifier import extract_risk_flags_for_document
from rag.retriever import answer_query

PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))

st.set_page_config(page_title="Wealth Document Intelligence", layout="wide")
st.title("Wealth Document Intelligence")
st.caption("Summarization, risk-flag extraction, and cited RAG querying over SEC filings")

doc_files = sorted(PROCESSED_DIR.glob("*.json")) if PROCESSED_DIR.exists() else []

tab_summary, tab_risk, tab_query = st.tabs(["Summarize a filing", "Risk flags", "Ask a question"])

with tab_summary:
    if not doc_files:
        st.info("No processed filings found. Run the ingestion + parsing scripts first.")
    else:
        selected = st.selectbox("Choose a filing", doc_files, format_func=lambda p: p.stem, key="sum_select")
        if st.button("Summarize", key="sum_btn"):
            doc = json.loads(selected.read_text())
            with st.spinner("Summarizing..."):
                summary = summarize_document(doc["chunks"])
            st.subheader(f"{doc.get('company_name', doc['ticker'])} — {doc.get('form')}")
            st.write(summary)

with tab_risk:
    if not doc_files:
        st.info("No processed filings found. Run the ingestion + parsing scripts first.")
    else:
        selected = st.selectbox("Choose a filing", doc_files, format_func=lambda p: p.stem, key="risk_select")
        if st.button("Extract risk flags", key="risk_btn"):
            doc = json.loads(selected.read_text())
            with st.spinner("Classifying risk sentences..."):
                flags = extract_risk_flags_for_document(doc["chunks"])
            if not flags:
                st.write("No risk-flagged sentences found.")
            for f in flags:
                st.markdown(f"- **[{f['sentiment']} · {f['score']}]** {f['sentence']}")

with tab_query:
    question = st.text_input("Ask a question across the indexed corpus")
    if st.button("Ask", key="query_btn") and question:
        with st.spinner("Retrieving and answering..."):
            try:
                result = answer_query(question)
            except FileNotFoundError:
                st.error("FAISS index not found. Run rag/build_index.py first.")
                result = None
        if result:
            st.subheader("Answer")
            st.write(result["answer"])
            st.subheader("Citations")
            for c in result["citations"]:
                st.markdown(f"- `{c['chunk_id']}` ({c['ticker']}, {c['section']}, score={c['score']:.3f})")
