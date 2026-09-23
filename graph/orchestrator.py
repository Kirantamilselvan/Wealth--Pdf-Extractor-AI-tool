"""
LangGraph orchestration: a router node dispatches to one of three specialist
agents (summarize / extract-risk / answer-query), each backed by the model
wrappers in models/ and rag/.

Usage:
    python graph/orchestrator.py --mode summarize --doc data/processed/AAPL_10-K_2024-11-01.json
    python graph/orchestrator.py --mode risk --doc data/processed/AAPL_10-K_2024-11-01.json
    python graph/orchestrator.py --mode query --question "What supply chain risks did Apple disclose?"
"""
import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

# allow running as a script from repo root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, END

from models.summarizer import summarize_document
from models.risk_classifier import extract_risk_flags_for_document
from rag.retriever import answer_query


class GraphState(TypedDict, total=False):
    mode: str
    doc: dict
    question: str
    result: dict


def route(state: GraphState) -> str:
    return state["mode"]


def summarize_node(state: GraphState) -> GraphState:
    doc = state["doc"]
    summary = summarize_document(doc["chunks"])
    state["result"] = {
        "type": "summary",
        "company": doc.get("company_name"),
        "form": doc.get("form"),
        "summary": summary,
    }
    return state


def risk_node(state: GraphState) -> GraphState:
    doc = state["doc"]
    flags = extract_risk_flags_for_document(doc["chunks"])
    state["result"] = {
        "type": "risk_flags",
        "company": doc.get("company_name"),
        "form": doc.get("form"),
        "risk_flags": flags,
    }
    return state


def query_node(state: GraphState) -> GraphState:
    result = answer_query(state["question"])
    state["result"] = {"type": "query_answer", **result}
    return state


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("summarize", summarize_node)
    graph.add_node("risk", risk_node)
    graph.add_node("query", query_node)

    graph.set_conditional_entry_point(
        route,
        {"summarize": "summarize", "risk": "risk", "query": "query"},
    )
    graph.add_edge("summarize", END)
    graph.add_edge("risk", END)
    graph.add_edge("query", END)
    return graph.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["summarize", "risk", "query"], required=True)
    parser.add_argument("--doc", help="path to a processed document JSON (for summarize/risk)")
    parser.add_argument("--question", help="question text (for query)")
    args = parser.parse_args()

    app = build_graph()
    state: GraphState = {"mode": args.mode}

    if args.mode in ("summarize", "risk"):
        if not args.doc:
            raise SystemExit("--doc is required for summarize/risk modes")
        state["doc"] = json.loads(Path(args.doc).read_text())
    elif args.mode == "query":
        if not args.question:
            raise SystemExit("--question is required for query mode")
        state["question"] = args.question

    final_state = app.invoke(state)
    print(json.dumps(final_state["result"], indent=2))


if __name__ == "__main__":
    main()
