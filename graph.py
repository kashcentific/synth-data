# graph.py

"""
LangGraph pipeline for the Synthetic Data Audit System.

Correct flow:
  START
    ↓
  extract_metadata_node   ← deterministic column statistics
    ↓
  thinker_node            ← dataset understanding + agent plan
    ↓
  researcher_node         ← metric research via DuckDuckGo/Wikipedia/Arxiv
    ↓
  evaluator_node          ← metric computation via LLM-generated Python + exec()
    ↓
  END
"""

import sys
import os

# Ensure the project root is on sys.path so all agents can do
# `from base import BaseAgent` correctly.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
from scipy import stats as scipy_stats
from typing import Any, Dict

from langgraph.graph import END, StateGraph
from typing import Literal

from agents.thinker import ThinkerAgent
from agents.researcher import ResearchAgent
from agents.evaluator import EvaluatorAgent
from config import CATEGORICAL_RATIO, SAMPLE_ROW_COUNT
from state import ThinkerState


# ──────────────────────────────────────────────────────────────────────
# Helper: infer column data type
# ──────────────────────────────────────────────────────────────────────

def _infer_col_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if series.dtype == object:
        ratio = series.nunique(dropna=True) / max(len(series), 1)
        return "categorical" if ratio < CATEGORICAL_RATIO else "text"
    return "unknown"


# ──────────────────────────────────────────────────────────────────────
# Node 1 — Metadata Extraction (deterministic, no LLM)
# ──────────────────────────────────────────────────────────────────────

def extract_metadata_node(state: ThinkerState) -> Dict[str, Any]:
    """
    Convert the HuggingFace dataset to pandas and extract rich column-level
    statistics that the Thinker and Evaluator will use.
    """
    print("\n[METADATA] Processing dataset...")

    ds = state["dataset"]
    df = ds.to_pandas()
    n_rows, n_cols = df.shape

    columns = []
    for col in df.columns:
        series = df[col]
        dtype  = _infer_col_dtype(series)

        col_info: Dict[str, Any] = {
            "name"         : col,
            "inferred_dtype": dtype,
            "null_count"   : int(series.isnull().sum()),
            "null_pct"     : round(series.isnull().mean() * 100, 2),
            "unique_count" : int(series.nunique(dropna=True)),
            "unique_pct"   : round(
                series.nunique(dropna=True) / max(n_rows, 1) * 100, 2
            ),
        }

        if dtype == "numerical":
            s = series.dropna()
            if not s.empty:
                col_info["stats"] = {
                    "min"     : round(float(s.min()), 4),
                    "max"     : round(float(s.max()), 4),
                    "mean"    : round(float(s.mean()), 4),
                    "std"     : round(float(s.std()), 4),
                    "skewness": round(float(scipy_stats.skew(s)), 4),
                    "kurtosis": round(float(scipy_stats.kurtosis(s)), 4),
                }

        elif dtype in ("categorical", "text"):
            vc = series.value_counts(dropna=True)
            if not vc.empty:
                col_info["top_values"] = vc.head(10).to_dict()
                col_info["value_count"] = len(vc)

        columns.append(col_info)

    dup_count = int(df.duplicated().sum())

    metadata = {
        "row_count"     : n_rows,
        "col_count"     : n_cols,
        "duplicate_rows": dup_count,
        "duplicate_pct" : round(dup_count / max(n_rows, 1) * 100, 2),
        "total_null_pct": round(df.isnull().mean().mean() * 100, 2),
        "columns"       : columns,
        "sample_rows"   : (
            df.head(SAMPLE_ROW_COUNT)
            .fillna("NULL")
            .to_dict(orient="records")
        ),
    }

    print(
        f"[METADATA] {n_rows} rows × {n_cols} cols | "
        f"{dup_count} duplicate rows | "
        f"{metadata['total_null_pct']:.1f}% null overall"
    )

    return {"raw_metadata": metadata, "errors": []}


# ──────────────────────────────────────────────────────────────────────
# Node 2 — Thinker
# ──────────────────────────────────────────────────────────────────────

_thinker_agent = ThinkerAgent()


def thinker_node(state: ThinkerState) -> Dict[str, Any]:
    print("\n[THINKER] Reasoning about dataset domain and quality profile...")
    return _thinker_agent.run(state)


# ──────────────────────────────────────────────────────────────────────
# Node 3 — Researcher
# ──────────────────────────────────────────────────────────────────────

_research_agent = ResearchAgent()


def researcher_node(state: ThinkerState) -> Dict[str, Any]:
    print("\n[RESEARCHER] Gathering external knowledge and proposing metrics...")
    return _research_agent.run(state)


# ──────────────────────────────────────────────────────────────────────
# Node 4 — Evaluator
# ──────────────────────────────────────────────────────────────────────

_evaluator_agent = EvaluatorAgent()


def evaluator_node(state: ThinkerState) -> Dict[str, Any]:
    print("\n[EVALUATOR] Initiating dynamic metric computation engine...")
    return _evaluator_agent.run(state)


# ──────────────────────────────────────────────────────────────────────
# Conditional routing
# ──────────────────────────────────────────────────────────────────────

def should_retry_researcher(state: ThinkerState) -> Literal["researcher", "evaluator"]:
    """
    After the researcher node, check whether it signalled a coverage gap.
    Route back to researcher for a targeted retry (max 2 retries = 3 total passes),
    otherwise proceed to the evaluator.
    """
    needs_retry = state.get("researcher_retry", False)
    iteration   = state.get("researcher_iteration", 0)

    if needs_retry and iteration < 3:
        print(f"\n[GRAPH] Researcher coverage gap detected (iteration {iteration}) — routing back to researcher")
        return "researcher"

    print(f"\n[GRAPH] Researcher output accepted (iteration {iteration}) — routing to evaluator")
    return "evaluator"


# ──────────────────────────────────────────────────────────────────────
# Graph Assembly
# ──────────────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(ThinkerState)

    builder.add_node("extract_metadata", extract_metadata_node)
    builder.add_node("thinker",          thinker_node)
    builder.add_node("researcher",       researcher_node)
    builder.add_node("evaluator",        evaluator_node)

    builder.set_entry_point("extract_metadata")
    builder.add_edge("extract_metadata", "thinker")
    builder.add_edge("thinker",          "researcher")

    # Conditional: researcher loops back on coverage gaps, otherwise goes to evaluator
    builder.add_conditional_edges(
        "researcher",
        should_retry_researcher,
        {"researcher": "researcher", "evaluator": "evaluator"},
    )

    builder.add_edge("evaluator", END)

    return builder.compile()