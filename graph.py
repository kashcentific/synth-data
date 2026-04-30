# graph.py

"""
LangGraph pipeline for the Synthetic Data Audit System.

Correct flow:
  START
    ↓
  extract_metadata_node   ← deterministic column statistics
    ↓
  universal_metrics_node  ← predefined text quality metrics (no LLM, always first)
    ↓
  thinker_node            ← dataset understanding + agent plan
    ↓
  researcher_node         ← metric research via DuckDuckGo/Wikipedia/Arxiv
    ↓
  evaluator_node          ← research metric computation via LLM-generated Python + exec()
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

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Any, Dict

from langgraph.graph import END, StateGraph
from typing import Literal

from agents.thinker import ThinkerAgent
from agents.researcher import ResearchAgent
from agents.evaluator import EvaluatorAgent
from agents.universal_text_metrics import UniversalTextMetrics
from config import CATEGORICAL_RATIO, SAMPLE_ROW_COUNT, MAX_ROWS_FOR_EVAL
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
    # Accept object dtype AND Arrow/StringDtype string columns.
    # datasets>=3.x + pyarrow>=14 returns string columns as ArrowDtype(large_string),
    # not plain object, so checking dtype==object alone silently misclassifies them.
    dtype_str = str(series.dtype).lower()
    is_stringlike = (
        series.dtype == object
        or "string" in dtype_str
        or "large_string" in dtype_str
        or "utf8" in dtype_str
    )
    if is_stringlike:
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

    # datasets>=3.x with pyarrow>=14 returns Arrow-backed extension types (e.g.
    # ArrowDtype(large_string)).  Convert each column to its closest numpy/object
    # equivalent so that downstream pandas operations (isnull, duplicated, etc.)
    # work correctly across all environments.
    try:
        df = df.convert_dtypes(dtype_backend="numpy_nullable")
    except Exception:
        pass

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
                # to_numpy() ensures scipy works even with nullable Int64/Float64 types
                s_np = s.to_numpy(dtype=float, na_value=float("nan"))
                s_np = s_np[~np.isnan(s_np)]
                if len(s_np):
                    col_info["stats"] = {
                        "min"     : round(float(s_np.min()), 4),
                        "max"     : round(float(s_np.max()), 4),
                        "mean"    : round(float(s_np.mean()), 4),
                        "std"     : round(float(s_np.std()), 4),
                        "skewness": round(float(scipy_stats.skew(s_np)), 4),
                        "kurtosis": round(float(scipy_stats.kurtosis(s_np)), 4),
                    }

        elif dtype == "categorical":
            vc = series.value_counts(dropna=True)
            if not vc.empty:
                # Truncate long categorical values so they don't bloat the prompt
                col_info["top_values"] = {
                    str(k)[:120]: int(v)
                    for k, v in vc.head(10).items()
                }
                col_info["value_count"] = len(vc)

        elif dtype == "text":
            # Never dump full text values — record length stats and a short sample only
            lengths = series.dropna().astype(str).str.len()
            col_info["avg_length"]  = round(float(lengths.mean()), 1) if not lengths.empty else 0
            col_info["max_length"]  = int(lengths.max())              if not lengths.empty else 0
            first_val = series.dropna().astype(str).iloc[0] if not series.dropna().empty else ""
            col_info["sample_text"] = first_val[:200] + ("..." if len(first_val) > 200 else "")

        columns.append(col_info)

    # df.duplicated() can raise on Arrow-backed extension types; convert to plain
    # object dtype so the comparison works regardless of HuggingFace Dataset version.
    try:
        dup_count = int(df.duplicated().sum())
    except Exception:
        try:
            dup_count = int(df.astype(str).duplicated().sum())
        except Exception:
            dup_count = 0

    # Truncate long string values in sample rows so the thinker prompt stays
    # within the model's context window (trajectory fields can be 70k+ chars).
    def _trim_row(row: dict, limit: int = 300) -> dict:
        out = {}
        for k, v in row.items():
            # Normalise to Python str (handles Arrow StringScalar, numpy str_, etc.)
            s = v if isinstance(v, str) else (str(v) if not isinstance(v, (int, float, bool, list, dict, bytes, type(None))) else v)
            if isinstance(s, str) and len(s) > limit:
                out[k] = s[:limit] + f"…[+{len(s)-limit}]"
            else:
                out[k] = s
        return out

    metadata = {
        "row_count"     : n_rows,
        "col_count"     : n_cols,
        "duplicate_rows": dup_count,
        "duplicate_pct" : round(dup_count / max(n_rows, 1) * 100, 2),
        "total_null_pct": round(df.isnull().mean().mean() * 100, 2),
        "columns"       : columns,
        "sample_rows"   : [
            _trim_row(r)
            for r in df.head(SAMPLE_ROW_COUNT).fillna("NULL").to_dict(orient="records")
        ],
    }

    print(
        f"[METADATA] {n_rows} rows × {n_cols} cols | "
        f"{dup_count} duplicate rows | "
        f"{metadata['total_null_pct']:.1f}% null overall"
    )

    return {"raw_metadata": metadata, "errors": []}


# ──────────────────────────────────────────────────────────────────────
# Node 2 — Universal Text Metrics (deterministic, runs before LLM agents)
# ──────────────────────────────────────────────────────────────────────

_utm = UniversalTextMetrics()


def _detect_text_cols_graph(df: pd.DataFrame):
    """Same dtype-agnostic detection used by the evaluator."""
    text_cols = []
    for col in df.columns:
        s = df[col]
        if (pd.api.types.is_numeric_dtype(s)
                or pd.api.types.is_bool_dtype(s)
                or pd.api.types.is_datetime64_any_dtype(s)):
            continue
        try:
            avg_len = s.dropna().astype(str).str.len().mean()
            if pd.notna(avg_len) and float(avg_len) > 15:
                text_cols.append(col)
        except Exception:
            pass
    return text_cols


def universal_metrics_node(state: ThinkerState) -> Dict[str, Any]:
    """
    Compute universal text quality metrics immediately after metadata extraction,
    before any LLM agent runs. Results are stored in state so the evaluator can
    merge them in without re-running.
    """
    print("\n[UNIVERSAL] Computing universal text metrics (pre-LLM)...")

    ds  = state["dataset"]
    df  = ds.to_pandas()

    try:
        df = df.convert_dtypes(dtype_backend="numpy_nullable")
    except Exception:
        pass

    if len(df) > MAX_ROWS_FOR_EVAL:
        df = df.sample(MAX_ROWS_FOR_EVAL, random_state=42).reset_index(drop=True)

    text_cols = _detect_text_cols_graph(df)
    if not text_cols:
        print("[UNIVERSAL] No text columns detected — universal metrics skipped.")
        return {"universal_metric_results": [], "errors": []}

    print(f"[UNIVERSAL] Text columns: {text_cols}")
    results = _utm.compute_all(df, text_cols, thinker_output={})

    ok      = sum(1 for r in results if not r.get("error"))
    skipped = len(results) - ok
    print(f"[UNIVERSAL] {ok} computed, {skipped} skipped.")
    return {"universal_metric_results": results, "errors": []}


# ──────────────────────────────────────────────────────────────────────
# Node 3 — Thinker
# ──────────────────────────────────────────────────────────────────────

_thinker_agent = ThinkerAgent()


def thinker_node(state: ThinkerState) -> Dict[str, Any]:
    print("\n[THINKER] Reasoning about dataset domain and quality profile...")
    return _thinker_agent.run(state)


# ──────────────────────────────────────────────────────────────────────
# Node 4 — Researcher
# ──────────────────────────────────────────────────────────────────────

_research_agent = ResearchAgent()


def researcher_node(state: ThinkerState) -> Dict[str, Any]:
    print("\n[RESEARCHER] Gathering external knowledge and proposing metrics...")
    return _research_agent.run(state)


# ──────────────────────────────────────────────────────────────────────
# Node 5 — Evaluator
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

    builder.add_node("extract_metadata",   extract_metadata_node)
    builder.add_node("universal_metrics",  universal_metrics_node)
    builder.add_node("thinker",            thinker_node)
    builder.add_node("researcher",         researcher_node)
    builder.add_node("evaluator",          evaluator_node)

    # Flow: metadata → universal metrics → thinker → researcher → evaluator
    builder.set_entry_point("extract_metadata")
    builder.add_edge("extract_metadata",  "universal_metrics")
    builder.add_edge("universal_metrics", "thinker")
    builder.add_edge("thinker",           "researcher")

    # Conditional: researcher loops back on coverage gaps, otherwise goes to evaluator
    builder.add_conditional_edges(
        "researcher",
        should_retry_researcher,
        {"researcher": "researcher", "evaluator": "evaluator"},
    )

    builder.add_edge("evaluator", END)

    return builder.compile()