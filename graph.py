# graph.py

"""
LangGraph definition for the Thinker-only pipeline.

Flow:

START
   ↓
extract_metadata_node   ← deterministic metadata extraction
   ↓
thinker_node            ← OpenAI reasoning via ThinkerAgent
   ↓
END
"""

import pandas as pd
from scipy import stats as scipy_stats
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from agents.thinker import ThinkerAgent
from config import CATEGORICAL_RATIO, SAMPLE_ROW_COUNT
from state import ThinkerState


# ---------------------------------------------------
# Helper: infer datatype
# ---------------------------------------------------

def _infer_col_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numerical"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if series.dtype == object:
        ratio = series.nunique(dropna=True) / max(len(series), 1)

        if ratio < CATEGORICAL_RATIO:
            return "categorical"
        return "text"

    return "unknown"


# ---------------------------------------------------
# Node 1: Metadata Extraction
# ---------------------------------------------------

def extract_metadata_node(state: ThinkerState) -> Dict[str, Any]:
    """
    Uses Hugging Face dataset directly instead of CSV.
    Converts to pandas and extracts deterministic metadata.
    """

    print("\n[METADATA] Processing Hugging Face dataset...")

    ds = state["dataset"]

    # convert small dataset sample to pandas dataframe
    df = ds.to_pandas()

    n_rows, n_cols = df.shape

    columns = []

    for col in df.columns:
        series = df[col]
        dtype = _infer_col_dtype(series)

        col_info = {
            "name": col,
            "inferred_dtype": dtype,
            "null_count": int(series.isnull().sum()),
            "null_pct": round(series.isnull().mean() * 100, 2),
            "unique_count": int(series.nunique(dropna=True)),
            "unique_pct": round(
                series.nunique(dropna=True) / max(n_rows, 1) * 100,
                2
            ),
        }

        # Numerical column stats
        if dtype == "numerical":
            s = series.dropna()

            if not s.empty:
                col_info["stats"] = {
                    "min": round(float(s.min()), 4),
                    "max": round(float(s.max()), 4),
                    "mean": round(float(s.mean()), 4),
                    "std": round(float(s.std()), 4),
                    "skewness": round(float(scipy_stats.skew(s)), 4),
                }

        # Categorical / text top values
        elif dtype in ("categorical", "text"):
            vc = series.value_counts(dropna=True)

            if not vc.empty:
                col_info["top_values"] = vc.head(5).to_dict()

        columns.append(col_info)

    dup_count = int(df.duplicated().sum())

    metadata = {
        "row_count": n_rows,
        "col_count": n_cols,
        "duplicate_rows": dup_count,
        "duplicate_pct": round(
            dup_count / max(n_rows, 1) * 100,
            2
        ),
        "total_null_pct": round(
            df.isnull().mean().mean() * 100,
            2
        ),
        "columns": columns,
        "sample_rows": df.head(SAMPLE_ROW_COUNT)
        .fillna("NULL")
        .to_dict(orient="records"),
    }

    print(
        f"[METADATA] {n_rows} rows × {n_cols} cols | "
        f"{dup_count} duplicates | "
        f"{metadata['total_null_pct']}% null overall"
    )

    return {
        "raw_metadata": metadata,
        "errors": []
    }


# ---------------------------------------------------
# Node 2: Thinker Node
# ---------------------------------------------------

_thinker_agent = ThinkerAgent()


def thinker_node(state: ThinkerState) -> Dict[str, Any]:
    return _thinker_agent.run(state)


# ---------------------------------------------------
# Graph Assembly
# ---------------------------------------------------

def build_graph():
    builder = StateGraph(ThinkerState)

    builder.add_node(
        "extract_metadata",
        extract_metadata_node
    )

    builder.add_node(
        "thinker",
        thinker_node
    )

    builder.set_entry_point("extract_metadata")

    builder.add_edge(
        "extract_metadata",
        "thinker"
    )

    builder.add_edge(
        "thinker",
        END
    )

    return builder.compile()