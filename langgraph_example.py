"""
langgraph_example.py

Minimal LangGraph that mirrors the real pipeline structure:
  extract_metadata -> thinker -> researcher (with retry loop) -> evaluator -> END

Nodes are lightweight stubs — no LLM calls, no tools.
Researcher loops back twice (simulating coverage gaps) before proceeding.
Runs end-to-end, then calls generate_pipeline_graph to visualise the trace.
"""

import operator
import random
import time
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from visualize import generate_pipeline_graph


# ── State ─────────────────────────────────────────────────────────────

class DemoState(TypedDict, total=False):
    raw_metadata:         Dict[str, Any]
    thinker_output:       Dict[str, Any]
    researcher_output:    Dict[str, Any]
    researcher_iteration: int
    researcher_retry:     bool
    per_metric_results:   List[Dict[str, Any]]
    evaluator_output:     Dict[str, Any]
    errors:               Annotated[List[str], operator.add]


# ── Node stubs ────────────────────────────────────────────────────────

def extract_metadata_node(state: DemoState) -> Dict:
    print("[extract_metadata] scanning dataset...")
    time.sleep(0.3)
    return {
        "raw_metadata": {
            "row_count":      500,
            "col_count":      7,
            "duplicate_pct":  1.2,
            "total_null_pct": 0.4,
        },
        "errors": [],
    }


def thinker_node(state: DemoState) -> Dict:
    print("[thinker] reasoning about domain...")
    time.sleep(0.3)
    return {
        "thinker_output": {
            "domain":       {"name": "demo/synthetic-tabular", "confidence": 0.88},
            "dataset_type": "tabular",
        }
    }


def researcher_node(state: DemoState) -> Dict:
    iteration = state.get("researcher_iteration", 0)
    print(f"[researcher] pass {iteration + 1}...")
    time.sleep(0.4)

    # Passes 0 and 1 → simulate a coverage gap (missing semantic metrics)
    # Pass 2 → coverage satisfied, proceed
    if iteration < 2:
        print(f"[researcher]   coverage gap detected — will retry (iteration {iteration + 1})")
        return {
            "researcher_output": {
                "final_metrics":    [{"metric_name": f"math_metric_{i}"} for i in range(4)],
                "confidence_level": "LOW",
                "research_summary": f"Pass {iteration + 1}: only math metrics found, semantic missing.",
            },
            "researcher_retry":     True,
            "researcher_iteration": iteration + 1,
        }
    else:
        print(f"[researcher]   coverage OK — 10 metrics (math + semantic), proceeding to evaluator")
        return {
            "researcher_output": {
                "final_metrics": (
                    [{"metric_name": f"math_metric_{i}",  "metric_type": "statistical"}   for i in range(5)] +
                    [{"metric_name": f"sem_metric_{i}",   "metric_type": "semantic_consistency"} for i in range(5)]
                ),
                "confidence_level": "HIGH",
                "research_summary": "Pass 3: full coverage — 5 math + 5 semantic metrics found.",
            },
            "researcher_retry":     False,
            "researcher_iteration": iteration + 1,
        }


def evaluator_node(state: DemoState) -> Dict:
    print("[evaluator] computing metrics...")
    time.sleep(0.4)

    strategies  = ["direct_math", "semantic", "custom_function"] * 4
    per_metrics = [
        {
            "metric_name":          f"metric_{i}",
            "quality_score":        random.choice([0.80, 0.85, 0.90, 1.00]),
            "anomalous_row_count":  random.randint(0, 30),
            "computation_strategy": strategies[i % len(strategies)],
        }
        for i in range(9)
    ]

    # One simulated failure
    per_metrics[5]["quality_score"]  = 0.30
    per_metrics[5]["anomalous_row_count"] = 0

    print("[evaluator] done — verdict: ACCEPTABLE_QUALITY")
    return {
        "per_metric_results": per_metrics,
        "evaluator_output":   {"final_verdict": "ACCEPTABLE_QUALITY"},
        "errors":             ["Metric 'metric_5' failed after 3 retries — fallback used"],
    }


# ── Routing function ──────────────────────────────────────────────────

def should_retry_researcher(state: DemoState) -> str:
    needs_retry = state.get("researcher_retry", False)
    iteration   = state.get("researcher_iteration", 0)
    if needs_retry and iteration < 3:
        print(f"[graph] routing back to researcher (iteration {iteration})")
        return "researcher"
    print(f"[graph] routing to evaluator")
    return "evaluator"


# ── Build & run ───────────────────────────────────────────────────────

def main():
    builder = StateGraph(DemoState)

    builder.add_node("extract_metadata", extract_metadata_node)
    builder.add_node("thinker",          thinker_node)
    builder.add_node("researcher",       researcher_node)
    builder.add_node("evaluator",        evaluator_node)

    builder.set_entry_point("extract_metadata")
    builder.add_edge("extract_metadata", "thinker")
    builder.add_edge("thinker",          "researcher")
    builder.add_conditional_edges(
        "researcher",
        should_retry_researcher,
        {"researcher": "researcher", "evaluator": "evaluator"},
    )
    builder.add_edge("evaluator", END)

    app = builder.compile()

    print("\n=== Running demo pipeline ===\n")
    final_state = app.invoke({"errors": []})

    print(f"\n=== Pipeline complete ===")
    print(f"  researcher iterations : {final_state.get('researcher_iteration')}")
    print(f"  metrics computed      : {len(final_state.get('per_metric_results', []))}")
    print(f"  verdict               : {final_state.get('evaluator_output', {}).get('final_verdict')}")
    print(f"  errors                : {final_state.get('errors', [])}")

    print("\n=== Generating pipeline graph ===\n")
    generate_pipeline_graph(final_state, output_path="example_pipeline_graph.png")


if __name__ == "__main__":
    main()
