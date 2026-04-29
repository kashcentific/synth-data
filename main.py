# main.py

import io
import json
import os
import sys

from datasets import load_dataset
from graph import build_graph
from config import REPORT_OUTPUT_PATH
from visualize import generate_pipeline_graph

BENCHMARK_REPORT_PATH   = "benchmarking_results.json"
BENCHMARK_WORKFLOW_PATH = "benchmarking_workflow.log"


# ──────────────────────────────────────────────────────────────────────
# Tee logger — mirrors everything printed to console into a log file
# ──────────────────────────────────────────────────────────────────────

class _Tee:
    """Writes to both the real stdout and a log file simultaneously."""
    def __init__(self, stream, log_path: str):
        self._stream  = stream
        self._logfile = open(log_path, "w", encoding="utf-8", errors="replace")

    def write(self, data):
        self._stream.write(data)
        self._logfile.write(data)

    def flush(self):
        self._stream.flush()
        self._logfile.flush()

    def close(self):
        self._logfile.close()

    # Delegate everything else (isatty, fileno, etc.) to the real stream
    def __getattr__(self, name):
        return getattr(self._stream, name)


# ──────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────

def _bar(char: str = "─", width: int = 70) -> str:
    return char * width


def _section(title: str, char: str = "─"):
    print(f"\n{_bar(char)}")
    print(f"  {title}")
    print(_bar(char))


def _sub(title: str):
    print(f"\n  ┌─ {title}")


def _row(label: str, value: str, indent: int = 4):
    pad = " " * indent
    # Wrap long values
    if len(value) > 100:
        print(f"{pad}{label}:")
        for line in value.splitlines():
            print(f"{pad}  {line}")
    else:
        print(f"{pad}{label}: {value}")


# ──────────────────────────────────────────────────────────────────────
# Display: thinker
# ──────────────────────────────────────────────────────────────────────

def _display_thinker(out: dict):
    if not out:
        print("\n[!] No thinker output.")
        return

    _section("① THINKER — DATASET UNDERSTANDING", "═")

    domain = out.get("domain", {})
    print(f"\n  Dataset Type : {out.get('dataset_type', '?')}")
    print(
        f"  Domain       : {domain.get('name', '?')} "
        f"(confidence {domain.get('confidence', 0):.0%})"
    )
    print(f"  Domain Why   : {domain.get('reasoning', '')}")

    _sub("Column Profiles")
    for cp in out.get("column_profiles", []):
        print(
            f"  │  {cp.get('column_name', ''):<25} "
            f"dtype={cp.get('inferred_dtype', ''):<12} "
            f"role={cp.get('semantic_role', '')}"
        )
        if cp.get("notes"):
            print(f"  │    ↳ {cp['notes']}")

    hints = out.get("surface_quality_hints", [])
    if hints:
        _sub("Surface Quality Hints")
        for h in hints:
            print(f"  │  • {h}")

    warnings = out.get("governance_warnings", [])
    if warnings:
        _sub("Governance Warnings")
        for w in warnings:
            print(f"  │  ⚠  {w}")

    ref = out.get("reference_dataset", {})
    _sub("Reference Dataset Assessment")
    print(f"  │  Needed : {ref.get('seems_needed', '?')}")
    print(f"  │  Why    : {ref.get('reasoning', '')}")

    _sub("Recommended Downstream Agents")
    for name, info in out.get("recommended_agents", {}).items():
        flag = "✓ RUN " if info.get("run") else "✗ SKIP"
        print(f"  │  [{flag}] {name:<12} — {info.get('reason', '')}")

    notes = out.get("execution_notes", [])
    if notes:
        _sub("Execution Notes for Downstream Agents")
        for n in notes:
            print(f"  │  → {n}")


# ──────────────────────────────────────────────────────────────────────
# Display: researcher
# ──────────────────────────────────────────────────────────────────────

def _display_researcher(out: dict):
    if not out:
        print("\n[!] No researcher output.")
        return

    _section("② RESEARCHER — EXTERNAL RESEARCH & METRIC PROPOSALS", "═")

    # Show what each tool actually returned
    context = out.get("research_context", {})
    if context:
        _sub("Research Tool Results")
        tool_icons = {
            "duckduckgo": "🌐 DuckDuckGo Web Search",
            "wikipedia" : "📖 Wikipedia",
            "arxiv"     : "📄 Arxiv Papers",
        }
        for key, icon in tool_icons.items():
            src = context.get(key, {})
            if src:
                print(f"  │")
                print(f"  │  {icon}")
                print(f"  │    Query  : \"{src.get('query', '')}\"")
                excerpt = src.get("excerpt", "").replace("\n", " ").strip()
                if excerpt:
                    print(f"  │    Found  : {excerpt[:200]}")
                else:
                    print(f"  │    Found  : (no results)")

    # Research summary
    summary = out.get("research_summary", "")
    if summary:
        _sub("Research Summary")
        print(f"  │  {summary}")

    # Proposed metrics with source traceability
    metrics = out.get("proposed_metrics", [])
    if metrics:
        _sub(f"Proposed Metrics ({len(metrics)} total)")
        for i, m in enumerate(metrics, 1):
            print(f"\n  │  {i:>2}. {m.get('metric_name')} [{m.get('metric_type')}]")
            print(f"  │      Description    : {m.get('description', '')}")
            print(f"  │      Reasoning      : {m.get('reasoning', '')}")
            src_inf = m.get("source_influence", "")
            if src_inf:
                print(f"  │      Source/Papers  : {src_inf}")
            print(f"  │      Execution Hint : {m.get('execution_hint', '')}")


# ──────────────────────────────────────────────────────────────────────
# Display: evaluator
# ──────────────────────────────────────────────────────────────────────

def _display_evaluator(evaluator_out: dict, per_metric: list):
    if not evaluator_out and not per_metric:
        print("\n[!] No evaluator output.")
        return

    _section("③ EVALUATOR — METRIC COMPUTATION & FINDINGS", "═")

    for idx, r in enumerate(per_metric, 1):
        name   = r.get("metric_name", "?")
        err    = r.get("error")
        acount = r.get("anomalous_row_count", 0)
        status = "✗ ERROR" if err else "✓ COMPUTED"

        print(f"\n  ┌─ [{idx}] {name}  [{status}]")

        # Why this metric was selected
        sel_reason = r.get("selection_reason", "")
        if sel_reason:
            print(f"  │  Why selected : {sel_reason}")

        # Which research source drove this metric
        src_inf = r.get("source_influence", "")
        if src_inf:
            print(f"  │  Research source : {src_inf}")

        # Execution output (the actual numbers)
        exec_out = (r.get("execution_output") or "").strip()
        if exec_out:
            print(f"  │  Execution output:")
            for line in exec_out.splitlines():
                print(f"  │    {line}")

        if err:
            print(f"  │  ⚠  Error : {err}")

        # Row-level findings
        print(f"  │  Anomalous rows : {acount}")
        narratives = r.get("anomalous_row_narratives", [])
        if narratives:
            print(f"  │  Flagged rows:")
            for n in narratives[:8]:
                print(f"  │    ⚠  {n}")

        # LLM interpretation
        interp = r.get("metric_interpretation", "")
        if interp:
            print(f"  │  Interpretation:")
            for line in interp.strip().splitlines():
                print(f"  │    {line}")

        print(f"  └{'─' * 66}")

    # Final verdict
    if evaluator_out:
        _section("③ EVALUATOR — FINAL VERDICT", "═")

        verdict = evaluator_out.get("final_verdict", "UNKNOWN")
        emoji   = {
            "HIGH_QUALITY"       : "✅",
            "ACCEPTABLE_QUALITY" : "⚠️ ",
            "POOR_QUALITY"       : "❌",
            # legacy fallbacks
            "SAFE"               : "✅",
            "USABLE_WITH_CAUTION": "⚠️ ",
            "UNSAFE"             : "❌",
        }.get(verdict, "❓")

        print(f"\n  {emoji}  DATA QUALITY VERDICT: {verdict}")
        print(f"\n  Reasoning:")
        for line in evaluator_out.get("verdict_reasoning", "").splitlines():
            print(f"    {line}")

        print(f"\n  Dataset-level Evidence:")
        for line in evaluator_out.get("dataset_level_evidence", "").splitlines():
            print(f"    {line}")

        obs = evaluator_out.get("quality_observations", evaluator_out.get("sample_level_inconsistencies", ""))
        if obs:
            print(f"\n  Quality Observations:")
            for line in obs.splitlines():
                print(f"    {line}")

        print(f"\n  Statistical Justification:")
        for line in evaluator_out.get("statistical_justification", "").splitlines():
            print(f"    {line}")

        quality = evaluator_out.get("quality_by_metric", evaluator_out.get("risk_by_metric", []))
        if quality:
            _sub("Quality by Metric")
            for rb in quality:
                lvl  = rb.get("quality_level", rb.get("risk_level", "?"))
                icon = {"GOOD": "🟢", "ACCEPTABLE": "🟡", "POOR": "🔴",
                        "LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(lvl, "⚪")
                print(f"  │  {icon} [{lvl:<10}] {rb.get('metric', '')}")
                print(f"  │             Finding : {rb.get('finding', '')}")
                note = rb.get("note", rb.get("why", ""))
                if note:
                    print(f"  │             Note    : {note}")


# ──────────────────────────────────────────────────────────────────────
# Save JSON report
# ──────────────────────────────────────────────────────────────────────

def _save_report(state: dict):
    report = {
        "thinker_output"    : state.get("thinker_output", {}),
        "researcher_output" : state.get("researcher_output", {}),
        "per_metric_results": state.get("per_metric_results", []),
        "evaluator_output"  : state.get("evaluator_output", {}),
        "errors"            : state.get("errors", []),
    }
    # Strip generated code from JSON report (it's verbose; kept on console)
    for r in report.get("per_metric_results", []):
        r.pop("generated_code", None)

    out_path = BENCHMARK_REPORT_PATH
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  [OK] Structured report saved  → {os.path.abspath(out_path)}")
    except Exception as exc:
        print(f"\n  [WARN] Could not save report: {exc}")


# ──────────────────────────────────────────────────────────────────────
# Master display
# ──────────────────────────────────────────────────────────────────────

def display_result(state: dict):
    _display_thinker(state.get("thinker_output", {}))
    _display_researcher(state.get("researcher_output", {}))
    _display_evaluator(
        state.get("evaluator_output", {}),
        state.get("per_metric_results", []),
    )

    errs = state.get("errors", [])
    if errs:
        _section("PIPELINE ERRORS", "!")
        for e in errs:
            print(f"  [ERR] {e}")

    _section("REPORT", "═")
    _save_report(state)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────


def main():
    # ── Start Tee logger before anything is printed ──────────────────
    tee = _Tee(sys.stdout, BENCHMARK_WORKFLOW_PATH)
    sys.stdout = tee

    try:
        _run_pipeline()
    finally:
        sys.stdout = tee._stream
        tee.close()


def _run_pipeline():
    _section("SYNAGENT — SYNTHETIC DATA AUDIT SYSTEM  [BENCHMARKING RUN]", "═")
    print(f"\n  Report   → {BENCHMARK_REPORT_PATH}")
    print(f"  Workflow → {BENCHMARK_WORKFLOW_PATH}")

    print("\n  Loading dataset: Solaris99/AgentBank (apps) ...")
    try:
        import pandas as pd
        from datasets import Dataset, load_dataset as _load

        raw_ds = _load("Solaris99/AgentBank", "apps", split="train[:500]")

        rows = []
        for item in raw_ds:
            convs    = item.get("conversations", [])
            problem  = next((c["value"] for c in convs if c.get("from") == "human"), "")
            solution = next((c["value"] for c in convs if c.get("from") == "gpt"),   "")
            rows.append({
                "id":                  item["id"],
                "problem":             problem,
                "solution":            solution,
                "has_thought_process": int(solution.strip().startswith("Thought:")),
                "problem_length":      len(problem),
                "solution_length":     len(solution),
                "num_turns":           len(convs),
            })

        df = pd.DataFrame(rows)
        ds = Dataset.from_pandas(df, preserve_index=False)
        print(f"  [OK] Loaded {len(ds)} rows x {len(ds.column_names)} columns.")
        print(f"  Columns : {ds.column_names}")
        print(f"  Avg problem length  : {df.problem_length.mean():.0f} chars")
        print(f"  Avg solution length : {df.solution_length.mean():.0f} chars")
        print(f"  Has thought process : {df.has_thought_process.sum()} / {len(df)} rows")
    except Exception as exc:
        print(f"  [ERROR] Failed to load dataset: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    user_hint = (
        "AgentBank APPS benchmark — agent-generated solutions to competitive programming problems. "
        "Each row contains a problem statement (problem) and the agent's step-by-step solution (solution). "
        "Check semantic consistency of solutions with their problems, solution length distribution, "
        "whether thought-process reasoning is present, and text quality across both columns."
    )

    app = build_graph()

    initial_state = {
        "dataset"  : ds,
        "user_hint": user_hint,
        "errors"   : [],
    }

    print("\n  Starting pipeline...\n")
    final_state = app.invoke(initial_state)

    display_result(final_state)

    # Auto-generate pipeline execution graph
    graph_path = os.path.splitext(BENCHMARK_REPORT_PATH)[0] + "_pipeline_graph.png"
    try:
        generate_pipeline_graph(final_state, output_path=graph_path)
    except Exception as exc:
        print(f"\n  [WARN] Could not generate pipeline graph: {exc}")

    print(f"\n  [OK] Workflow log saved       → {os.path.abspath(BENCHMARK_WORKFLOW_PATH)}")


if __name__ == "__main__":
    main()
