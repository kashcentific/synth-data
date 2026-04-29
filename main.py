# main.py

import json
import os
import sys

from datasets import load_dataset
from graph import build_graph
from config import REPORT_OUTPUT_PATH


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
            "SAFE"                : "✅",
            "USABLE_WITH_CAUTION" : "⚠️ ",
            "UNSAFE"              : "❌",
        }.get(verdict, "❓")

        print(f"\n  {emoji}  VERDICT: {verdict}")
        print(f"\n  Reasoning:")
        for line in evaluator_out.get("verdict_reasoning", "").splitlines():
            print(f"    {line}")

        print(f"\n  Dataset-level Evidence:")
        for line in evaluator_out.get("dataset_level_evidence", "").splitlines():
            print(f"    {line}")

        print(f"\n  Sample-level Inconsistencies:")
        for line in evaluator_out.get("sample_level_inconsistencies", "").splitlines():
            print(f"    {line}")

        print(f"\n  Failure Cases:")
        for line in evaluator_out.get("failure_cases", "").splitlines():
            print(f"    {line}")

        print(f"\n  Statistical Justification:")
        for line in evaluator_out.get("statistical_justification", "").splitlines():
            print(f"    {line}")

        risk = evaluator_out.get("risk_by_metric", [])
        if risk:
            _sub("Risk by Metric")
            for rb in risk:
                lvl  = rb.get("risk_level", "?")
                icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(lvl, "⚪")
                print(f"  │  {icon} [{lvl:<6}] {rb.get('metric', '')}")
                print(f"  │           Finding : {rb.get('finding', '')}")
                print(f"  │           Why     : {rb.get('why', '')}")


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

    try:
        with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  ✔  Full JSON report saved → {os.path.abspath(REPORT_OUTPUT_PATH)}")
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
    _section("SYNAGENT — SYNTHETIC DATA AUDIT SYSTEM", "═")

    print("\n  Loading dataset: Jira Public Dataset (cesaranasco/jira-dataset-public) ...")
    try:
        import kagglehub
        import pandas as pd
        from datasets import Dataset

        csv_path = kagglehub.dataset_download("cesaranasco/jira-dataset-public")
        csv_file = os.path.join(csv_path, "GFG_FINAL.csv")

        df_raw = pd.read_csv(csv_file, low_memory=False)

        # Select the most informative columns for data quality analysis
        keep_cols = [
            "Summary",          # free text — semantic metrics
            "Description",      # free text — semantic metrics
            "Issue Type",       # categorical — class distribution
            "Status",           # categorical — status distribution
            "Priority",         # categorical — severe imbalance expected
            "Resolution",       # categorical — high null rate (~69%)
            "Project key",      # identifier / grouping
            "Project name",     # categorical
            "Project type",     # categorical
            "Reporter",         # categorical
            "Created",          # datetime string
            "Updated",          # datetime string
            "Resolved",         # datetime string — high null rate
        ]
        keep_cols = [c for c in keep_cols if c in df_raw.columns]
        df = df_raw[keep_cols].head(500).reset_index(drop=True)

        # Fill NaN with empty string for object cols so HF Dataset serialises cleanly
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].fillna("")

        ds = Dataset.from_pandas(df, preserve_index=False)
        print(f"  [OK] Loaded {len(ds)} rows x {len(ds.column_names)} columns.")
        print(f"  Columns: {ds.column_names}")
    except Exception as exc:
        print(f"  [ERROR] Failed to load dataset: {exc}")
        sys.exit(1)

    user_hint = (
        "Jira public issue tracker data with ticket summaries, descriptions, "
        "issue types (Bug/Suggestion), statuses, priorities, and resolution info. "
        "Expect class imbalance in Priority and Issue Type. "
        "Check text quality in Summary and Description columns."
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


if __name__ == "__main__":
    main()