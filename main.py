# main.py

import json
import sys

from datasets import load_dataset
from graph import build_graph


def print_section(title: str, char: str = "─", width: int = 60):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(char * width)


def display_result(state: dict):
    out = state.get("thinker_output", {})

    if not out:
        print("\n[ERROR] No thinker output found.")
        return

    print_section("THINKER OUTPUT", "═")

    dtype = out.get("dataset_type", "unknown")
    domain = out.get("domain", {})
    ref = out.get("reference_dataset", {})
    agents = out.get("recommended_agents", {})

    print(f"\nDataset Type : {dtype}")
    print(
        f"Domain       : {domain.get('name', '?')} "
        f"(confidence {domain.get('confidence', 0):.0%})"
    )
    print(f"Domain Why   : {domain.get('reasoning', '')}")

    print_section("COLUMN PROFILES")

    for cp in out.get("column_profiles", []):
        print(
            f"{cp.get('column_name', ''):<25} "
            f"dtype={cp.get('inferred_dtype', ''):<12} "
            f"role={cp.get('semantic_role', '')}"
        )

        if cp.get("notes"):
            print(f"   ↳ {cp.get('notes')}")

    hints = out.get("surface_quality_hints", [])

    if hints:
        print_section("SURFACE QUALITY HINTS")
        for h in hints:
            print(f"• {h}")

    warnings = out.get("governance_warnings", [])

    if warnings:
        print_section("GOVERNANCE WARNINGS")
        for w in warnings:
            print(f"⚠ {w}")

    print_section("REFERENCE DATASET")
    print(f"Needed : {ref.get('seems_needed', '?')}")
    print(f"Why    : {ref.get('reasoning', '')}")

    print_section("RECOMMENDED AGENTS")

    for name, info in agents.items():
        run_flag = "✓ RUN" if info.get("run") else "✗ SKIP"
        print(f"[{run_flag}] {name:<12} — {info.get('reason', '')}")

    notes = out.get("execution_notes", [])

    if notes:
        print_section("EXECUTION NOTES")
        for n in notes:
            print(f"→ {n}")

    hint_influence = out.get("user_hint_influence", "")

    if hint_influence:
        print_section("USER HINT INFLUENCE")
        print(hint_influence)

    print_section("FULL JSON OUTPUT")
    print(json.dumps(out, indent=2))

    errs = state.get("errors", [])

    if errs:
        print_section("ERRORS", "!")
        for e in errs:
            print(e)


def main():
    print_section("SYNAGENT — THINKER", "═")

    print("\nLoading dataset: ag_news ...")

    try:
        # small + fast dataset for testing
        ds = load_dataset("ag_news", split="train[:500]")
        print("[SUCCESS] Dataset loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {str(e)}")
        sys.exit(1)

    user_hint = input("\nAny input: ").strip()
    user_hint = user_hint if user_hint else None

    app = build_graph()

    initial_state = {
        "dataset": ds,
        "user_hint": user_hint,
        "errors": [],
    }

    final_state = app.invoke(initial_state)

    display_result(final_state)


if __name__ == "__main__":
    main()