"""
visualize.py — Pipeline execution trace graph renderer.

Builds the graph from what actually ran:
  - Each node execution becomes a box in the trace
  - Researcher appears N times if it looped (unrolled execution)
  - Arrows are labelled with the routing decision that caused each transition
  - Colour-coded by node type and outcome
"""

import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Palette ───────────────────────────────────────────────────────────
_BG          = "#0d1117"
_COL = {
    "start_end":  ("#1a2a1a", "#3a7a3a"),   # (fill, border)
    "extract":    ("#0d2035", "#2a6aaa"),
    "thinker":    ("#1a1535", "#6a50cc"),
    "researcher": ("#0d2530", "#2a9aaa"),
    "researcher_retry": ("#2a1a00", "#cc8800"),
    "evaluator_high":   ("#0a2a0a", "#30cc50"),
    "evaluator_ok":     ("#2a2000", "#ccaa00"),
    "evaluator_poor":   ("#2a0a0a", "#cc3030"),
}
_EDGE_NORMAL  = "#3a6aaa"
_EDGE_RETRY   = "#cc8800"
_EDGE_OK      = "#30cc50"
_TEXT_TITLE   = "#e8eeff"
_TEXT_SUB     = "#7a9acc"
_TEXT_LABEL   = "#aabbcc"

_VERDICT_LABEL = {
    "HIGH_QUALITY":        "HIGH QUALITY",
    "ACCEPTABLE_QUALITY":  "ACCEPTABLE QUALITY",
    "POOR_QUALITY":        "POOR QUALITY",
    "SAFE":                "SAFE",
    "USABLE_WITH_CAUTION": "USABLE WITH CAUTION",
    "UNSAFE":              "UNSAFE",
}

# ── Step data class ───────────────────────────────────────────────────

class _Step:
    def __init__(self, node_id: str, title: str, lines: List[str],
                 fill: str, border: str, edge_label: str = "", edge_color: str = _EDGE_NORMAL):
        self.node_id    = node_id
        self.title      = title
        self.lines      = lines
        self.fill       = fill
        self.border     = border
        self.edge_label = edge_label      # label on the arrow INTO this step
        self.edge_color = edge_color


# ── Build trace from state ────────────────────────────────────────────

def _build_trace(state: Dict[str, Any]) -> List[_Step]:
    raw         = state.get("raw_metadata", {})
    thinker     = state.get("thinker_output", {})
    researcher  = state.get("researcher_output", {})
    evaluator   = state.get("evaluator_output", {})
    per_metrics = state.get("per_metric_results", [])
    iteration   = state.get("researcher_iteration", 1)
    verdict     = evaluator.get("final_verdict", "UNKNOWN")

    # ── evaluator colour key ─────────────────────────────────────────
    ev_key = {
        "HIGH_QUALITY": "evaluator_high", "SAFE": "evaluator_high",
        "ACCEPTABLE_QUALITY": "evaluator_ok", "USABLE_WITH_CAUTION": "evaluator_ok",
        "POOR_QUALITY": "evaluator_poor", "UNSAFE": "evaluator_poor",
    }.get(verdict, "evaluator_ok")

    steps: List[_Step] = []

    # START
    steps.append(_Step(
        "start", "START", [],
        *_COL["start_end"], edge_label="", edge_color=_EDGE_NORMAL
    ))

    # extract_metadata
    steps.append(_Step(
        "extract", "extract_metadata",
        [
            f"rows : {raw.get('row_count','?')}   cols : {raw.get('col_count','?')}",
            f"nulls: {raw.get('total_null_pct',0):.1f}%   dups : {raw.get('duplicate_pct',0):.1f}%",
        ],
        *_COL["extract"], edge_label="dataset loaded", edge_color=_EDGE_NORMAL
    ))

    # thinker
    domain     = thinker.get("domain", {}).get("name", "unknown")
    ds_type    = thinker.get("dataset_type", "?")
    conf_raw   = thinker.get("domain", {}).get("confidence", 0)
    conf_str   = f"{conf_raw:.0%}" if isinstance(conf_raw, float) else str(conf_raw)
    steps.append(_Step(
        "thinker", "thinker",
        [
            f"domain : {domain[:28]}",
            f"type   : {ds_type}",
            f"conf   : {conf_str}",
        ],
        *_COL["thinker"], edge_label="metadata ready", edge_color=_EDGE_NORMAL
    ))

    # Researcher — unrolled for every iteration
    final_metrics = researcher.get("final_metrics", researcher.get("proposed_metrics", []))
    n_final       = len(final_metrics)
    res_conf      = researcher.get("confidence_level", "?")

    for i in range(1, iteration + 1):
        is_last        = (i == iteration)
        is_retry       = (i > 1)
        still_retrying = (not is_last)

        if is_retry:
            fill, border = _COL["researcher_retry"]
            in_label     = f"coverage gap — retry {i-1}"
            in_color     = _EDGE_RETRY
        else:
            fill, border = _COL["researcher"]
            in_label     = "thinker output"
            in_color     = _EDGE_NORMAL

        if is_last:
            lines = [
                f"pass {i} of {iteration}",
                f"metrics   : {n_final}",
                f"confidence: {res_conf}",
                "coverage: OK  ->  proceed" if not still_retrying else "coverage: GAP",
            ]
        else:
            lines = [
                f"pass {i} of {iteration}",
                "metrics insufficient",
                "missing: math or semantic",
                "decision: RETRY",
            ]

        steps.append(_Step(
            f"researcher_{i}",
            f"researcher  [pass {i}/{iteration}]",
            lines,
            fill, border,
            edge_label=in_label,
            edge_color=in_color,
        ))

    # evaluator
    n_computed = len(per_metrics)
    avg_qual   = sum(r.get("quality_score", 0) for r in per_metrics) / max(n_computed, 1)
    strats: Dict[str, int] = {}
    for r in per_metrics:
        s = r.get("computation_strategy", "custom")
        short = {"direct_math": "math", "custom_function": "custom", "semantic": "sem"}.get(s, s[:6])
        strats[short] = strats.get(short, 0) + 1
    strat_str = "  ".join(f"{k}:{v}" for k, v in strats.items())

    ev_fill, ev_border = _COL[ev_key]
    steps.append(_Step(
        "evaluator", "evaluator",
        [
            f"computed : {n_computed} metrics",
            f"avg qual : {avg_qual:.2f}",
            f"strategy : {strat_str}" if strat_str else "",
            f"verdict  : {_VERDICT_LABEL.get(verdict, verdict)}",
        ],
        ev_fill, ev_border,
        edge_label="coverage OK  ->  evaluate",
        edge_color=_EDGE_OK,
    ))

    # END
    steps.append(_Step(
        "end", "END", [],
        *_COL["start_end"],
        edge_label="done", edge_color=_EDGE_OK
    ))

    return steps


# ── Drawing helpers ───────────────────────────────────────────────────

_NW, _NH = 5.4, 1.8   # node width, height
_GAP     = 0.55        # vertical gap between nodes
_X       = 0           # all nodes centred at x=0


def _draw_node(ax, cx: float, cy: float, step: _Step, is_small: bool = False):
    w = 2.0 if is_small else _NW
    h = 0.65 if is_small else _NH

    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.05",
        linewidth=2.2,
        edgecolor=step.border,
        facecolor=step.fill,
        zorder=3,
    )
    ax.add_patch(box)

    if is_small:
        ax.text(cx, cy, step.title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=step.border,
                fontfamily="monospace", zorder=4)
        return

    # Title bar
    title_y = cy + h * 0.26
    ax.text(cx, title_y, step.title,
            ha="center", va="center",
            fontsize=9.5, fontweight="bold",
            color=step.border, fontfamily="monospace", zorder=4)

    # Divider
    ax.plot([cx - w * 0.42, cx + w * 0.42],
            [cy + h * 0.06, cy + h * 0.06],
            color=step.border, linewidth=0.9, alpha=0.55, zorder=4)

    # Info lines
    lh    = h * 0.155
    start = cy - h * 0.08
    for i, line in enumerate([l for l in step.lines if l][:4]):
        ax.text(cx, start - i * lh, line,
                ha="center", va="center",
                fontsize=7.3, color=_TEXT_SUB,
                fontfamily="monospace", zorder=4)


def _draw_edge(ax, x: float, y_from: float, y_to: float,
               label: str, color: str, is_small_from: bool, is_small_to: bool):
    bot_from = y_from - (0.65 / 2 if is_small_from else _NH / 2)
    top_to   = y_to   + (0.65 / 2 if is_small_to   else _NH / 2)

    ax.annotate("",
        xy=(x, top_to + 0.01), xytext=(x, bot_from - 0.01),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=2.0,
            mutation_scale=14,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=2,
    )
    if label:
        mid_y = (bot_from + top_to) / 2
        ax.text(x + _NW * 0.52, mid_y, label,
                ha="left", va="center",
                fontsize=7.5, color=color,
                style="italic", fontfamily="monospace", zorder=5,
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor=_BG, edgecolor=color,
                          linewidth=0.9, alpha=0.85))


# ── Main entry point ─────────────────────────────────────────────────

def generate_pipeline_graph(final_state: Dict[str, Any],
                             output_path: str = "pipeline_graph.png"):
    steps  = _build_trace(final_state)
    errors = final_state.get("errors", [])
    verdict = final_state.get("evaluator_output", {}).get("final_verdict", "?")
    domain  = final_state.get("thinker_output", {}).get("domain", {}).get("name", "?")

    # ── Compute layout ───────────────────────────────────────────────
    # Each step occupies (NH + GAP) of vertical space; START/END are small
    def _height(s: _Step):
        return 0.65 if s.node_id in ("start", "end") else _NH

    # Assign y-centres top-to-bottom
    positions = []       # (cx, cy, is_small)
    y = 0.0
    for s in steps:
        h = _height(s)
        positions.append((_X, y, s.node_id in ("start", "end")))
        y -= h + _GAP

    total_h = abs(y) + 0.5
    canvas_h = max(total_h + 2.0, 8.0)
    canvas_w = 13.0

    fig, ax = plt.subplots(figsize=(canvas_w, canvas_h))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    x_pad = (_NW / 2) + 3.5
    ax.set_xlim(-x_pad, x_pad)
    ax.set_ylim(y - 1.2, 1.8)
    ax.axis("off")

    # ── Title ────────────────────────────────────────────────────────
    ax.text(0, 1.4, "DATA QUALITY PIPELINE  —  EXECUTION TRACE",
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color=_TEXT_TITLE, fontfamily="monospace")
    ax.text(0, 0.9,
            f"domain: {domain}   |   verdict: {_VERDICT_LABEL.get(verdict, verdict)}",
            ha="center", va="center",
            fontsize=8, color=_TEXT_SUB, fontfamily="monospace")

    # ── Draw nodes ───────────────────────────────────────────────────
    for step, (cx, cy, is_small) in zip(steps, positions):
        _draw_node(ax, cx, cy, step, is_small)

    # ── Draw edges ───────────────────────────────────────────────────
    for i in range(1, len(steps)):
        _, y_from, small_from = positions[i - 1]
        _, y_to,   small_to   = positions[i]
        step = steps[i]
        _draw_edge(ax, _X, y_from, y_to,
                   step.edge_label, step.edge_color,
                   small_from, small_to)

    # ── Vertical "timeline" spine ────────────────────────────────────
    ax.plot([_X - _NW * 0.51, _X - _NW * 0.51],
            [positions[0][1], positions[-1][1]],
            color="#2a3a5a", linewidth=1.0, linestyle=":", alpha=0.4, zorder=1)

    # ── Error notice ─────────────────────────────────────────────────
    if errors:
        ax.text(0, y - 0.7,
                f"  {len(errors)} pipeline error(s) recorded  ",
                ha="center", va="center", fontsize=8,
                color="#e06060", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor="#1a0505", edgecolor="#cc3030", linewidth=1.2))

    # ── Legend ───────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=_COL["researcher"][0],
                       edgecolor=_COL["researcher"][1],      label="researcher (first pass)"),
        mpatches.Patch(facecolor=_COL["researcher_retry"][0],
                       edgecolor=_COL["researcher_retry"][1], label="researcher (retry pass)"),
        mpatches.Patch(facecolor=_COL["evaluator_high"][0],
                       edgecolor=_COL["evaluator_high"][1],   label="HIGH_QUALITY"),
        mpatches.Patch(facecolor=_COL["evaluator_ok"][0],
                       edgecolor=_COL["evaluator_ok"][1],     label="ACCEPTABLE_QUALITY"),
        mpatches.Patch(facecolor=_COL["evaluator_poor"][0],
                       edgecolor=_COL["evaluator_poor"][1],   label="POOR_QUALITY"),
    ]
    ax.legend(handles=legend_items,
              loc="lower right",
              facecolor="#111622", edgecolor="#334466",
              labelcolor=_TEXT_SUB, fontsize=7.5,
              framealpha=0.9, title="node types",
              title_fontsize=7, labelspacing=0.6)

    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    plt.close(fig)
    print(f"  [OK] Pipeline graph saved        -> {os.path.abspath(output_path)}")
