import json
import sys
import os

# Ensure project root is on sys.path so "from base import BaseAgent" works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict

from base import BaseAgent


class ThinkerAgent(BaseAgent):
    """
    The Thinker Agent is the reasoning brain of the system.

    It receives the deterministic metadata extracted from the CSV and a
    user-provided hint (optional), and returns a fully structured analysis:

    - dataset type and domain
    - semantic role of every column
    - surface-level quality hints
    - governance warning flags
    - reference dataset assessment
    - which downstream agents should run and why

    This output drives all future orchestration decisions.
    """

    def build_prompt(
        self,
        metadata: Dict[str, Any],
        user_hint: str | None,
    ) -> str:

        hint_block = (
            f"\nUser Hint: \"{user_hint}\"\n"
            "Take this hint into account when reasoning about domain, intent, "
            "and which agents to prioritise.\n"
            if user_hint
            else "\nNo user hint provided.\n"
        )

        return f"""You are the Thinker Agent — the reasoning brain of a Synthetic Data Audit System.

Your job: deeply analyse a dataset's metadata and produce a structured audit plan.
You are NOT running quality checks yet. You are understanding the dataset and deciding
what should happen next.
{hint_block}
Dataset Metadata:
{json.dumps(metadata, indent=2)}

Think carefully and return ONLY valid JSON — no markdown, no extra text.

Return exactly this structure:

{{
  "dataset_type": "<tabular|code|qna|dialogue|jira|trajectory|mixed>",

  "domain": {{
    "name": "<e.g. healthcare, e-commerce, finance, software, HR, NLP, general>",
    "confidence": <0.0–1.0>,
    "reasoning": "<why you believe this is the domain>"
  }},

  "column_profiles": [
    {{
      "column_name": "<exact column name>",
      "inferred_dtype": "<numerical|categorical|text|boolean|datetime|code|unknown>",
      "semantic_role": "<identifier|label|feature|text_content|code|question|answer|timestamp|metadata|other>",
      "notes": "<anything worth flagging — suspicious values, ambiguous role, potential PII, etc.>"
    }}
  ],

  "surface_quality_hints": [
    "<e.g. high null rate in column X>",
    "<e.g. severe class imbalance likely in column Y>",
    "<e.g. duplicate rows detected>"
  ],

  "governance_warnings": [
    "<e.g. column 'email' likely contains real PII>",
    "<e.g. 'age' + 'zip' combination is a quasi-identifier risk>"
  ],

  "reference_dataset": {{
    "seems_needed": <true|false>,
    "reasoning": "<why a real/reference dataset would or wouldn't help evaluation>"
  }},

  "recommended_agents": {{
    "governance":  {{ "run": true,  "reason": "<always on — privacy is non-negotiable>" }},
    "math":        {{ "run": <true|false>, "reason": "<why>" }},
    "semantic":    {{ "run": <true|false>, "reason": "<why>" }},
    "utility":     {{ "run": <true|false>, "reason": "<why — only true when downstream task is clear>" }}
  }},

  "execution_notes": [
    "<e.g. Math agent should focus on class balance for column 'label'>",
    "<e.g. Semantic agent should check answer–question alignment>",
    "<e.g. TSTR requires a real test set — flag if missing>"
  ],

  "user_hint_influence": "<how the user hint (if any) changed your reasoning>",

  "reasoning_trace": "<your full step-by-step thinking before arriving at conclusions>"
}}
"""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        metadata  = state["raw_metadata"]
        user_hint = state.get("user_hint")

        # Show what the Thinker is analyzing
        print("\n" + "="*70)
        print("[THINKER] 🧠 THINKING PHASE — Dataset Understanding")
        print("="*70)
        print(f"[THINKER] Analyzing {len(metadata.get('columns', []))} columns...")
        print(f"[THINKER] Dataset size: {metadata.get('n_rows', '?')} rows")
        if user_hint:
            print(f"[THINKER] User hint provided: \"{user_hint}\"")
        
        print("\n[THINKER] 📋 Column Analysis:")
        for col_info in metadata.get('columns', [])[:5]:  # Show first 5
            col_name = col_info.get('name', '?')
            col_type = col_info.get('inferred_dtype', '?')
            print(f"[THINKER]   • {col_name} ({col_type})")
        if len(metadata.get('columns', [])) > 5:
            print(f"[THINKER]   ... and {len(metadata['columns']) - 5} more columns")
        
        print("\n[THINKER] 🤔 Now reasoning about domain, data quality, and next steps...")
        raw = self.call_llm(self.build_prompt(metadata, user_hint), stream=True)
        result = self.parse_json(raw)

        if result.get("_parse_error"):
            return {
                "thinker_output": result,
                "errors": ["Thinker: JSON parse failed — raw response stored in thinker_output.raw"],
            }

        print("\n[THINKER] ✓ Analysis complete")
        return {
            "thinker_output": result,
            "errors": [],
        }
