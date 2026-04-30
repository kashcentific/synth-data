# state.py

from typing import Annotated, Any, Dict, List, Optional, TypedDict
import operator


class ThinkerState(TypedDict, total=False):
    """
    Shared LangGraph state flowing through the entire pipeline:
      extract_metadata -> thinker -> researcher -> evaluator -> END
    """

    # --- Inputs ------------------------------------------------
    dataset: Any                      # raw HuggingFace dataset object
    user_hint: Optional[str]          # optional user-supplied context

    # --- Metadata extraction output ----------------------------
    raw_metadata: Dict[str, Any]      # deterministic column statistics

    # --- Universal metrics (computed immediately after metadata) ---
    universal_metric_results: List[Dict[str, Any]]  # always first in pipeline

    # --- Agent outputs -----------------------------------------
    thinker_output: Dict[str, Any]    # domain/type/column profiles/agents
    researcher_output: Dict[str, Any] # proposed metrics + research context
    researcher_iteration: int          # how many times researcher has looped
    researcher_retry: bool             # researcher signals it needs another pass

    # --- Evaluator ---------------------------------------------
    per_metric_results: List[Dict[str, Any]]  # one entry per computed metric
    evaluator_output: Dict[str, Any]          # final synthesised verdict

    # --- Error accumulator (merged across all nodes) -----------
    errors: Annotated[List[str], operator.add]
