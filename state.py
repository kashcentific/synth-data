# state.py

from typing import Annotated, Any, Dict, List, Optional, TypedDict
import operator


class ThinkerState(TypedDict, total=False):
    """
    Shared LangGraph state for Thinker pipeline
    """

    # -----------------------------------
    # Input from main.py
    # -----------------------------------

    dataset: Any
    user_hint: Optional[str]

    # -----------------------------------
    # Metadata extraction output
    # -----------------------------------

    raw_metadata: Dict[str, Any]

    # -----------------------------------
    # Thinker output
    # -----------------------------------

    thinker_output: Dict[str, Any]

    # -----------------------------------
    # Error accumulator
    # -----------------------------------

    errors: Annotated[List[str], operator.add]