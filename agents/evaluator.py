"""
evaluator.py — EvaluatorAgent (Non-Deterministic Strategy Edition)

Per-metric execution strategy is decided at runtime:
  direct_math     — tries scipy/sympy first; optionally enriched by WolframAlpha formula lookup
  custom_function — LLM builds a named function from scratch by inspecting actual data
  semantic        — TF-IDF cosine / sentence-transformers / LLM-scored semantic consistency

After the main loop, auto-detects if text columns exist with no semantic metric computed
and injects an automatic semantic consistency pass.
"""

import contextlib
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Any, Dict, List, Optional

from base import BaseAgent
from config import MAX_ROWS_FOR_EVAL
from agents.universal_text_metrics import UniversalTextMetrics
from agents.tool_registry import ToolRegistry, MetricFn

# Optional semantic tools — degrade gracefully if not installed
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# Optional WolframAlpha
try:
    import wolframalpha as _wolframalpha_mod
    _WOLFRAM_AVAILABLE = True
except ImportError:
    _WOLFRAM_AVAILABLE = False


class EvaluatorAgent(BaseAgent):

    def __init__(self, model: str | None = None, temperature: float | None = None):
        kwargs = {}
        if model is not None:
            kwargs["model"] = model
        if temperature is not None:
            kwargs["temperature"] = temperature
        super().__init__(**kwargs)

        print("\n[EVALUATOR] Initializing computation tools...")

        # WolframAlpha — optional formula lookup
        self.wolfram_client = None
        if _WOLFRAM_AVAILABLE:
            app_id = os.getenv("WOLFRAM_APP_ID", "")
            if app_id:
                try:
                    self.wolfram_client = _wolframalpha_mod.Client(app_id)
                    print("[EVALUATOR]   * WolframAlpha: OK")
                except Exception as e:
                    print(f"[EVALUATOR]   * WolframAlpha: FAILED ({e})")
            else:
                print("[EVALUATOR]   * WolframAlpha: WOLFRAM_APP_ID not set — skipped")
        else:
            print("[EVALUATOR]   * WolframAlpha: not installed (pip install wolframalpha) — skipped")

        # Sentence transformers
        self._st_model = None
        if _ST_AVAILABLE:
            try:
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                print("[EVALUATOR]   * SentenceTransformer (all-MiniLM-L6-v2): OK")
            except Exception as e:
                print(f"[EVALUATOR]   * SentenceTransformer: FAILED ({e})")

        if _SKLEARN_AVAILABLE:
            print("[EVALUATOR]   * sklearn TF-IDF cosine: OK")
        else:
            print("[EVALUATOR]   * sklearn: not available — LLM-based semantic fallback active")

        self.utm = UniversalTextMetrics()
        self.tool_registry = ToolRegistry()

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _detect_identifier_cols(self, df: pd.DataFrame) -> List[str]:
        id_keywords = {"id", "key", "uuid", "index", "no", "num", "code", "ref", "serial"}
        return [col for col in df.columns if any(kw in col.lower() for kw in id_keywords)]

    def _detect_text_cols(self, df: pd.DataFrame) -> List[str]:
        """Returns columns that look like free text (avg length > 15 chars).

        Deliberately dtype-agnostic: with datasets>=3.x + pyarrow>=14, to_pandas()
        returns string columns as ArrowDtype(large_string), not plain object.
        Checking dtype==object would silently skip all text columns.
        """
        text_cols = []
        for col in df.columns:
            s = df[col]
            # Numeric / bool / datetime can never be free text
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

    def _format_row_narrative(
        self, df: pd.DataFrame, indices: List[int], id_cols: List[str], max_val_len: int = 90
    ) -> List[str]:
        narratives = []
        for i in indices[:15]:
            if not (0 <= i < len(df)):
                continue
            row   = df.iloc[i]
            parts = [f"Row {i}"]
            for col in id_cols:
                parts.append(f"{col}={str(row.get(col, ''))!r}")
            for col in df.columns:
                if col in id_cols:
                    continue
                val = str(row[col])
                if len(val) > max_val_len:
                    val = val[:max_val_len - 3] + "..."
                parts.append(f"{col}={val!r}")
            narratives.append(" | ".join(parts))
        return narratives

    # ──────────────────────────────────────────────────────────────────
    # WolframAlpha formula lookup
    # ──────────────────────────────────────────────────────────────────

    def _get_metric_formula_from_wolfram(self, metric_name: str, description: str) -> str:
        """
        Queries WolframAlpha for the mathematical formula of the metric.
        Returns a text snippet that is injected into the code-generation prompt.
        """
        if not self.wolfram_client:
            return ""
        try:
            query = f"mathematical formula definition {metric_name} statistics"
            res   = self.wolfram_client.query(query)
            pods  = [pod.text for pod in res.pods if pod.text and len(pod.text) < 500]
            formula_text = "\n".join(pods[:3]) if pods else ""
            if formula_text:
                print(f"[EVALUATOR]   WolframAlpha formula hint retrieved ({len(formula_text)} chars)")
            return formula_text
        except Exception as e:
            print(f"[EVALUATOR]   WolframAlpha query failed: {e}")
            return ""

    # ──────────────────────────────────────────────────────────────────
    # Strategy classification
    # ──────────────────────────────────────────────────────────────────

    def _classify_metric_strategy(
        self, metric: Dict, df: pd.DataFrame, col_profiles: List[Dict]
    ) -> Dict[str, str]:
        """
        Decides how to compute this metric:
          direct_math     — scipy/numpy is enough (entropy, KL-div, correlations, tests)
          semantic        — needs NLP/embedding approach (text coherence, similarity)
          custom_function — needs a bespoke function built from the data

        Uses fast heuristics first, falls back to LLM classification.
        """
        mtype = metric.get("metric_type", "").lower()
        blob  = (metric.get("metric_name", "") + " " + metric.get("description", "") + " " +
                 metric.get("execution_hint", "") + " " + mtype).lower()

        sem_signals  = {"semantic", "coherence", "embedding", "similarity", "tfidf", "cosine",
                        "readability", "text quality", "fluency", "nlp", "sentence", "language"}
        math_signals = {"entropy", "kl", "divergence", "wasserstein", "chi", "ks_test", "kolmogorov",
                        "distribution", "variance", "skewness", "kurtosis", "correlation",
                        "statistical", "scipy", "numpy", "frequency", "mutual_info"}

        if any(s in blob for s in sem_signals):
            return {"strategy": "semantic", "reasoning": "metric description mentions semantic/NLP concepts"}
        if any(s in blob for s in math_signals):
            return {"strategy": "direct_math", "reasoning": "metric maps to known scipy/numpy operations"}

        # Ask LLM for ambiguous cases
        col_names = [c.get("name") for c in col_profiles]
        text_cols = self._detect_text_cols(df)

        prompt = f"""Classify how this data quality metric should be computed.

Metric: {metric.get('metric_name')}
Description: {metric.get('description')}
Execution hint: {metric.get('execution_hint', '')}
Available columns: {col_names}
Text columns detected: {text_cols}

Choose ONE strategy:
- "direct_math": metric can be computed with scipy/numpy/pandas directly (statistical tests, distributions)
- "semantic": metric requires NLP/embedding/text similarity computation
- "custom_function": needs a purpose-built Python function that inspects the actual data

Return ONLY valid JSON:
{{"strategy": "direct_math|semantic|custom_function", "reasoning": "<1 sentence>"}}"""

        raw = self.call_llm(prompt, stream=False)
        try:
            stripped = raw.strip() if raw else ""
            parsed   = self.parse_json(stripped)
            if isinstance(parsed, dict) and "strategy" in parsed:
                return parsed
        except Exception:
            pass
        return {"strategy": "custom_function", "reasoning": "fallback: could not classify"}

    # ──────────────────────────────────────────────────────────────────
    # Code execution (shared)
    # ──────────────────────────────────────────────────────────────────

    def _normalise_exec_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert nullable/Arrow-backed numeric columns (Int64, Float64) to plain
        float64 numpy-backed dtype so that scipy functions accept them without
        silently failing. String columns are left as-is.
        """
        out = df.copy()
        for col in out.columns:
            s = out[col]
            if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
                try:
                    out[col] = s.astype("float64")
                except Exception:
                    pass
        return out

    def _execute_code(self, df: pd.DataFrame, code: str) -> Dict[str, Any]:
        buf      = io.StringIO()
        local_ns: Dict[str, Any] = {
            "df":               self._normalise_exec_df(df),
            "pd":               pd,
            "np":               np,
            "scipy_stats":      scipy_stats,
            "anomalous_indices": [],
        }

        # Optionally expose sklearn cosine for semantic code
        if _SKLEARN_AVAILABLE:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as _cs
            local_ns["TfidfVectorizer"]   = TfidfVectorizer
            local_ns["cosine_similarity"] = _cs

        if self._st_model:
            local_ns["sentence_model"] = self._st_model

        try:
            with contextlib.redirect_stdout(buf):
                exec(code, local_ns)  # noqa: S102

            raw_anomalous = local_ns.get("anomalous_indices", [])
            if hasattr(raw_anomalous, "tolist"):
                raw_anomalous = raw_anomalous.tolist()
            try:
                anomalous = [int(i) for i in raw_anomalous]
            except (TypeError, ValueError):
                anomalous = []

            return {
                "success":           True,
                "output":            buf.getvalue(),
                "anomalous_indices": anomalous[:50],
                "error":             None,
            }
        except Exception as exc:
            return {
                "success":           False,
                "output":            buf.getvalue(),
                "anomalous_indices": [],
                "error":             str(exc),
            }

    # ──────────────────────────────────────────────────────────────────
    # Tool registry helpers
    # ──────────────────────────────────────────────────────────────────

    def _execute_from_registry(
        self, fn: MetricFn, df: pd.DataFrame, col: str
    ) -> Dict[str, Any]:
        """
        Call a pre-built or LLM-cached MetricFn and normalise its output
        into the same shape that _execute_code returns.
        """
        try:
            value, anomalous, output = fn(df, col)
            # Ensure the output contains the METRIC_VALUE marker so the
            # downstream display code can extract the number.
            if "METRIC_VALUE:" not in output:
                output = f"METRIC_VALUE: {value:.4f}\n" + output
            return {
                "success":           True,
                "output":            output,
                "anomalous_indices": [int(i) for i in anomalous],
                "error":             None,
            }
        except Exception as exc:
            return {
                "success":           False,
                "output":            "",
                "anomalous_indices": [],
                "error":             str(exc),
            }

    def _make_registry_fn(self, code: str) -> MetricFn:
        """
        Wrap a code string (as generated by the LLM) into a MetricFn so it
        can be stored in the tool registry and reused without re-generating.
        The `col` param is accepted for interface consistency but the code
        already embeds the specific column names it needs.
        """
        def fn(df: pd.DataFrame, col: str) -> tuple:
            result = self._execute_code(df, code)
            output    = result.get("output", "")
            anomalous = result.get("anomalous_indices", [])
            value     = 0.5
            for line in output.splitlines():
                if line.startswith("METRIC_VALUE:"):
                    try:
                        value = float(line.replace("METRIC_VALUE:", "").strip())
                    except ValueError:
                        pass
                    break
            return value, anomalous, output
        return fn

    # ──────────────────────────────────────────────────────────────────
    # Strategy: direct_math — scipy-first with optional Wolfram hint
    # ──────────────────────────────────────────────────────────────────

    def _execute_direct_math(
        self, df: pd.DataFrame, metric: Dict, col_profiles: List[Dict],
        wolfram_formula: str, retry_count: int
    ) -> Dict[str, Any]:
        col_info = json.dumps(
            [{"name": c.get("name"), "dtype": c.get("inferred_dtype"),
              "null_pct": c.get("null_pct"), "unique_count": c.get("unique_count")}
             for c in col_profiles], indent=2
        )
        try:
            sample_str = df.head(5).to_string(max_cols=10)
        except Exception:
            sample_str = "(unavailable)"

        formula_block = (
            f"\nMathematical formula (from WolframAlpha):\n{wolfram_formula}\n"
            if wolfram_formula else ""
        )
        retry_note = (
            f"\n[RETRY {retry_count}] CRITICAL: previous code produced NO output (silent failure). "
            "Most common cause: bare 'except: pass' swallowed an error. "
            "FIX: never use 'except: pass' — always print the exception. "
            "Also ensure your print statements are OUTSIDE any conditional blocks so they always execute."
        ) if retry_count > 0 else ""

        prompt = f"""You are a Python data scientist implementing a mathematical data quality metric.
Use scipy/numpy/pandas for a direct mathematical computation.
{formula_block}
Metric Name       : {metric.get('metric_name')}
Metric Description: {metric.get('description')}
Execution Hint    : {metric.get('execution_hint')}

Available in scope: df, pd, np, scipy_stats
DataFrame schema:
{col_info}

Sample data (first 5 rows):
{sample_str}

CRITICAL RULES — failure to follow these causes silent bugs:
1. ALWAYS convert numeric columns to float before scipy: vals = df['col'].dropna().to_numpy(dtype=float)
2. Prefer scipy.stats functions (entropy, ks_2samp, chi2_contingency, pearsonr, etc.)
3. Write ONLY executable Python — no markdown, no explanations
4. Set: anomalous_indices = [<list of integer row indices that are statistical outliers>]
5. The print statements below are MANDATORY — they must execute unconditionally:
       print("METRIC_VALUE:", <main numeric result>)
       print("ANOMALOUS_ROWS:", anomalous_indices[:20])
6. In except blocks NEVER use 'pass' — always print the error:
       except Exception as e: print("ERROR:", e); print("METRIC_VALUE:", 0.0)
7. Do NOT call sys.exit().{retry_note}

Write the code now (raw Python only):"""

        raw_code = self.call_llm(prompt, stream=True)
        raw_code = re.sub(r"```(?:python)?\s*\n?(.*?)\n?\s*```", r"\1", raw_code, flags=re.DOTALL).strip()
        exec_result = self._execute_code(df, raw_code)
        return {"code": raw_code, "execution": exec_result, "retry_count": retry_count}

    # ──────────────────────────────────────────────────────────────────
    # Strategy: custom_function — LLM builds a named function from scratch
    # ──────────────────────────────────────────────────────────────────

    def _build_and_execute_custom_function(
        self, df: pd.DataFrame, metric: Dict, col_profiles: List[Dict], retry_count: int
    ) -> Dict[str, Any]:
        col_info = json.dumps(
            [{"name": c.get("name"), "dtype": c.get("inferred_dtype"),
              "null_pct": c.get("null_pct"), "unique_count": c.get("unique_count")}
             for c in col_profiles], indent=2
        )
        try:
            sample_str  = df.head(5).to_string(max_cols=10)
            stats_str   = df.describe(include="all").to_string(max_cols=10)
        except Exception:
            sample_str = stats_str = "(unavailable)"

        retry_note = (
            f"\n[RETRY {retry_count}] CRITICAL: previous code produced NO output (silent failure). "
            "Most common cause: bare 'except: pass' or print inside an if-block that never ran. "
            "Simplify the function and ensure print('METRIC_VALUE:', ...) ALWAYS executes."
        ) if retry_count > 0 else ""

        fn_name = re.sub(r'[^a-z0-9]', '_', metric.get('metric_name', 'metric').lower())

        prompt = f"""You are a Python data scientist. Build a NAMED FUNCTION that computes this data quality metric.

Metric Name       : {metric.get('metric_name')}
Metric Description: {metric.get('description')}
Execution Hint    : {metric.get('execution_hint')}

Available in scope: df, pd, np, scipy_stats
DataFrame schema:
{col_info}

Sample data (first 5 rows):
{sample_str}

Descriptive statistics:
{stats_str}

REQUIRED CODE STRUCTURE — copy this pattern exactly:
```
def compute_{fn_name}(df):
    try:
        # ALWAYS convert numeric cols to float first: vals = df['col'].dropna().to_numpy(dtype=float)
        metric_value = <computed float>
        anomalous_idx = [<integer row indices>]
    except Exception as e:
        print("ERROR:", e)
        metric_value = 0.0
        anomalous_idx = []
    return metric_value, anomalous_idx

metric_value, anomalous_indices = compute_{fn_name}(df)
print("METRIC_VALUE:", metric_value)   # MANDATORY — must always execute
print("ANOMALOUS_ROWS:", anomalous_indices[:20])
```

RULES:
1. Convert numeric columns: .dropna().to_numpy(dtype=float) before any scipy call
2. The except block must print the error and set metric_value — NEVER use bare 'except: pass'
3. The final print("METRIC_VALUE:", ...) is OUTSIDE the function — it always runs
4. Return (float, list_of_int) from the function
5. Do NOT call sys.exit().{retry_note}

Write the code now (raw Python only):"""

        raw_code = self.call_llm(prompt, stream=True)
        raw_code = re.sub(r"```(?:python)?\s*\n?(.*?)\n?\s*```", r"\1", raw_code, flags=re.DOTALL).strip()
        exec_result = self._execute_code(df, raw_code)
        return {"code": raw_code, "execution": exec_result, "retry_count": retry_count}

    # ──────────────────────────────────────────────────────────────────
    # Strategy: semantic — NLP-based consistency check
    # ──────────────────────────────────────────────────────────────────

    def _execute_semantic_metric(
        self, df: pd.DataFrame, metric: Dict, col_profiles: List[Dict]
    ) -> Dict[str, Any]:
        """
        Computes semantic consistency using the best available tool:
          1. sentence-transformers (best quality)
          2. sklearn TF-IDF cosine (good, always available if sklearn installed)
          3. LLM-scored sampling (fallback, always works)
        """
        text_cols = self._detect_text_cols(df)
        if not text_cols:
            return {
                "code": "# No text columns found",
                "execution": {
                    "success": False, "output": "No text columns detected for semantic metric.",
                    "anomalous_indices": [], "error": "No text columns"
                },
                "retry_count": 0,
            }

        col = text_cols[0]
        texts = df[col].dropna().astype(str).tolist()

        print(f"[EVALUATOR]   Semantic target column: '{col}' ({len(texts)} texts)")

        # ── Approach 1: sentence-transformers ─────────────────────────
        if self._st_model and len(texts) <= 2000:
            code = f"""
import numpy as np
texts = df['{col}'].dropna().astype(str).tolist()
valid_idx = df['{col}'].dropna().index.tolist()

embeddings = sentence_model.encode(texts, show_progress_bar=False)
# Compute pairwise cosine similarity sample (vs mean embedding)
mean_emb = embeddings.mean(axis=0, keepdims=True)
from numpy.linalg import norm
sims = np.dot(embeddings, mean_emb.T).flatten() / (
    norm(embeddings, axis=1) * norm(mean_emb) + 1e-9
)
threshold = np.percentile(sims, 10)
anomalous_local = [i for i, s in enumerate(sims) if s < threshold]
anomalous_indices = [valid_idx[i] for i in anomalous_local if i < len(valid_idx)]

mean_sim = float(np.mean(sims))
print("METRIC_VALUE:", round(mean_sim, 4))
print("ANOMALOUS_ROWS:", anomalous_indices[:20])
print(f"Semantic consistency (mean cosine to centroid): {{mean_sim:.4f}}")
print(f"Low-similarity threshold: {{threshold:.4f}}")
print(f"Flagged {{len(anomalous_indices)}} outlier rows")
"""
            exec_result = self._execute_code(df, code)
            if exec_result["success"]:
                return {"code": code, "execution": exec_result, "retry_count": 0}

        # ── Approach 2: TF-IDF cosine ─────────────────────────────────
        if _SKLEARN_AVAILABLE:
            code = f"""
texts = df['{col}'].fillna('').astype(str).tolist()
valid_mask = df['{col}'].notna()
valid_idx  = df.index[valid_mask].tolist()
valid_texts = [t for t, m in zip(texts, valid_mask) if m]

vec  = TfidfVectorizer(max_features=500, stop_words='english')
tfidf = vec.fit_transform(valid_texts)

# Compare each doc to mean TF-IDF vector
mean_vec = tfidf.mean(axis=0)
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(tfidf, mean_vec).flatten()

threshold = float(np.percentile(sims, 10))
anomalous_local = [i for i, s in enumerate(sims) if s < threshold]
anomalous_indices = [valid_idx[i] for i in anomalous_local if i < len(valid_idx)]

mean_sim = float(np.mean(sims))
print("METRIC_VALUE:", round(mean_sim, 4))
print("ANOMALOUS_ROWS:", anomalous_indices[:20])
print(f"TF-IDF semantic consistency (mean cosine): {{mean_sim:.4f}}")
print(f"Low-similarity threshold: {{threshold:.4f}}")
print(f"Flagged {{len(anomalous_indices)}} outlier rows")
"""
            exec_result = self._execute_code(df, code)
            if exec_result["success"]:
                return {"code": code, "execution": exec_result, "retry_count": 0}

        # ── Approach 3: LLM-based scoring on a sample ─────────────────
        sample_size  = min(30, len(texts))
        sample_texts = texts[:sample_size]
        sample_with_idx = [{"index": i, "text": t[:200]} for i, t in enumerate(sample_texts)]

        prompt = f"""You are a data quality analyst. Score each text for semantic consistency
with the overall dataset topic/domain. The dataset column is '{col}'.

Texts (sample of {sample_size}):
{json.dumps(sample_with_idx, indent=2)}

For each text, output a consistency score 0.0-1.0 (1.0 = perfectly consistent, 0.0 = clearly off-topic/anomalous).

Return ONLY valid JSON array:
[{{"index": 0, "score": 0.9, "note": "..."}}]"""

        raw_llm = self.call_llm(prompt, stream=False)
        try:
            stripped = raw_llm.strip() if raw_llm else ""
            scores   = self.parse_json(stripped if stripped.startswith("[") else f"[{stripped}]")
            if isinstance(scores, list) and len(scores) == 1 and isinstance(scores[0], list):
                scores = scores[0]
            if isinstance(scores, list) and all(isinstance(s, dict) for s in scores):
                anomalous_local = [s["index"] for s in scores if s.get("score", 1.0) < 0.5]
                mean_score      = sum(s.get("score", 1.0) for s in scores) / max(len(scores), 1)
                output_text = (
                    f"METRIC_VALUE: {round(mean_score, 4)}\n"
                    f"ANOMALOUS_ROWS: {anomalous_local[:20]}\n"
                    f"LLM-scored semantic consistency on {sample_size} samples: {mean_score:.4f}\n"
                    f"Flagged {len(anomalous_local)} potentially off-topic rows"
                )
                return {
                    "code": "# LLM-based semantic scoring",
                    "execution": {
                        "success":           True,
                        "output":            output_text,
                        "anomalous_indices": anomalous_local,
                        "error":             None,
                    },
                    "retry_count": 0,
                }
        except Exception:
            pass

        return {
            "code": "# Semantic metric — all approaches failed",
            "execution": {
                "success": False,
                "output":  "All semantic computation approaches failed.",
                "anomalous_indices": [],
                "error":   "Semantic metric failed",
            },
            "retry_count": 0,
        }

    # ──────────────────────────────────────────────────────────────────
    # Auto semantic check — adds a semantic pass if text cols + no semantic done
    # ──────────────────────────────────────────────────────────────────

    def _auto_semantic_check(
        self, df: pd.DataFrame, col_profiles: List[Dict], per_metric_results: List[Dict]
    ) -> List[Dict]:
        """
        If text columns exist but no semantic metric was computed, automatically
        runs a semantic consistency check and returns result entries to append.
        """
        text_cols = self._detect_text_cols(df)
        if not text_cols:
            return []

        computed_types = {r.get("metric_type", "") for r in per_metric_results}
        if "semantic_consistency" in computed_types or "text_quality" in computed_types:
            print(f"[EVALUATOR]   Semantic metric already computed — skipping auto-check")
            return []

        print(f"\n[EVALUATOR] Auto-detected text columns: {text_cols}")
        print(f"[EVALUATOR] No semantic metric computed — running automatic semantic consistency check...")

        id_cols   = self._detect_identifier_cols(df)
        auto_metric = {
            "metric_name":      "auto_semantic_consistency",
            "metric_type":      "semantic_consistency",
            "description":      f"Automatic semantic consistency check for text column '{text_cols[0]}'",
            "execution_hint":   f"Pairwise semantic similarity for column '{text_cols[0]}'",
            "relevance_score":  0.85,
            "feasibility_score": 0.9,
            "reasoning":        "Auto-injected: text columns present but no semantic metric was proposed",
        }

        result        = self._execute_semantic_metric(df, auto_metric, col_profiles)
        exec_result   = result["execution"]

        if not exec_result.get("success"):
            print(f"[EVALUATOR]   Auto semantic check failed: {exec_result.get('error', 'unknown')}")
            return []

        anomalous_indices = exec_result.get("anomalous_indices", [])
        narratives        = self._format_row_narrative(df, anomalous_indices, id_cols)

        interpretation = self._interpret_metric(
            "auto_semantic_consistency",
            auto_metric["description"],
            exec_result.get("output", ""),
            narratives,
            exec_result.get("error"),
        )

        try:
            valid_idx         = [i for i in anomalous_indices if 0 <= i < len(df)][:10]
            anomalous_samples = df.iloc[valid_idx].fillna("NULL").to_dict(orient="records") if valid_idx else []
        except Exception:
            anomalous_samples = []

        print(f"[EVALUATOR]   Auto semantic check: {len(anomalous_indices)} anomalous rows found")

        return [{
            "metric_name":              "auto_semantic_consistency",
            "metric_type":              "semantic_consistency",
            "description":              auto_metric["description"],
            "reasoning":                auto_metric["reasoning"],
            "source_influence":         "Auto-injected by evaluator",
            "feasibility_score":        0.9,
            "quality_score":            0.85,
            "execution_output":         exec_result.get("output", "(no output)"),
            "generated_code":           result.get("code", ""),
            "error":                    exec_result.get("error"),
            "anomalous_row_count":      len(anomalous_indices),
            "anomalous_row_indices":    anomalous_indices[:20],
            "anomalous_row_narratives": narratives,
            "anomalous_samples":        anomalous_samples,
            "metric_interpretation":    interpretation,
            "retry_count":              0,
        }]

    # ──────────────────────────────────────────────────────────────────
    # ReAct loop: feasibility reasoning
    # ──────────────────────────────────────────────────────────────────

    def _reason_metric_feasibility(self, metrics: List[Dict], col_profiles: List[Dict]) -> List[Dict]:
        col_names = [c.get("name") for c in col_profiles]
        col_info  = json.dumps(col_profiles, indent=2)

        prompt = f"""You are a data quality metric feasibility analyst.

SCOPE: This is a DATA QUALITY pipeline. Automatically mark any privacy, PII, security,
compliance, or governance metrics as not computable (is_computable: false, feasibility_score: 0.0).
Only quality metrics (statistical, distributional, semantic, text-quality, domain-specific) are in scope.

Available Columns: {', '.join(col_names)}

Column Details:
{col_info}

Proposed Metrics:
{json.dumps(metrics[:10], indent=2)}

For EACH in-scope metric assess:
1. Can it be computed with available columns?
2. Is the execution hint realistic?
3. Implementation feasibility (0.0-1.0)?

Return JSON array:
[
  {{"metric_name": "...", "is_computable": true, "feasibility_score": 0.9, "reason": "..."}}
]"""

        raw = self.call_llm(prompt, stream=False)
        try:
            stripped = raw.strip() if raw else ""
            result   = self.parse_json(stripped if stripped.startswith("[") else f"[{stripped}]")
            if isinstance(result, list) and all(isinstance(s, dict) for s in result):
                return result
            if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
                return result[0]
        except Exception:
            pass
        return []

    def _observe_result_quality(self, exec_result: Dict[str, Any]) -> Dict[str, Any]:
        output = exec_result.get("output") or ""
        has_metric_value = "METRIC_VALUE:" in output

        qa = {
            "success":         exec_result.get("success", False),
            "has_error":       bool(exec_result.get("error")),
            "has_output":      bool(output.strip()),
            "has_metric_value": has_metric_value,
            "anomalies_found": len(exec_result.get("anomalous_indices", [])) > 0,
            "quality_score":   0.0,
            "issues":          [],
        }

        if qa["has_error"]:
            qa["issues"].append(f"Execution error: {exec_result['error']}")
            qa["quality_score"] = 0.2
        elif not qa["has_output"]:
            # Code ran without exception but printed nothing — silent bug in LLM code
            qa["issues"].append("No output produced — code ran silently (likely silent except: pass)")
            qa["quality_score"] = 0.1   # force retry
        elif not has_metric_value:
            qa["issues"].append("Output produced but missing METRIC_VALUE: line")
            qa["quality_score"] = 0.35  # retry
        elif qa["success"]:
            qa["quality_score"] = 0.8 + (0.2 if qa["anomalies_found"] else 0.0)

        return qa

    def _should_retry_metric(self, quality_assessment: Dict, retry_count: int) -> bool:
        return quality_assessment["quality_score"] < 0.6 and retry_count < 2

    # ──────────────────────────────────────────────────────────────────
    # Interpretation
    # ──────────────────────────────────────────────────────────────────

    def _interpret_metric(
        self, metric_name: str, metric_description: str,
        exec_output: str, anomalous_narratives: List[str], exec_error: Optional[str]
    ) -> str:
        capped      = anomalous_narratives[:5]
        rows_block  = "\n".join(f"  {n}" for n in capped) if capped else "  (none flagged)"
        error_note  = f"NOTE: execution raised an error: {exec_error}" if exec_error else ""

        prompt = (
            f"You are a data quality analyst interpreting a metric result. Write 3-5 sentences.\n"
            f"Metric: {metric_name} — {metric_description}\n\n"
            f"Execution output:\n{(exec_output or '(none)')[:600]}\n"
            f"{error_note}\n\n"
            f"Up to 5 flagged rows:\n{rows_block}\n\n"
            f"IMPORTANT CONTEXT:\n"
            f"- This is a DATA QUALITY assessment only. Do NOT flag privacy, PII, or security concerns.\n"
            f"- Code snippets, execution traces, long text, technical jargon, agent reasoning, and\n"
            f"  mathematical notation are all NORMAL content for benchmark and research datasets.\n"
            f"- Only flag rows as anomalous if they show genuine quality issues: broken structure,\n"
            f"  unexpected null patterns, statistical outliers, or clear formatting inconsistencies.\n"
            f"- Be calibrated: finding some outliers is expected and healthy, not alarming.\n\n"
            f"Write prose only (no JSON). Cite actual numbers and values. "
            f"Describe what was found and whether it represents a quality concern or normal variation."
        )

        try:
            result = self.call_llm(prompt, fallback="(interpretation unavailable)", stream=True)
            return result or "(interpretation unavailable)"
        except Exception as exc:
            return f"(interpretation failed: {exc})"

    # ──────────────────────────────────────────────────────────────────
    # Final synthesis
    # ──────────────────────────────────────────────────────────────────

    def _synthesise_report(
        self, per_metric_results: List[Dict], thinker_out: Dict, df_shape: tuple
    ) -> Dict[str, Any]:
        domain = thinker_out.get("domain", {}).get("name", "unknown")

        universal_blocks = []
        research_blocks  = []
        for r in per_metric_results:
            block = {
                "metric":          r["metric_name"],
                "category":        r.get("metric_category", "research"),
                "exec_output":     (r.get("execution_output") or "")[:250],
                "interpretation":  r.get("metric_interpretation", "")[:250],
                "anomalous_count": r["anomalous_row_count"],
                "sample_rows":     r.get("anomalous_row_narratives", [])[:2],
                "error":           r.get("error"),
            }
            if r.get("metric_source") == "universal":
                universal_blocks.append(block)
            else:
                research_blocks.append(block)

        prompt = f"""You are the Final Synthesis Engine of a Data Quality Audit System.

Dataset domain  : {domain}
DataFrame shape : {df_shape[0]} rows x {df_shape[1]} columns

UNIVERSAL TEXT METRICS (predefined, always run for text data):
{json.dumps(universal_blocks, indent=2)}

RESEARCH-BASED METRICS (researcher-proposed, dataset-specific):
{json.dumps(research_blocks, indent=2)}

YOUR ROLE: Assess DATA QUALITY only — structure, consistency, completeness, statistical properties.
This is NOT a safety, privacy, or governance review.

CALIBRATION RULES — read carefully before assigning verdict:
1. Code, execution traces, agent reasoning, mathematical notation, long sequences = NORMAL content.
   Do NOT penalise a dataset for containing these.
2. Benchmark and research datasets are HIGH_QUALITY or ACCEPTABLE_QUALITY by default.
   Only downgrade if you have concrete statistical evidence of systematic quality failure.
3. Some outliers and anomalies are EXPECTED and HEALTHY in any real dataset.
   Penalise only if the proportion is unusually high (>20% of rows) or the pattern is systematic.
4. Metric computation errors count against the pipeline, not the data.
5. UNSAFE/POOR_QUALITY requires strong, specific evidence: majority of rows broken,
   systematic inconsistency, extreme null rates (>60%), or proven structural corruption.

Verdict scale (data quality, not safety):
- HIGH_QUALITY    : data is well-structured, consistent, and suitable for its domain purpose
- ACCEPTABLE_QUALITY : minor quality issues that do not impair usability; expected variations present
- POOR_QUALITY    : significant systematic quality failures backed by concrete metric evidence

Instructions:
- Cite actual numbers from execution outputs.
- Be factual and calibrated — do not amplify minor findings.
- Never invent statistics not present in the evidence above.

Return ONLY valid JSON — no markdown:

{{
  "evaluated_metrics": ["<list of metric names computed>"],
  "dataset_level_evidence": "<overall findings with actual numbers from execution>",
  "sample_level_inconsistencies": "<only genuine structural/formatting anomalies — not domain-normal content>",
  "quality_observations": "<what the metrics reveal about the data's fitness for purpose>",
  "statistical_justification": "<empirical summary with numbers supporting the verdict>",
  "quality_by_metric": [
    {{
      "metric"     : "<metric_name>",
      "finding"    : "<what was found — cite numbers>",
      "quality_level" : "<GOOD|ACCEPTABLE|POOR>",
      "note"       : "<1 sentence — be calibrated>"
    }}
  ],
  "final_verdict"     : "<HIGH_QUALITY|ACCEPTABLE_QUALITY|POOR_QUALITY>",
  "verdict_reasoning" : "<2-3 sentences grounded in metric evidence — benchmark/agent data is acceptable by default>"
}}"""

        raw = self.call_llm(prompt, stream=True)
        return self.parse_json(raw)

    # ──────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        researcher_out = state.get("researcher_output", {})
        thinker_out    = state.get("thinker_output", {})
        raw_metadata   = state.get("raw_metadata", {})
        dataset        = state.get("dataset")

        if not researcher_out:
            return {"evaluator_output": {}, "per_metric_results": [], "errors": ["Evaluator: researcher_output missing."]}
        if dataset is None:
            return {"evaluator_output": {}, "per_metric_results": [], "errors": ["Evaluator: dataset missing from state."]}

        df = dataset.to_pandas()
        # Normalise Arrow-backed extension types (datasets>=3.x + pyarrow>=14) to
        # numpy/nullable equivalents so that dtype checks, isnull, and exec() code work.
        try:
            df = df.convert_dtypes(dtype_backend="numpy_nullable")
        except Exception:
            pass

        original_len = len(df)
        if original_len > MAX_ROWS_FOR_EVAL:
            df = df.sample(MAX_ROWS_FOR_EVAL, random_state=42).reset_index(drop=True)
            print(f"[EVALUATOR] Dataset capped: {original_len} -> {MAX_ROWS_FOR_EVAL} rows.")

        col_profiles     = raw_metadata.get("columns", [])
        proposed_metrics = researcher_out.get("final_metrics", researcher_out.get("proposed_metrics", []))
        id_cols          = self._detect_identifier_cols(df)

        print("\n" + "=" * 70)
        print("[EVALUATOR] AGENTIC EVALUATION PHASE — Non-Deterministic Strategy Dispatch")
        print("=" * 70)
        print(f"[EVALUATOR] Dataset shape: {len(df)} rows x {len(df.columns)} columns")
        if id_cols:
            print(f"[EVALUATOR] Identifier columns: {id_cols}")
        else:
            print(f"[EVALUATOR] No explicit ID columns — using row indices")

        text_cols = self._detect_text_cols(df)
        if text_cols:
            print(f"[EVALUATOR] Text columns detected: {text_cols}")

        # ── Universal Text Metrics — already computed by universal_metrics_node ──
        # Pull from state; never recompute here so results always appear first.
        universal_results: List[Dict] = state.get("universal_metric_results", [])
        if universal_results:
            ok = sum(1 for r in universal_results if not r.get("error"))
            print(f"[EVALUATOR] Universal metrics: {ok}/{len(universal_results)} pre-computed (from pipeline node)")
        else:
            print(f"[EVALUATOR] No pre-computed universal metrics in state")

        if not proposed_metrics:
            return {"evaluator_output": {}, "per_metric_results": [], "errors": ["Evaluator: no proposed metrics."]}

        # ── Phase 1: Feasibility reasoning ────────────────────────────
        print(f"\n[EVALUATOR] Phase 1 (REASON): Evaluating metric feasibility...")
        feasibility_scores = self._reason_metric_feasibility(proposed_metrics, col_profiles)

        for metric in proposed_metrics:
            metric_name = metric.get("metric_name", "")
            for score_entry in feasibility_scores:
                if score_entry.get("metric_name") == metric_name:
                    metric["feasibility_score"] = score_entry.get("feasibility_score", 0.5)
                    metric["is_computable"]      = score_entry.get("is_computable", True)
                    metric["feasibility_reason"] = score_entry.get("reason", "")
                    break

        computable = [m for m in proposed_metrics
                      if m.get("is_computable", True) and m.get("feasibility_score", 0) > 0.4]
        print(f"[EVALUATOR] {len(computable)} / {len(proposed_metrics)} metrics are feasible")

        # ── Phase 2: ACT & OBSERVE — strategy-dispatched computation ──
        per_metric_results: List[Dict] = []

        # Primary text col used by tool-registry built-ins (text-based tools)
        primary_text_col = text_cols[0] if text_cols else ""

        for idx, metric in enumerate(computable, 1):
            name        = metric.get("metric_name", "unknown")
            description = metric.get("description", "")

            print(f"\n[EVALUATOR] Metric {idx}/{len(computable)}: '{name}'")
            print(f"[EVALUATOR]   Feasibility: {metric.get('feasibility_score', 0):.2f} | "
                  f"{metric.get('feasibility_reason', '')[:60]}")

            # ── 0. Tool-registry check (pre-built or previously cached) ─
            registry_fn = self.tool_registry.get(name)
            if registry_fn is not None:
                print(f"[EVALUATOR]   Strategy: registry_hit — using pre-built tool")
                exec_result = self._execute_from_registry(registry_fn, df, primary_text_col)
                quality     = self._observe_result_quality(exec_result)
                print(f"[EVALUATOR]   Registry quality: {quality['quality_score']:.2f}")

                if exec_result.get("success"):
                    anomalous_indices = exec_result.get("anomalous_indices", [])
                    narratives        = self._format_row_narrative(df, anomalous_indices, id_cols)
                    interpretation    = self._interpret_metric(
                        name, description, exec_result.get("output", ""),
                        narratives, exec_result.get("error"),
                    )
                    try:
                        valid_idx         = [i for i in anomalous_indices if 0 <= i < len(df)][:10]
                        anomalous_samples = df.iloc[valid_idx].fillna("NULL").to_dict(orient="records") if valid_idx else []
                    except Exception:
                        anomalous_samples = []

                    per_metric_results.append({
                        "metric_name":              name,
                        "metric_type":              metric.get("metric_type", "other"),
                        "description":              description,
                        "reasoning":                metric.get("reasoning", ""),
                        "source_influence":         metric.get("source_influence", ""),
                        "feasibility_score":        metric.get("feasibility_score", 0),
                        "quality_score":            quality["quality_score"],
                        "execution_output":         exec_result.get("output") or "(no output)",
                        "generated_code":           "# Tool registry (pre-built or cached)",
                        "error":                    None,
                        "anomalous_row_count":      len(anomalous_indices),
                        "anomalous_row_indices":    anomalous_indices[:20],
                        "anomalous_row_narratives": narratives,
                        "anomalous_samples":        anomalous_samples,
                        "metric_interpretation":    interpretation,
                        "retry_count":              0,
                        "computation_strategy":     "tool_registry",
                    })
                    continue
                else:
                    print(f"[EVALUATOR]   Registry function failed ({exec_result.get('error','')})"
                          " — falling back to LLM")

            # ── 1. Classify strategy (LLM path) ───────────────────────
            strategy_info = self._classify_metric_strategy(metric, df, col_profiles)
            strategy      = strategy_info.get("strategy", "custom_function")
            print(f"[EVALUATOR]   Strategy: {strategy} — {strategy_info.get('reasoning', '')[:70]}")

            # ── Semantic metrics: single attempt, no retry loop ────────
            if strategy == "semantic":
                print(f"[EVALUATOR]   Running semantic computation...")
                sem_result  = self._execute_semantic_metric(df, metric, col_profiles)
                exec_result = sem_result["execution"]
                quality     = self._observe_result_quality(exec_result)

                print(f"[EVALUATOR]   Semantic quality: {quality['quality_score']:.2f}")

                if not exec_result.get("success"):
                    print(f"[EVALUATOR]   Semantic metric failed — skipping")
                    continue

                # Register the working code so subsequent similar metrics skip LLM
                if sem_result.get("code") and sem_result["code"].strip() not in (
                    "# LLM-based semantic scoring", "# Semantic metric — all approaches failed"
                ):
                    self.tool_registry.register(
                        name, self._make_registry_fn(sem_result["code"]), source="llm_generated"
                    )

                anomalous_indices = exec_result.get("anomalous_indices", [])
                narratives        = self._format_row_narrative(df, anomalous_indices, id_cols)
                interpretation    = self._interpret_metric(name, description, exec_result.get("output", ""), narratives, exec_result.get("error"))

                try:
                    valid_idx         = [i for i in anomalous_indices if 0 <= i < len(df)][:10]
                    anomalous_samples = df.iloc[valid_idx].fillna("NULL").to_dict(orient="records") if valid_idx else []
                except Exception:
                    anomalous_samples = []

                per_metric_results.append({
                    "metric_name":              name,
                    "metric_type":              metric.get("metric_type", "semantic_consistency"),
                    "description":              description,
                    "reasoning":                metric.get("reasoning", ""),
                    "source_influence":         metric.get("source_influence", ""),
                    "feasibility_score":        metric.get("feasibility_score", 0),
                    "quality_score":            quality["quality_score"],
                    "execution_output":         exec_result.get("output") or "(no output)",
                    "generated_code":           sem_result.get("code", ""),
                    "error":                    exec_result.get("error"),
                    "anomalous_row_count":      len(anomalous_indices),
                    "anomalous_row_indices":    anomalous_indices[:20],
                    "anomalous_row_narratives": narratives,
                    "anomalous_samples":        anomalous_samples,
                    "metric_interpretation":    interpretation,
                    "retry_count":              0,
                    "computation_strategy":     "semantic",
                })
                continue

            # ── Math / custom metrics: retry loop ─────────────────────
            wolfram_formula = ""
            if strategy == "direct_math" and self.wolfram_client:
                wolfram_formula = self._get_metric_formula_from_wolfram(name, description)

            retry_count  = 0
            best_result  = None
            best_quality = 0.0
            active_strategy = strategy

            while retry_count < 3:
                print(f"\n[EVALUATOR]   ACT [{active_strategy}]: "
                      f"{'Generating code' if retry_count == 0 else f'Retry {retry_count}'}...")

                if active_strategy == "direct_math":
                    result = self._execute_direct_math(df, metric, col_profiles, wolfram_formula, retry_count)
                    # If direct_math fails twice, fall back to custom_function
                    if retry_count >= 1 and result["execution"].get("error"):
                        print(f"[EVALUATOR]   Falling back to custom_function strategy...")
                        active_strategy = "custom_function"
                        result = self._build_and_execute_custom_function(df, metric, col_profiles, retry_count)
                else:
                    result = self._build_and_execute_custom_function(df, metric, col_profiles, retry_count)

                exec_result = result["execution"]
                print(f"[EVALUATOR]   OBSERVE: quality check...")
                quality = self._observe_result_quality(exec_result)

                print(f"[EVALUATOR]   Quality score: {quality['quality_score']:.2f}")
                if quality["issues"]:
                    print(f"[EVALUATOR]   Issues: {', '.join(quality['issues'][:2])}")

                if quality["quality_score"] > best_quality:
                    best_quality = quality["quality_score"]
                    best_result  = {"code": result["code"], "execution": exec_result, "quality": quality,
                                    "strategy": active_strategy}

                if not self._should_retry_metric(quality, retry_count):
                    print(f"[EVALUATOR]   Result acceptable (quality: {quality['quality_score']:.2f})")
                    break

                retry_count += 1
                print(f"[EVALUATOR]   ITERATE: retrying...")

            if best_result is None:
                print(f"[EVALUATOR]   Failed to compute '{name}' after retries")
                continue

            # ── Cache the best working code into the tool registry ─────
            if best_result.get("code") and best_result["execution"].get("success"):
                self.tool_registry.register(
                    name, self._make_registry_fn(best_result["code"]), source="llm_generated"
                )

            exec_result       = best_result["execution"]
            anomalous_indices = exec_result.get("anomalous_indices", [])

            print(f"[EVALUATOR]   Found {len(anomalous_indices)} anomalous rows")

            narratives = self._format_row_narrative(df, anomalous_indices, id_cols)

            print(f"[EVALUATOR]   Interpreting result...")
            interpretation = self._interpret_metric(
                name, description,
                exec_result.get("output", "") or "",
                narratives, exec_result.get("error"),
            )
            print(f"[EVALUATOR]   {interpretation[:100]}...")

            try:
                valid_idx         = [i for i in anomalous_indices if 0 <= i < len(df)][:10]
                anomalous_samples = df.iloc[valid_idx].fillna("NULL").to_dict(orient="records") if valid_idx else []
            except Exception:
                anomalous_samples = []

            per_metric_results.append({
                "metric_name":              name,
                "metric_type":              metric.get("metric_type", "other"),
                "description":              description,
                "reasoning":                metric.get("reasoning", ""),
                "source_influence":         metric.get("source_influence", ""),
                "feasibility_score":        metric.get("feasibility_score", 0),
                "quality_score":            best_result["quality"]["quality_score"],
                "execution_output":         exec_result.get("output") or "(no output)",
                "generated_code":           best_result["code"],
                "error":                    exec_result.get("error"),
                "anomalous_row_count":      len(anomalous_indices),
                "anomalous_row_indices":    anomalous_indices[:20],
                "anomalous_row_narratives": narratives,
                "anomalous_samples":        anomalous_samples,
                "metric_interpretation":    interpretation,
                "retry_count":              retry_count,
                "computation_strategy":     best_result.get("strategy", active_strategy),
            })

        # ── Auto semantic consistency check ───────────────────────────
        auto_sem = self._auto_semantic_check(df, col_profiles, per_metric_results)
        if auto_sem:
            per_metric_results.extend(auto_sem)
            print(f"[EVALUATOR]   Added {len(auto_sem)} auto-semantic metric(s)")

        # ── Merge: universal (first) + research metrics ────────────────
        # Universal metrics are prepended so they appear as a distinct group
        all_metric_results = universal_results + per_metric_results

        # ── Phase 3: Consolidation ─────────────────────────────────────
        print(f"\n[EVALUATOR] Phase 3 (CONSOLIDATE): Building final evidence-backed report...")
        u_ok = sum(1 for r in universal_results  if not r.get("error"))
        r_ok = sum(1 for r in per_metric_results if not r.get("error"))
        print(f"[EVALUATOR]   Universal metrics : {u_ok}/{len(universal_results)}")
        print(f"[EVALUATOR]   Research metrics  : {r_ok}/{len(per_metric_results)}")
        print(f"[EVALUATOR]   Total             : {len(all_metric_results)}")

        if not all_metric_results:
            return {
                "evaluator_output": {},
                "per_metric_results": [],
                "errors": ["Evaluator: No metrics could be computed successfully."],
            }

        final_report = self._synthesise_report(all_metric_results, thinker_out, df.shape)

        if final_report.get("_parse_error"):
            return {
                "evaluator_output":   final_report,
                "per_metric_results": all_metric_results,
                "errors": ["Evaluator: JSON parse failed in final synthesis."],
            }

        verdict     = final_report.get("final_verdict", "UNKNOWN")
        computed    = [r for r in all_metric_results if not r.get("error")]
        avg_quality = sum(r.get("quality_score", 0) for r in computed) / max(len(computed), 1)
        print(f"\n[EVALUATOR] FINAL VERDICT: {verdict}")
        print(f"[EVALUATOR]   Average quality: {avg_quality:.2f}/1.0")

        return {
            "evaluator_output":   final_report,
            "per_metric_results": all_metric_results,
            "errors": [],
        }
