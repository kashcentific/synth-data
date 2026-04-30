"""
tool_registry.py — Standard metric tool registry.

Built-in implementations (registered at init):
  NLTK       : Flesch-Kincaid readability, Lexical Density, Stopword Ratio, Avg Sentence Length
  HF evaluate: Perplexity (GPT-2, reference-free), MAUVE (distribution quality, reference-free)
  deepeval   : Toxicity, Bias (LLM-as-judge, reference-free)

After any successful LLM code-generation in the evaluator, the resulting code is
wrapped and stored here under the metric name so that subsequent metrics with the
same (or similar) name skip the LLM call entirely.

Interface for every registered function:
    fn(df: pd.DataFrame, col: str) -> (value: float, anomalous: List[int], output: str)
"""

import math
import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

MetricFn = Callable[[pd.DataFrame, str], Tuple[float, List[int], str]]

# ── Optional library gates ─────────────────────────────────────────────────────

try:
    import nltk as _nltk_mod
    for _pkg, _path in [
        ("punkt",                      "tokenizers/punkt"),
        ("punkt_tab",                  "tokenizers/punkt_tab"),
        ("stopwords",                  "corpora/stopwords"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ]:
        try:
            _nltk_mod.data.find(_path)
        except LookupError:
            try:
                _nltk_mod.download(_pkg, quiet=True)
            except Exception:
                pass
    _NLTK = True
except ImportError:
    _NLTK = False

try:
    import evaluate as _hf_evaluate
    _HF_EVALUATE = True
except ImportError:
    _HF_EVALUATE = False

try:
    from deepeval.metrics import ToxicityMetric as _ToxicityMetric
    from deepeval.metrics import BiasMetric as _BiasMetric
    from deepeval.test_case import LLMTestCase as _LLMTestCase
    _DEEPEVAL = True
except ImportError:
    _DEEPEVAL = False


# ── Column helpers ────────────────────────────────────────────────────────────

def _get_texts(df: pd.DataFrame, col: str, max_n: int = 200) -> List[str]:
    if col and col in df.columns:
        return df[col].dropna().astype(str).tolist()[:max_n]
    # fallback: longest object column
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        return []
    best = max(obj_cols, key=lambda c: df[c].dropna().astype(str).str.len().mean())
    return df[best].dropna().astype(str).tolist()[:max_n]


# ── Registry class ────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry of pre-built and LLM-generated metric functions.

    Lookup order (inside evaluator):
      1. registry.get(metric_name)  →  exact or fuzzy match
      2. LLM code-gen               →  on miss or on registry function failure
      3. registry.register(...)     →  cache LLM result for future reuse
    """

    def __init__(self):
        self._registry: Dict[str, MetricFn] = {}
        self._sources:  Dict[str, str]       = {}
        self._build_all()

        print(f"\n[TOOL_REGISTRY] Initialized — {len(self._registry)} built-in tools")
        src_groups: Dict[str, List[str]] = {}
        for k, s in self._sources.items():
            src_groups.setdefault(s, []).append(k)
        for src, names in src_groups.items():
            print(f"[TOOL_REGISTRY]   [{src:<15}] {', '.join(names)}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, metric_name: str) -> Optional[MetricFn]:
        """Fuzzy lookup: exact normalized key, then substring match."""
        key = self._norm(metric_name)
        if key in self._registry:
            return self._registry[key]
        for k in self._registry:
            if k in key or key in k:
                return self._registry[k]
        return None

    def register(self, metric_name: str, fn: MetricFn, source: str = "llm_generated"):
        """Store a function (built-in or LLM-generated) under the normalized key."""
        key = self._norm(metric_name)
        self._registry[key] = fn
        self._sources[key]  = source
        print(f"[TOOL_REGISTRY] + Registered '{metric_name}' ({source})")

    def list_tools(self) -> List[Dict]:
        return [{"name": k, "source": self._sources.get(k, "?")} for k in sorted(self._registry)]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _norm(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "_", name.lower()).strip("_")

    def _add(self, name: str, fn: MetricFn, source: str):
        self._registry[self._norm(name)] = fn
        self._sources[self._norm(name)]  = source

    def _build_all(self):
        self._build_nltk()
        self._build_hf_evaluate()
        self._build_deepeval()

    # ==================================================================
    # NLTK tools
    # ==================================================================

    def _build_nltk(self):
        if not _NLTK:
            print("[TOOL_REGISTRY] NLTK not installed — NLTK tools skipped")
            return

        n0 = len(self._registry)

        # ── Flesch-Kincaid Reading Ease ────────────────────────────────
        def _flesch(df: pd.DataFrame, col: str) -> Tuple[float, List[int], str]:
            texts = _get_texts(df, col)
            if not texts:
                return 0.0, [], "No texts"
            scores: List[float] = []
            for tx in texts:
                try:
                    words  = _nltk_mod.word_tokenize(tx)
                    sents  = _nltk_mod.sent_tokenize(tx)
                    alpha  = [w for w in words if w.isalpha()]
                    if not alpha or not sents:
                        continue
                    sylls  = sum(max(len(re.findall(r"[aeiouAEIOU]", w)), 1) for w in alpha)
                    fre    = 206.835 - 1.015 * (len(alpha) / len(sents)) \
                                     - 84.6  * (sylls    / len(alpha))
                    scores.append(fre)
                except Exception:
                    pass
            if not scores:
                return 0.0, [], "Flesch: no valid sentences"
            mean_fre  = float(np.mean(scores))
            threshold = float(np.percentile(scores, 10))
            anomalous = [i for i, s in enumerate(scores) if s < threshold]
            out = (
                f"METRIC_VALUE: {mean_fre:.2f}\n"
                f"Flesch Reading Ease (mean): {mean_fre:.2f}\n"
                f"  Interpretation: 0-30=Very Hard  60-70=Standard  90-100=Very Easy\n"
                f"Range: [{min(scores):.1f}, {max(scores):.1f}]\n"
                f"Hard texts (bottom 10%): {len(anomalous)}"
            )
            return float(np.clip(mean_fre / 100.0, 0.0, 1.0)), anomalous, out

        for alias in ("flesch_reading_ease", "flesch_kincaid", "readability", "reading_ease"):
            self._add(alias, _flesch, "nltk")

        # ── Lexical Density ────────────────────────────────────────────
        CONTENT_TAGS = {
            "NN","NNS","NNP","NNPS",
            "VB","VBD","VBG","VBN","VBP","VBZ",
            "JJ","JJR","JJS",
            "RB","RBR","RBS",
        }

        def _lex_density(df: pd.DataFrame, col: str) -> Tuple[float, List[int], str]:
            texts = _get_texts(df, col, max_n=100)
            if not texts:
                return 0.0, [], "No texts"
            densities: List[float] = []
            for tx in texts:
                try:
                    tokens = _nltk_mod.word_tokenize(tx)
                    tagged = _nltk_mod.pos_tag(tokens)
                    content = sum(1 for _, tag in tagged if tag in CONTENT_TAGS)
                    densities.append(content / max(len(tokens), 1))
                except Exception:
                    densities.append(0.0)
            if not densities:
                return 0.0, [], "POS tagging failed"
            mean_ld   = float(np.mean(densities))
            threshold = float(np.percentile(densities, 10))
            anomalous = [i for i, d in enumerate(densities) if d < threshold]
            out = (
                f"METRIC_VALUE: {mean_ld:.4f}\n"
                f"Lexical Density (content words / total words): {mean_ld:.4f}\n"
                f"Low-content texts (bottom 10%): {len(anomalous)}"
            )
            return mean_ld, anomalous, out

        for alias in ("lexical_density", "content_density", "content_word_ratio"):
            self._add(alias, _lex_density, "nltk")

        # ── Stopword Ratio ─────────────────────────────────────────────
        def _stopword_ratio(df: pd.DataFrame, col: str) -> Tuple[float, List[int], str]:
            texts = _get_texts(df, col)
            if not texts:
                return 0.0, [], "No texts"
            try:
                from nltk.corpus import stopwords as _sw
                sw_set = set(_sw.words("english"))
            except Exception:
                return 0.0, [], "NLTK stopwords unavailable"
            ratios: List[float] = []
            for tx in texts:
                try:
                    tokens = [w.lower() for w in _nltk_mod.word_tokenize(tx) if w.isalpha()]
                    if not tokens:
                        continue
                    ratios.append(sum(1 for t in tokens if t in sw_set) / len(tokens))
                except Exception:
                    pass
            if not ratios:
                return 0.0, [], "No valid texts"
            mean_sr   = float(np.mean(ratios))
            threshold = float(np.percentile(ratios, 85))
            anomalous = [i for i, r in enumerate(ratios) if r > threshold]
            out = (
                f"METRIC_VALUE: {mean_sr:.4f}\n"
                f"Stopword Ratio: {mean_sr:.4f}\n"
                f"High-stopword texts (function-word heavy, top 15%): {len(anomalous)}"
            )
            return 1.0 - mean_sr, anomalous, out

        self._add("stopword_ratio", _stopword_ratio, "nltk")

        # ── Average Sentence Length ────────────────────────────────────
        def _avg_sent_len(df: pd.DataFrame, col: str) -> Tuple[float, List[int], str]:
            texts = _get_texts(df, col)
            if not texts:
                return 0.0, [], "No texts"
            per_text_lens: List[float] = []
            for tx in texts:
                try:
                    sents = _nltk_mod.sent_tokenize(tx)
                    lens  = [len(_nltk_mod.word_tokenize(s)) for s in sents if s.strip()]
                    per_text_lens.append(float(np.mean(lens)) if lens else 0.0)
                except Exception:
                    pass
            if not per_text_lens:
                return 0.0, [], "No sentences"
            mean_len  = float(np.mean(per_text_lens))
            threshold = float(np.percentile(per_text_lens, 90))
            anomalous = [i for i, l in enumerate(per_text_lens) if l > threshold]
            # score: 15-20 words/sentence is readable; penalise extremes
            score = float(np.clip(1.0 - abs(mean_len - 17.0) / 50.0, 0.0, 1.0))
            out = (
                f"METRIC_VALUE: {mean_len:.1f}\n"
                f"Average Sentence Length: {mean_len:.1f} words/sentence\n"
                f"Range: [{min(per_text_lens):.1f}, {max(per_text_lens):.1f}]\n"
                f"Very-long-sentence texts (top 10%): {len(anomalous)}"
            )
            return score, anomalous, out

        for alias in ("average_sentence_length", "sentence_length", "avg_sentence_len"):
            self._add(alias, _avg_sent_len, "nltk")

        print(f"[TOOL_REGISTRY] NLTK: {len(self._registry) - n0} tools registered")

    # ==================================================================
    # HuggingFace evaluate tools
    # ==================================================================

    def _build_hf_evaluate(self):
        if not _HF_EVALUATE:
            print("[TOOL_REGISTRY] `evaluate` not installed — HF evaluate tools skipped")
            return

        n0 = len(self._registry)

        # ── Perplexity (GPT-2, reference-free) ────────────────────────
        try:
            _ppl = _hf_evaluate.load("perplexity", module_type="metric")

            def _perplexity(df: pd.DataFrame, col: str) -> Tuple[float, List[int], str]:
                texts = _get_texts(df, col, max_n=50)  # GPT-2 is slow; cap at 50
                if not texts:
                    return 0.0, [], "No texts"
                try:
                    result = _ppl.compute(predictions=texts, model_id="gpt2")
                    ppls   = result["perplexities"]
                    mean_p = float(np.mean(ppls))
                    p85    = float(np.percentile(ppls, 85))
                    anomalous = [i for i, p in enumerate(ppls) if p > p85]
                    # lower perplexity = more fluent; invert-normalize
                    score = float(np.clip(1.0 / (1.0 + mean_p / 200.0), 0.0, 1.0))
                    out = (
                        f"METRIC_VALUE: {mean_p:.2f}\n"
                        f"Mean Perplexity (GPT-2): {mean_p:.2f}\n"
                        f"  Lower = more fluent / predictable text\n"
                        f"Range: [{min(ppls):.1f}, {max(ppls):.1f}]\n"
                        f"High-perplexity texts (p85+): {len(anomalous)}"
                    )
                    return score, anomalous, out
                except Exception as exc:
                    return 0.0, [], f"Perplexity compute failed: {exc}"

            for alias in ("perplexity", "text_perplexity", "language_model_score", "fluency_perplexity"):
                self._add(alias, _perplexity, "hf_evaluate")
            print("[TOOL_REGISTRY] HF evaluate: perplexity OK")
        except Exception as exc:
            print(f"[TOOL_REGISTRY] HF evaluate perplexity: FAILED ({exc})")

        # ── MAUVE (reference-free distribution quality) ────────────────
        try:
            _mauve = _hf_evaluate.load("mauve")

            def _mauve_metric(df: pd.DataFrame, col: str) -> Tuple[float, List[int], str]:
                texts = _get_texts(df, col, max_n=100)
                if len(texts) < 10:
                    return 0.0, [], "Need ≥10 texts for MAUVE"
                try:
                    half   = len(texts) // 2
                    result = _mauve.compute(
                        predictions=texts[:half],
                        references =texts[half:],
                        featurize_model_name="gpt2",
                    )
                    score = float(result.mauve)
                    out = (
                        f"METRIC_VALUE: {score:.4f}\n"
                        f"MAUVE Score: {score:.4f}\n"
                        f"  1.0 = identical distribution, 0.0 = maximal divergence\n"
                        f"  Compares first-half vs second-half of the dataset"
                    )
                    return score, [], out
                except Exception as exc:
                    return 0.0, [], f"MAUVE compute failed: {exc}"

            for alias in ("mauve", "distribution_mauve", "mauve_score", "text_distribution_quality"):
                self._add(alias, _mauve_metric, "hf_evaluate")
            print("[TOOL_REGISTRY] HF evaluate: mauve OK")
        except Exception as exc:
            print(f"[TOOL_REGISTRY] HF evaluate mauve: FAILED ({exc})")

        print(f"[TOOL_REGISTRY] HF evaluate: {len(self._registry) - n0} tools registered")

    # ==================================================================
    # deepeval tools (LLM-as-judge, reference-free)
    # ==================================================================

    def _build_deepeval(self):
        if not _DEEPEVAL:
            print("[TOOL_REGISTRY] deepeval not installed — deepeval tools skipped")
            return

        n0 = len(self._registry)

        # ── Toxicity (deepeval ToxicityMetric) ────────────────────────
        try:
            _tox = _ToxicityMetric(threshold=0.5)

            def _deepeval_tox(df: pd.DataFrame, col: str) -> Tuple[float, List[int], str]:
                texts = _get_texts(df, col, max_n=20)  # LLM calls are expensive
                if not texts:
                    return 1.0, [], "No texts"
                scores, anomalous = [], []
                for i, tx in enumerate(texts):
                    try:
                        tc = _LLMTestCase(input=tx, actual_output=tx)
                        _tox.measure(tc)
                        s = float(_tox.score)
                        scores.append(s)
                        if s < 0.5:   # score < 0.5 = toxic in deepeval convention
                            anomalous.append(i)
                    except Exception:
                        scores.append(1.0)
                mean_s = float(np.mean(scores)) if scores else 1.0
                out = (
                    f"METRIC_VALUE: {mean_s:.4f}\n"
                    f"deepeval Toxicity Score (mean): {mean_s:.4f}\n"
                    f"  1.0 = non-toxic, 0.0 = highly toxic (LLM judge)\n"
                    f"Flagged toxic texts: {len(anomalous)}/{len(texts)}"
                )
                return mean_s, anomalous, out

            for alias in ("deepeval_toxicity", "toxicity_deepeval", "llm_toxicity"):
                self._add(alias, _deepeval_tox, "deepeval")
            print("[TOOL_REGISTRY] deepeval: ToxicityMetric OK")
        except Exception as exc:
            print(f"[TOOL_REGISTRY] deepeval ToxicityMetric: FAILED ({exc})")

        # ── Bias (deepeval BiasMetric) ─────────────────────────────────
        try:
            _bias = _BiasMetric(threshold=0.5)

            def _deepeval_bias(df: pd.DataFrame, col: str) -> Tuple[float, List[int], str]:
                texts = _get_texts(df, col, max_n=20)
                if not texts:
                    return 1.0, [], "No texts"
                scores, anomalous = [], []
                for i, tx in enumerate(texts):
                    try:
                        tc = _LLMTestCase(input=tx, actual_output=tx)
                        _bias.measure(tc)
                        s = float(_bias.score)
                        scores.append(s)
                        if s < 0.5:
                            anomalous.append(i)
                    except Exception:
                        scores.append(1.0)
                mean_s = float(np.mean(scores)) if scores else 1.0
                out = (
                    f"METRIC_VALUE: {mean_s:.4f}\n"
                    f"deepeval Bias Score (mean): {mean_s:.4f}\n"
                    f"  1.0 = unbiased, 0.0 = highly biased (LLM judge)\n"
                    f"Flagged biased texts: {len(anomalous)}/{len(texts)}"
                )
                return mean_s, anomalous, out

            for alias in ("deepeval_bias", "bias_deepeval", "llm_bias", "bias_score"):
                self._add(alias, _deepeval_bias, "deepeval")
            print("[TOOL_REGISTRY] deepeval: BiasMetric OK")
        except Exception as exc:
            print(f"[TOOL_REGISTRY] deepeval BiasMetric: FAILED ({exc})")

        print(f"[TOOL_REGISTRY] deepeval: {len(self._registry) - n0} tools registered")
