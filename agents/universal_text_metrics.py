"""
universal_text_metrics.py

Predefined, reference-free text quality metrics.
Applied automatically whenever the dataset is detected as text-based.

Categories (all ground-truth-free, continuous scores — no binary classifiers):
  Validity   — USL-H (unified semantic-linguistic: TTR × TF-IDF specificity × language richness)
  Fidelity   — EDS (embedding distribution similarity), RUBER-Unreferenced coherence
  Diversity  — Self-Cosine Similarity, TTR, Distinct-N, Response Entropy (Ent-n)

Removed metrics (produced degenerate 0/1 on non-social-media corpora):
  GAR  — CoLA BERT classifier trained on English sentences; outputs 0 or 1 uniformly on code/trajectories
  TP, EMT, NT2T — Toxic-BERT classifiers; always return 0 on code data (wrong domain for data quality)
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── Optional dependencies — all degrade gracefully ──────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# ── Text-dataset detection vocabulary ────────────────────────────────────────
_TEXT_SIGNALS = {
    "text", "nlp", "dialogue", "conversational", "language", "chat", "document",
    "qa", "question", "answer", "summarization", "translation", "generation",
    "agent", "trajectory", "story", "review", "corpus", "instruction", "prompt",
    "response", "reasoning", "software", "engineering", "benchmark",
}
_NON_TEXT_SIGNALS = {
    "image", "audio", "video", "numerical", "time_series", "mathematical",
    "tabular_numeric", "sensor",
}


class UniversalTextMetrics:
    """
    Computes the four universal text-quality metric categories for any text dataset.
    All metrics are reference-free (no ground truth, no human annotators required).
    """

    def __init__(self):
        self._st_model      = None
        self._models_loaded = False

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load_models(self):
        if self._models_loaded:
            return
        print("\n[UNIVERSAL] Loading text metric models (lazy)...")

        if _ST_AVAILABLE:
            try:
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                print("[UNIVERSAL]   * SentenceTransformer (all-MiniLM-L6-v2): OK")
            except Exception as exc:
                print(f"[UNIVERSAL]   * SentenceTransformer: FAILED ({exc})")
        else:
            print("[UNIVERSAL]   * SentenceTransformer: not installed — embedding metrics skipped")

        self._models_loaded = True

    # ------------------------------------------------------------------
    # Dataset type detection
    # ------------------------------------------------------------------

    def is_text_dataset(self, thinker_output: Dict) -> bool:
        """
        Returns True if the thinker classified this as a text/NLP/agent dataset.
        Checks dataset_type + domain name; defaults to True for unknown types
        since the evaluator already has text-column detection as a fallback.
        """
        dtype  = thinker_output.get("dataset_type", "").lower()
        domain = thinker_output.get("domain", {}).get("name", "").lower()
        blob   = f"{dtype} {domain}"

        if any(s in blob for s in _NON_TEXT_SIGNALS) and not any(s in blob for s in _TEXT_SIGNALS):
            return False

        if any(s in blob for s in _TEXT_SIGNALS):
            return True

        # Ambiguous — let the evaluator's text-column detector decide;
        # return True so metrics are attempted (they degrade gracefully).
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pick_primary_col(self, df: pd.DataFrame, text_cols: List[str]) -> Optional[str]:
        """Column with highest average text length — most information-rich."""
        if not text_cols:
            return None
        return max(
            text_cols,
            key=lambda c: df[c].dropna().astype(str).str.len().mean() if c in df.columns else 0,
        )

    def _get_texts(self, df: pd.DataFrame, col: str, max_n: int = 200) -> List[str]:
        return df[col].dropna().astype(str).tolist()[:max_n]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())


    # ------------------------------------------------------------------
    # Result / skip factories
    # ------------------------------------------------------------------

    def _result(
        self,
        name: str,
        category: str,
        col: str,
        description: str,
        metric_value: float,
        anomalous_indices: List[int],
        output: str,
    ) -> Dict:
        return {
            "metric_name":              name,
            "metric_type":              f"universal_{category}",
            "metric_category":          category,
            "metric_source":            "universal",
            "target_column":            col,
            "description":              description,
            "reasoning":                f"Predefined universal {category} metric for text datasets — no ground truth required.",
            "source_influence":         "Universal Text Metrics (built-in)",
            "feasibility_score":        1.0,
            "quality_score":            float(np.clip(metric_value, 0.0, 1.0)),
            "execution_output":         output,
            "generated_code":           "# Universal metric — direct computation",
            "error":                    None,
            "anomalous_row_count":      len(anomalous_indices),
            "anomalous_row_indices":    anomalous_indices[:20],
            "anomalous_row_narratives": [],
            "anomalous_samples":        [],
            "metric_interpretation":    description,
            "retry_count":              0,
            "computation_strategy":     "universal_predefined",
            "paper_citation":           {},
        }

    def _skip(self, name: str, category: str, reason: str) -> Dict:
        return {
            "metric_name":              name,
            "metric_type":              f"universal_{category}",
            "metric_category":          category,
            "metric_source":            "universal",
            "target_column":            "",
            "description":              f"Universal {category} metric — skipped ({reason})",
            "reasoning":                f"Predefined universal {category} metric.",
            "source_influence":         "Universal Text Metrics (built-in)",
            "feasibility_score":        0.0,
            "quality_score":            0.0,
            "execution_output":         f"SKIPPED: {reason}",
            "generated_code":           "# Skipped",
            "error":                    reason,
            "anomalous_row_count":      0,
            "anomalous_row_indices":    [],
            "anomalous_row_narratives": [],
            "anomalous_samples":        [],
            "metric_interpretation":    f"Skipped: {reason}",
            "retry_count":              0,
            "computation_strategy":     "universal_predefined",
            "paper_citation":           {},
        }

    # ==================================================================
    # VALIDITY
    # ==================================================================

    def _compute_usl_h(self, texts: List[str], col: str) -> Dict:
        """
        USL-H — Unified Semantic Linguistic Score (H variant).
        H = harmonic mean of:
          U = uniqueness (TTR across corpus)
          S = specificity (mean max TF-IDF weight per doc)
          L = language quality (CoLA GAR score; fallback: fraction of texts > 10 tokens)
        """
        name = "USL_H_Score"
        if not texts or not _SKLEARN_AVAILABLE:
            return self._skip(name, "validity", "sklearn unavailable or no texts")
        try:
            # U: corpus-level TTR
            all_tokens = [t for tx in texts for t in self._tokenize(tx)]
            u = len(set(all_tokens)) / max(len(all_tokens), 1)

            # S: mean max TF-IDF weight per document
            vec   = TfidfVectorizer(max_features=1000, stop_words="english")
            tfidf = vec.fit_transform(texts)
            per_doc_max = np.asarray(tfidf.max(axis=1).todense()).flatten()
            s = float(per_doc_max.mean())

            # L: language richness — fraction of texts with > 10 meaningful tokens
            l = sum(1 for tx in texts if len(self._tokenize(tx)) > 10) / max(len(texts), 1)

            # Harmonic mean of U, S, L
            components = [u, s, l]
            usl_h = (len(components) / sum(1.0 / max(c, 1e-9) for c in components))

            threshold = float(np.percentile(per_doc_max, 10))
            anomalous = [i for i, v in enumerate(per_doc_max) if v < threshold]

            output = (
                f"METRIC_VALUE: {usl_h:.4f}\n"
                f"USL-H Score: {usl_h:.4f}\n"
                f"  U (corpus TTR / uniqueness):  {u:.4f}\n"
                f"  S (mean max TF-IDF weight):   {s:.4f}\n"
                f"  L (lang richness >10 tokens): {l:.4f}\n"
                f"Low-specificity threshold (p10): {threshold:.4f}\n"
                f"Low-specificity texts: {len(anomalous)}"
            )
            return self._result(name, "validity", col,
                f"USL-H={usl_h:.4f} — harmonic mean of uniqueness ({u:.3f}), "
                f"specificity ({s:.3f}), and language richness ({l:.3f}).",
                usl_h, anomalous, output)
        except Exception as exc:
            return self._skip(name, "validity", str(exc))

    # ==================================================================
    # FIDELITY
    # ==================================================================

    def _compute_eds(self, texts: List[str], col: str) -> Dict:
        """
        EDS — Embedding Distribution Similarity.
        Mean cosine similarity of each sentence embedding to the corpus centroid.
        High EDS = coherent, tight embedding distribution (good fidelity to a topic).
        Reports std-dev and flags degenerate cases (< 10 texts or std < 0.001).
        """
        name = "EDS_Embedding_Distribution_Similarity"
        if not self._st_model or not texts:
            return self._skip(name, "fidelity", "SentenceTransformer unavailable")
        if len(texts) < 5:
            return self._skip(name, "fidelity", f"Too few texts ({len(texts)}) for reliable EDS")
        try:
            sample   = texts[:150]
            emb      = self._st_model.encode(sample, show_progress_bar=False, batch_size=32)
            centroid = emb.mean(axis=0, keepdims=True)

            norms  = np.linalg.norm(emb, axis=1, keepdims=True)
            c_norm = np.linalg.norm(centroid)
            sims   = (emb @ centroid.T).flatten() / (norms.flatten() * c_norm + 1e-9)

            eds   = float(np.mean(sims))
            std   = float(np.std(sims))

            # Degenerate case: all embeddings essentially identical direction
            if std < 0.001:
                return self._skip(
                    name, "fidelity",
                    f"Degenerate: std={std:.5f} — all embeddings near-identical "
                    f"(corpus too homogeneous or sample too small to discriminate)"
                )

            threshold = float(np.percentile(sims, 15))
            anomalous = [i for i, s in enumerate(sims) if s < threshold]

            output = (
                f"METRIC_VALUE: {eds:.4f}\n"
                f"EDS (mean cosine to centroid): {eds:.4f}\n"
                f"Std dev of similarities: {std:.4f}\n"
                f"Low-similarity threshold (p15): {threshold:.4f}\n"
                f"Outlier texts (below threshold): {len(anomalous)}/{len(sample)}"
            )
            return self._result(name, "fidelity", col,
                f"EDS={eds:.4f} (std={std:.4f}) — mean cosine similarity to corpus centroid. "
                f"{len(anomalous)} texts are far from the distribution centroid.",
                eds, anomalous, output)
        except Exception as exc:
            return self._skip(name, "fidelity", str(exc))

    def _compute_ruber_u(
        self, texts: List[str], col: str, df: pd.DataFrame, text_cols: List[str]
    ) -> Dict:
        """
        RUBER-Unreferenced — coherence without ground truth.
        If a context column exists (e.g. problem_statement → trajectory):
          score = mean cosine sim between context and response embeddings.
        Else: mean cosine similarity between consecutive texts.
        """
        name = "RUBER_Unreferenced_Coherence"
        if not self._st_model or len(texts) < 2:
            return self._skip(name, "fidelity", "SentenceTransformer unavailable or < 2 texts")
        try:
            other_cols = [c for c in text_cols if c != col]

            if other_cols:
                ctx_col    = other_cols[0]
                ctx_texts  = df[ctx_col].dropna().astype(str).tolist()[:100]
                resp_texts = texts[:len(ctx_texts)]

                ctx_emb  = self._st_model.encode(ctx_texts,  show_progress_bar=False, batch_size=32)
                resp_emb = self._st_model.encode(resp_texts, show_progress_bar=False, batch_size=32)

                nc = np.linalg.norm(ctx_emb,  axis=1)
                nr = np.linalg.norm(resp_emb, axis=1)
                sims = (ctx_emb * resp_emb).sum(axis=1) / (nc * nr + 1e-9)
                mode = f"context-response ({ctx_col} → {col})"
            else:
                sample = texts[:100]
                emb = self._st_model.encode(sample, show_progress_bar=False, batch_size=32)
                sims = np.array([
                    float((emb[i] @ emb[i+1]) /
                          (np.linalg.norm(emb[i]) * np.linalg.norm(emb[i+1]) + 1e-9))
                    for i in range(len(emb) - 1)
                ])
                mode = "adjacent-pair"

            ruber     = float(np.mean(sims))
            threshold = float(np.percentile(sims, 15))
            anomalous = [i for i, s in enumerate(sims) if s < threshold]

            output = (
                f"METRIC_VALUE: {ruber:.4f}\n"
                f"RUBER-U Coherence [{mode}]: {ruber:.4f}\n"
                f"Low-coherence threshold (p15): {threshold:.4f}\n"
                f"Low-coherence pairs: {len(anomalous)}"
            )
            return self._result(name, "fidelity", col,
                f"RUBER-U={ruber:.4f} ({mode}). Measures embedding coherence without ground truth.",
                ruber, anomalous, output)
        except Exception as exc:
            return self._skip(name, "fidelity", str(exc))

    # ==================================================================
    # DIVERSITY
    # ==================================================================

    def _compute_self_cosine(self, texts: List[str], col: str) -> Dict:
        """
        Self-Cosine Similarity — mean pairwise cosine over a sample of pairs.
        Lower = more diverse corpus. Diversity score = 1 - self_cosine.
        """
        name = "Self_Cosine_Similarity"
        if not self._st_model or not texts:
            return self._skip(name, "diversity", "SentenceTransformer unavailable")
        try:
            import random
            sample = texts[:30]  # Reduced from 100 for speed
            emb    = self._st_model.encode(sample, show_progress_bar=False, batch_size=16)
            n      = len(emb)

            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            pairs     = random.sample(all_pairs, min(200, len(all_pairs)))  # Reduced from 500

            sims = [
                float((emb[i] @ emb[j]) /
                      (np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]) + 1e-9))
                for i, j in pairs
            ]
            mean_sim  = float(np.mean(sims))
            diversity = 1.0 - mean_sim

            # Flag texts most similar to their neighbours (least diverse)
            per_text = [
                float(np.mean([
                    (emb[i] @ emb[j]) /
                    (np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]) + 1e-9)
                    for j in range(min(5, n)) if j != i  # Reduced from 10
                ]))
                for i in range(n)
            ]
            threshold = float(np.percentile(per_text, 85))
            anomalous = [i for i, s in enumerate(per_text) if s > threshold]

            output = (
                f"METRIC_VALUE: {mean_sim:.4f}\n"
                f"Mean pairwise cosine similarity: {mean_sim:.4f}\n"
                f"Diversity score (1 - sim):        {diversity:.4f}\n"
                f"Sampled {len(pairs)} pairs from {n} texts\n"
                f"Near-duplicate / low-diversity texts: {len(anomalous)}"
            )
            return self._result(name, "diversity", col,
                f"Self-cosine={mean_sim:.4f}, diversity={diversity:.4f}. "
                f"{len(anomalous)} texts are near-duplicates of their neighbours.",
                diversity, anomalous, output)
        except Exception as exc:
            return self._skip(name, "diversity", str(exc))

    def _compute_ttr(self, texts: List[str], col: str) -> Dict:
        """TTR — Type-Token Ratio per text; mean across corpus."""
        name = "TTR_Type_Token_Ratio"
        if not texts:
            return self._skip(name, "diversity", "No texts")
        try:
            ttrs = []
            for tx in texts:
                tokens = self._tokenize(tx)
                ttrs.append(len(set(tokens)) / max(len(tokens), 1))

            mean_ttr  = float(np.mean(ttrs))
            threshold = float(np.percentile(ttrs, 10))
            anomalous = [i for i, t in enumerate(ttrs) if t < threshold]

            output = (
                f"METRIC_VALUE: {mean_ttr:.4f}\n"
                f"Mean TTR (per text): {mean_ttr:.4f}\n"
                f"Std dev: {float(np.std(ttrs)):.4f}\n"
                f"Low-TTR threshold (p10): {threshold:.4f}\n"
                f"Repetitive texts: {len(anomalous)}"
            )
            return self._result(name, "diversity", col,
                f"TTR={mean_ttr:.4f} — mean type-token ratio per text. "
                f"{len(anomalous)} texts are lexically repetitive.",
                mean_ttr, anomalous, output)
        except Exception as exc:
            return self._skip(name, "diversity", str(exc))

    def _compute_distinct_n(self, texts: List[str], col: str) -> Dict:
        """Distinct-N: fraction of unique unigrams (D1) and bigrams (D2) across corpus."""
        name = "Distinct_N_Unique_Ngrams"
        if not texts:
            return self._skip(name, "diversity", "No texts")
        try:
            unigrams, bigrams = [], []
            for tx in texts:
                toks = self._tokenize(tx)
                unigrams.extend(toks)
                bigrams.extend(zip(toks, toks[1:]))

            d1 = len(set(unigrams)) / max(len(unigrams), 1)
            d2 = len(set(bigrams))  / max(len(bigrams),  1)

            per_text_d = [
                len(set(self._tokenize(tx))) / max(len(self._tokenize(tx)), 1)
                for tx in texts
            ]
            threshold = float(np.percentile(per_text_d, 10)) if per_text_d else 0.0
            anomalous = [i for i, d in enumerate(per_text_d) if d < threshold]

            output = (
                f"METRIC_VALUE: {(d1 + d2) / 2:.4f}\n"
                f"Distinct-1 (unique unigram fraction): {d1:.4f}  "
                f"({len(set(unigrams))} unique / {len(unigrams)} total)\n"
                f"Distinct-2 (unique bigram fraction):  {d2:.4f}  "
                f"({len(set(bigrams))} unique / {len(bigrams)} total)\n"
                f"Repetitive texts (low distinctness): {len(anomalous)}"
            )
            return self._result(name, "diversity", col,
                f"Distinct-1={d1:.4f}, Distinct-2={d2:.4f}. "
                f"Higher values indicate richer lexical variety.",
                (d1 + d2) / 2, anomalous, output)
        except Exception as exc:
            return self._skip(name, "diversity", str(exc))

    def _compute_response_entropy(self, texts: List[str], col: str) -> Dict:
        """Ent-N: Shannon entropy of unigram and bigram distributions (normalised)."""
        name = "Response_Entropy_Ent_N"
        if not texts:
            return self._skip(name, "diversity", "No texts")
        try:
            all_tokens  = [t for tx in texts for t in self._tokenize(tx)]
            all_bigrams = list(zip(all_tokens, all_tokens[1:]))

            def entropy(ngrams):
                if not ngrams:
                    return 0.0
                counts = Counter(ngrams)
                total  = sum(counts.values())
                return -sum((c / total) * math.log2(c / total + 1e-12) for c in counts.values())

            ent1 = entropy(all_tokens)
            ent2 = entropy(all_bigrams)

            vocab     = len(set(all_tokens))
            max_ent   = math.log2(vocab) if vocab > 1 else 1.0
            norm_ent1 = ent1 / max_ent

            output = (
                f"METRIC_VALUE: {norm_ent1:.4f}\n"
                f"Ent-1 (unigram entropy): {ent1:.4f} bits  "
                f"(normalised: {norm_ent1:.4f})\n"
                f"Ent-2 (bigram entropy):  {ent2:.4f} bits\n"
                f"Vocabulary: {vocab} unique tokens across {len(all_tokens)} total"
            )
            return self._result(name, "diversity", col,
                f"Ent-1={ent1:.2f}bits (norm={norm_ent1:.4f}), Ent-2={ent2:.2f}bits. "
                f"Higher entropy indicates more varied language use.",
                norm_ent1, [], output)
        except Exception as exc:
            return self._skip(name, "diversity", str(exc))

    # ==================================================================
    # Entry point
    # ==================================================================

    def compute_all(
        self, df: pd.DataFrame, text_cols: List[str], thinker_output: Dict
    ) -> List[Dict]:
        """
        Run all four metric categories. Returns results in per_metric_results format.
        Skipped metrics are included (with error field set) so they appear in the report.
        """
        self._load_models()

        if not text_cols:
            print("[UNIVERSAL] No text columns — skipping universal metrics.")
            return []

        primary = self._pick_primary_col(df, text_cols)
        print(f"[UNIVERSAL] Primary column for universal metrics: '{primary}'")

        texts = self._get_texts(df, primary, max_n=200)
        print(f"[UNIVERSAL] Sample size: {len(texts)} texts")

        results: List[Dict] = []

        def _run(r: Dict) -> Dict:
            """Append result and print a one-line summary immediately."""
            val_line = ""
            for line in (r.get("execution_output") or "").splitlines():
                if line.startswith("METRIC_VALUE:"):
                    val_line = line.replace("METRIC_VALUE:", "").strip()
                    break
            err = r.get("error")
            if err:
                status = f"SKIP  — {err[:80]}"
            elif val_line:
                status = f"score = {val_line}"
            else:
                status = "OK"
            print(f"[UNIVERSAL]   {r['metric_name']:<50} {status}")
            results.append(r)
            return r

        print("\n[UNIVERSAL] ── VALIDITY ──────────────────────────────────")
        _run(self._compute_usl_h(texts, primary))

        print("\n[UNIVERSAL] ── FIDELITY ──────────────────────────────────")
        _run(self._compute_eds(texts, primary))
        _run(self._compute_ruber_u(texts, primary, df, text_cols))

        print("\n[UNIVERSAL] ── DIVERSITY ─────────────────────────────────")
        _run(self._compute_self_cosine(texts, primary))
        _run(self._compute_ttr(texts, primary))
        _run(self._compute_distinct_n(texts, primary))
        _run(self._compute_response_entropy(texts, primary))

        ok      = sum(1 for r in results if not r.get("error"))
        skipped = len(results) - ok
        print(f"\n[UNIVERSAL] Done — {ok} computed, {skipped} skipped.\n")

        return results
