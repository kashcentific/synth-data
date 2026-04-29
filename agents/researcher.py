import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List, Tuple

from base import BaseAgent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper


class ResearchAgent(BaseAgent):
    """
    Agentic researcher that iterates until it produces a well-balanced set of
    metrics covering BOTH mathematical/statistical AND semantic/text-quality types.

    Graph routing: if coverage gaps remain after Phase 4, returns researcher_retry=True
    so the conditional edge in graph.py loops back here for a targeted gap-filling pass.
    Max 3 total iterations (initial + 2 retries).
    """

    def __init__(self, model: str | None = None, temperature: float | None = None):
        kwargs = {}
        if model is not None:
            kwargs["model"] = model
        if temperature is not None:
            kwargs["temperature"] = temperature
        super().__init__(**kwargs)

        print("\n[RESEARCHER] Initializing research tools...")

        try:
            print("[RESEARCHER]   * DuckDuckGo: ", end="", flush=True)
            self.search_tool = DuckDuckGoSearchRun()
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            self.search_tool = None

        try:
            print("[RESEARCHER]   * Wikipedia: ", end="", flush=True)
            self.wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            self.wikipedia = None

        try:
            print("[RESEARCHER]   * Arxiv: ", end="", flush=True)
            self.arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            self.arxiv = None

    # ------------------------------------------------------------------
    # Tool runner
    # ------------------------------------------------------------------

    def _safe_run(self, tool, query: str, label: str, char_limit: int = 1500) -> Tuple[str, str]:
        print(f"\n[RESEARCHER]   {label}")
        print(f"[RESEARCHER]      Query : \"{query}\"")

        if tool is None:
            print(f"[RESEARCHER]      WARNING: Tool not available.")
            return f"[{label}] Tool not initialised.", ""

        try:
            result = tool.run(query)
            excerpt = result[:300].replace("\n", " ").strip()
            print(f"[RESEARCHER]      OK: {len(result)} chars")
            print(f"[RESEARCHER]      Preview : {excerpt!r}")
            return result[:char_limit], excerpt
        except Exception as exc:
            print(f"[RESEARCHER]      ERROR: {exc}")
            return f"[{label}] Failed: {exc}", ""

    # ------------------------------------------------------------------
    # Initial context gathering
    # ------------------------------------------------------------------

    def _gather_context(self, domain: str, dataset_type: str, column_summary: str) -> Dict[str, Any]:
        search_query = f"{domain} {dataset_type} synthetic dataset quality evaluation metrics"
        arxiv_query  = f"synthetic data evaluation metrics {domain} machine learning fidelity"
        wiki_query   = f"{domain} machine learning dataset evaluation benchmark"

        web_full,   web_excerpt   = self._safe_run(self.search_tool, search_query, "DuckDuckGo Web Search")
        wiki_full,  wiki_excerpt  = self._safe_run(self.wikipedia,   wiki_query,   "Wikipedia",    char_limit=1200)
        arxiv_full, arxiv_excerpt = self._safe_run(self.arxiv,       arxiv_query,  "Arxiv Papers", char_limit=2000)

        combined_text = (
            f"## Web Search Results\n{web_full}\n\n"
            f"## Wikipedia Summary\n{wiki_full}\n\n"
            f"## Arxiv Research Papers\n{arxiv_full}\n\n"
            f"## Dataset Column Profile\n{column_summary}"
        )

        sources = {
            "duckduckgo": {"query": search_query, "excerpt": web_excerpt},
            "wikipedia":  {"query": wiki_query,   "excerpt": wiki_excerpt},
            "arxiv":      {"query": arxiv_query,  "excerpt": arxiv_excerpt},
        }

        return {"text": combined_text, "sources": sources}

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def build_prompt(self, thinker_output: Dict[str, Any], external_context: str) -> str:
        domain       = thinker_output.get("domain", {}).get("name", "general")
        dataset_type = thinker_output.get("dataset_type", "tabular")

        return f"""You are the Research Agent in a Data Quality Audit System.

You have gathered real research context from DuckDuckGo, Wikipedia, and Arxiv (see below).
Your job: propose 15 concrete, COMPUTABLE data quality evaluation metrics for this dataset.

SCOPE — DATA QUALITY ONLY:
This pipeline evaluates DATA QUALITY. Do NOT propose metrics related to:
- Privacy, PII detection, or anonymisation
- Security vulnerabilities or access control
- Regulatory compliance (GDPR, HIPAA, etc.)
- Sensitive information detection
Those concerns are handled by a separate Data Governance pipeline.

REQUIRED COVERAGE — include BOTH types:
- At least 4 mathematical/statistical metrics (entropy, KL-divergence, statistical tests, distributions, consistency scores)
- At least 4 semantic/text-quality metrics (semantic coherence, embedding similarity, text readability, NLP-based quality)

Dataset Domain  : {domain}
Dataset Type    : {dataset_type}

Full Thinker Profile:
{json.dumps(thinker_output, indent=2)}

External Research Context:
{external_context}

For EVERY metric:
- "source_influence": which tool (DuckDuckGo/Wikipedia/Arxiv) influenced this metric
- "execution_hint": concrete Python/pandas/scipy/sklearn steps with actual column names

Return ONLY valid JSON — no markdown:

{{
  "proposed_metrics": [
    {{
      "metric_name": "<short name>",
      "metric_type": "<distribution|label_noise|semantic_consistency|text_quality|domain_specific|utility|statistical|other>",
      "description": "<what this metric measures>",
      "reasoning": "<why this metric matters for data quality of this specific dataset>",
      "source_influence": "<which tool returned what finding>",
      "execution_hint": "<concrete Python/pandas/scipy steps with actual column names>"
    }}
  ],
  "research_summary": "<2-3 sentence summary citing the most relevant findings>"
}}"""

    # ------------------------------------------------------------------
    # Relevance scoring
    # ------------------------------------------------------------------

    def _evaluate_metric_relevance(self, metrics: list, data_profile: Dict) -> list:
        profile_str  = json.dumps(data_profile, indent=2)
        metrics_str  = json.dumps(metrics[:10], indent=2)

        prompt = f"""You are a data quality expert. Score these proposed metrics for relevance to this dataset profile.

Dataset Profile:
{profile_str}

Proposed Metrics:
{metrics_str}

For EACH metric provide:
- "metric_name": the metric name
- "relevance_score": 0.0 to 1.0
- "reasoning": why this score

Return ONLY valid JSON array:
[
  {{"metric_name": "...", "relevance_score": 0.85, "reasoning": "..."}}
]"""

        raw = self.call_llm(prompt, stream=False)
        try:
            stripped = raw.strip() if raw else ""
            scores = self.parse_json(stripped if stripped.startswith("[") else f"[{stripped}]")
            if isinstance(scores, list) and all(isinstance(s, dict) for s in scores):
                return scores
            if isinstance(scores, list) and len(scores) == 1 and isinstance(scores[0], list):
                return scores[0]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # Deep evidence search for top metrics
    # ------------------------------------------------------------------

    def _search_for_refinement(self, high_value_metrics: list, domain: str) -> Dict[str, Any]:
        print(f"\n[RESEARCHER] Phase 3: Deep-diving into top metrics...")
        evidence = {}

        for metric in high_value_metrics[:8]:
            metric_name = metric.get("metric_name", "")
            print(f"[RESEARCHER]   -> Researching '{metric_name}'...")

            query = f"{domain} {metric_name} evaluation methodology best practices"

            web_result,   _ = self._safe_run(self.search_tool, query,                          f"Web Search: {metric_name}") if self.search_tool else ("", "")
            wiki_result,  _ = self._safe_run(self.wikipedia,   f"{domain} {metric_name}",      f"Wikipedia: {metric_name}") if self.wikipedia   else ("", "")
            arxiv_result, _ = self._safe_run(self.arxiv,       f"{metric_name} evaluation {domain}", f"Arxiv: {metric_name}") if self.arxiv  else ("", "")

            evidence[metric_name] = {
                "web":   web_result[:800]   if web_result   else "",
                "wiki":  wiki_result[:800]  if wiki_result  else "",
                "arxiv": arxiv_result[:800] if arxiv_result else "",
            }

        return evidence

    # ------------------------------------------------------------------
    # Coverage assessment  — are both math AND semantic types present?
    # ------------------------------------------------------------------

    def _assess_metric_coverage(self, metrics: list) -> Dict[str, Any]:
        """
        Checks if metrics cover both math/statistical AND semantic/text types.
        Returns assessment dict with needs_retry flag and missing_types list.
        """
        # Privacy/governance metrics are out of scope — exclude them from coverage counting
        excluded_type_tags = {"privacy", "security", "governance", "compliance", "pii"}
        excluded_keywords  = {"privacy", "pii", "gdpr", "hipaa", "sensitive", "personal data",
                               "anonymi", "redact", "compliance", "security", "vulnerab"}

        math_type_tags  = {"distribution", "statistical", "utility", "domain_specific", "label_noise"}
        sem_type_tags   = {"semantic_consistency", "text_quality"}
        math_keywords   = {"entropy", "divergence", "distribution", "statistical", "numeric", "correlation",
                           "variance", "skew", "kurtosis", "ks_test", "chi", "wasserstein", "jensen", "kl",
                           "imbalance", "frequency", "null", "duplicate", "outlier"}
        sem_keywords    = {"semantic", "coherence", "consistency", "embedding", "similarity", "readability",
                           "text quality", "nlp", "sentiment", "fluency", "tfidf", "cosine", "language"}

        math_metrics = []
        sem_metrics  = []

        for m in metrics:
            mtype = m.get("metric_type", "").lower()
            blob  = (m.get("metric_name", "") + " " + m.get("description", "") + " " + mtype).lower()

            # Skip out-of-scope privacy/governance metrics entirely
            if mtype in excluded_type_tags or any(kw in blob for kw in excluded_keywords):
                continue

            is_sem  = mtype in sem_type_tags  or any(kw in blob for kw in sem_keywords)
            is_math = mtype in math_type_tags or any(kw in blob for kw in math_keywords)

            if is_sem:
                sem_metrics.append(m)
            elif is_math:
                math_metrics.append(m)
            else:
                math_metrics.append(m)  # default to math bucket

        missing_types = []
        reasons       = []

        if len(math_metrics) < 2:
            missing_types.append("mathematical_statistical")
            reasons.append(f"Only {len(math_metrics)} math/statistical metrics (need >= 2)")
        if len(sem_metrics) < 2:
            missing_types.append("semantic_textual")
            reasons.append(f"Only {len(sem_metrics)} semantic/text metrics (need >= 2)")
        if len(metrics) < 6:
            reasons.append(f"Total metric count too low: {len(metrics)} (need >= 6)")

        needs_retry = bool(missing_types) or len(metrics) < 6

        print(f"[RESEARCHER] Coverage: {len(math_metrics)} math + {len(sem_metrics)} semantic = {len(metrics)} total")
        if needs_retry:
            for r in reasons:
                print(f"[RESEARCHER]   GAP: {r}")
        else:
            print(f"[RESEARCHER]   Coverage OK — both types present")

        return {
            "needs_retry":    needs_retry,
            "missing_types":  missing_types,
            "reasons":        reasons,
            "math_count":     len(math_metrics),
            "semantic_count": len(sem_metrics),
            "total_count":    len(metrics),
        }

    # ------------------------------------------------------------------
    # Targeted retry search — fires fresh queries for missing types
    # ------------------------------------------------------------------

    def _targeted_retry_search(
        self, domain: str, missing_types: List[str], dataset_type: str,
        column_summary: str, iteration: int
    ) -> str:
        print(f"\n[RESEARCHER] Iteration {iteration}: Targeted search for missing types: {missing_types}")
        contexts = []

        if "mathematical_statistical" in missing_types:
            queries = [
                f"{domain} KL divergence Wasserstein Jensen-Shannon statistical distance evaluation {dataset_type}",
                f"chi-square test entropy distribution evaluation {domain} data quality metrics",
            ]
            tools  = [self.search_tool, self.arxiv]
            labels = [f"Web [Math iter={iteration}]", f"Arxiv [Math iter={iteration}]"]
            for q, tool, label in zip(queries, tools, labels):
                result, _ = self._safe_run(tool, q, label) if tool else ("", "")
                if result and not result.startswith("["):
                    contexts.append(f"## Math Metrics (iter {iteration})\n{result}")

        if "semantic_textual" in missing_types:
            queries = [
                f"{domain} text semantic coherence consistency NLP evaluation metrics quality",
                f"semantic similarity sentence embeddings text quality {domain} dataset",
                f"semantic consistency evaluation NLP deep learning dataset benchmark",
            ]
            tools  = [self.search_tool, self.wikipedia, self.arxiv]
            labels = [
                f"Web [Sem iter={iteration}]",
                f"Wiki [Sem iter={iteration}]",
                f"Arxiv [Sem iter={iteration}]",
            ]
            for q, tool, label in zip(queries, tools, labels):
                result, _ = self._safe_run(tool, q, label) if tool else ("", "")
                if result and not result.startswith("["):
                    contexts.append(f"## Semantic Metrics (iter {iteration})\n{result}")

        return "\n\n".join(contexts)

    # ------------------------------------------------------------------
    # Generate additional metrics for missing types
    # ------------------------------------------------------------------

    def _generate_additional_metrics(
        self, domain: str, dataset_type: str, missing_types: List[str],
        new_context: str, existing_metrics: list, iteration: int
    ) -> list:
        existing_names = {m.get("metric_name", "").lower() for m in existing_metrics}

        type_desc = " AND ".join(
            "mathematical/statistical (KL-divergence, entropy, chi-square, Wasserstein, correlation tests)"
            if t == "mathematical_statistical"
            else "semantic/text-quality (semantic coherence, TF-IDF cosine similarity, text readability, embedding-based)"
            for t in missing_types
        )

        prompt = f"""You are a data quality research specialist filling coverage gaps.

Dataset: {domain} ({dataset_type})
MISSING TYPES: {type_desc}

SCOPE RESTRICTION: This is a DATA QUALITY pipeline. Do NOT generate metrics for:
privacy, PII detection, security, compliance, GDPR, sensitive data, anonymisation.
Those belong in a separate governance pipeline.

New research evidence:
{new_context[:2500]}

Already proposed metrics (DO NOT repeat these):
{json.dumps(sorted(existing_names), indent=2)}

Generate 4-6 NEW metrics covering ONLY the missing types: {type_desc}

Rules:
- Mathematical metrics: include scipy/numpy formulas in execution_hint
- Semantic metrics: include TF-IDF cosine or sentence-transformers approach in execution_hint
- Each metric must be clearly different from the existing ones above

Return ONLY valid JSON array (no markdown):
[
  {{
    "metric_name": "...",
    "metric_type": "distribution|semantic_consistency|text_quality|statistical|...",
    "description": "...",
    "reasoning": "Why this fills the data quality gap",
    "source_influence": "...",
    "execution_hint": "<concrete Python steps>",
    "relevance_score": 0.80
  }}
]"""

        raw = self.call_llm(prompt, stream=True)
        try:
            stripped = raw.strip() if raw else ""
            result = self.parse_json(stripped if stripped.startswith("[") else f"[{stripped}]")
            if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
                result = result[0]
            if isinstance(result, list) and all(isinstance(m, dict) for m in result):
                new_metrics = [m for m in result if m.get("metric_name", "").lower() not in existing_names]
                print(f"[RESEARCHER]   Generated {len(new_metrics)} new metrics for missing types")
                return new_metrics
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        thinker_out = state.get("thinker_output", {})
        iteration   = state.get("researcher_iteration", 0)

        if not thinker_out:
            return {
                "researcher_output":   {},
                "researcher_retry":    False,
                "researcher_iteration": iteration,
                "errors": ["Researcher: thinker_output missing."],
            }

        raw_meta = state.get("raw_metadata", {})
        column_summary = "\n".join(
            f"  * {c.get('name')} ({c.get('inferred_dtype')}, "
            f"{c.get('null_pct', 0):.1f}% null, {c.get('unique_count', '?')} unique)"
            for c in raw_meta.get("columns", [])
        )

        domain       = thinker_out.get("domain", {}).get("name", "general")
        dataset_type = thinker_out.get("dataset_type", "tabular")

        print("\n" + "=" * 70)
        if iteration == 0:
            print("[RESEARCHER] AGENTIC RESEARCH PHASE — Metric Discovery & Refinement")
        else:
            print(f"[RESEARCHER] RETRY ITERATION {iteration} — Targeted Gap-Filling")
        print("=" * 70)
        print(f"[RESEARCHER] Domain: '{domain}' | Type: '{dataset_type}'")

        # ── RETRY MODE ─────────────────────────────────────────────────
        if iteration > 0:
            existing_output  = state.get("researcher_output", {})
            existing_metrics = existing_output.get("final_metrics",
                               existing_output.get("proposed_metrics", []))

            print(f"[RESEARCHER] Existing metrics: {len(existing_metrics)}")
            coverage      = self._assess_metric_coverage(existing_metrics)
            missing_types = coverage["missing_types"]

            new_context = self._targeted_retry_search(
                domain, missing_types, dataset_type, column_summary, iteration
            )

            additional = []
            if new_context:
                additional = self._generate_additional_metrics(
                    domain, dataset_type, missing_types, new_context, existing_metrics, iteration
                )

            all_metrics = existing_metrics + additional
            print(f"\n[RESEARCHER] Consolidating {len(all_metrics)} metrics (existing + {len(additional)} new)...")

            top_for_consolidation = sorted(
                all_metrics, key=lambda m: m.get("relevance_score", 0.5), reverse=True
            )[:14]

            retry_prompt = f"""You are the Research Agent consolidating metrics for a DATA QUALITY pipeline.

Dataset: {domain} ({dataset_type})

SCOPE: Data quality only. Remove any privacy, PII, security, compliance, or governance metrics.

REQUIREMENT: Ensure final set contains BOTH:
- At least 2 mathematical/statistical metrics
- At least 2 semantic/text-quality metrics

All available metrics (existing + newly discovered):
{json.dumps(top_for_consolidation, indent=2)}

New research context:
{new_context[:1500]}

Output a FINAL set of 8-12 metrics with balanced coverage.

Return ONLY valid JSON (no markdown):
{{
  "final_metrics": [
    {{
      "metric_name": "...",
      "metric_type": "...",
      "description": "...",
      "reasoning": "...",
      "source_influence": "...",
      "execution_hint": "...",
      "relevance_score": 0.85,
      "supporting_evidence": "..."
    }}
  ],
  "research_summary": "...",
  "confidence_level": "<HIGH|MEDIUM|LOW>"
}}"""

            final_response = self.call_llm(retry_prompt, stream=True)
            result = self.parse_json(final_response)

            if result.get("_parse_error"):
                result = existing_output  # fall back to existing output

            result["research_context"]       = existing_output.get("research_context", {})
            result["deep_research_evidence"] = existing_output.get("deep_research_evidence", {})

            final_metrics = result.get("final_metrics", [])
            new_coverage  = self._assess_metric_coverage(final_metrics)
            should_retry  = new_coverage["needs_retry"] and (iteration < 2)

            n          = len(final_metrics)
            confidence = result.get("confidence_level", "MEDIUM")
            print(f"\n[RESEARCHER] RETRY {iteration} COMPLETE: {n} metrics, {confidence} confidence")
            if should_retry:
                print(f"[RESEARCHER]   Still missing types — will retry (iteration {iteration + 1})")

            return {
                "researcher_output":    result,
                "researcher_retry":     should_retry,
                "researcher_iteration": iteration + 1,
                "errors": [],
            }

        # ── INITIAL MODE: Full 4-phase research ────────────────────────
        print(f"\n[RESEARCHER] Phase 1: Initial research from external sources...")
        context_data = self._gather_context(domain, dataset_type, column_summary)

        print(f"\n[RESEARCHER] Phase 1b: Generating 15 initial metric proposals...")
        prompt = self.build_prompt(thinker_out, context_data["text"])
        response_text = self.call_llm(prompt, stream=True)
        initial_metrics = self.parse_json(response_text)

        if initial_metrics.get("_parse_error"):
            return {
                "researcher_output":    initial_metrics,
                "researcher_retry":     False,
                "researcher_iteration": 1,
                "errors": ["Researcher: JSON parse failed in Phase 1."],
            }

        all_metrics = initial_metrics.get("proposed_metrics", [])
        print(f"\n[RESEARCHER] Generated {len(all_metrics)} initial metrics")

        # ── Phase 2: Relevance scoring ─────────────────────────────────
        print(f"\n[RESEARCHER] Phase 2: Evaluating metric relevance...")
        relevance_scores = self._evaluate_metric_relevance(all_metrics, thinker_out)

        for metric in all_metrics:
            metric_name = metric.get("metric_name", "")
            for score_entry in relevance_scores:
                if score_entry.get("metric_name") == metric_name:
                    metric["relevance_score"]    = score_entry.get("relevance_score", 0.5)
                    metric["relevance_reasoning"] = score_entry.get("reasoning", "")
                    break

        ranked_metrics = sorted(all_metrics, key=lambda m: m.get("relevance_score", 0), reverse=True)
        print(f"[RESEARCHER] Ranked metrics by relevance:")
        for i, m in enumerate(ranked_metrics[:8], 1):
            score = m.get("relevance_score", 0)
            print(f"[RESEARCHER]   {i}. {m.get('metric_name')} (relevance: {score:.2f})")

        # ── Phase 3: Deep evidence for top metrics ─────────────────────
        print(f"\n[RESEARCHER] Phase 3: Deep research on top {min(8, len(ranked_metrics))} metrics...")
        top_metrics  = ranked_metrics[:8]
        deep_evidence = self._search_for_refinement(top_metrics, domain)
        print(f"[RESEARCHER] Gathered detailed evidence for top metrics")

        # ── Phase 4: Consolidation ─────────────────────────────────────
        print(f"\n[RESEARCHER] Phase 4: Consolidating evidence and finalizing metrics...")

        final_prompt = f"""You are the Research Agent finalizing metric selection for a DATA QUALITY pipeline.

Dataset Domain: {domain}
Dataset Type: {dataset_type}

SCOPE — DATA QUALITY ONLY. Exclude any metric related to:
privacy, PII, security, compliance, GDPR, sensitive data, anonymisation.
Those are handled by a separate Data Governance pipeline.

REQUIRED COVERAGE:
- At least 3 mathematical/statistical metrics (entropy, KL-divergence, distribution tests, etc.)
- At least 3 semantic/text-quality metrics (semantic coherence, embedding similarity, readability, etc.)

Top-ranked metrics with evidence:
{json.dumps(top_metrics, indent=2)}

Deep Research Evidence (by metric):
{json.dumps(deep_evidence, indent=2)}

Output a FINAL structured set of 10-12 MOST RELIABLE data quality metrics with balanced coverage.

Return ONLY valid JSON (no markdown):
{{
  "final_metrics": [
    {{
      "metric_name": "...",
      "metric_type": "distribution|label_noise|semantic_consistency|text_quality|domain_specific|utility|statistical|other",
      "description": "...",
      "reasoning": "...",
      "source_influence": "<cite specific evidence from web/wiki/arxiv>",
      "execution_hint": "...",
      "relevance_score": 0.85,
      "supporting_evidence": "..."
    }}
  ],
  "research_summary": "<comprehensive summary citing compelling findings>",
  "confidence_level": "<HIGH|MEDIUM|LOW>"
}}"""

        final_response = self.call_llm(final_prompt, stream=True)
        result = self.parse_json(final_response)

        if result.get("_parse_error"):
            return {
                "researcher_output":    result,
                "researcher_retry":     False,
                "researcher_iteration": 1,
                "errors": ["Researcher: JSON parse failed in Phase 4."],
            }

        result["research_context"]       = context_data["sources"]
        result["deep_research_evidence"] = deep_evidence

        final_metrics = result.get("final_metrics", [])
        n          = len(final_metrics)
        confidence = result.get("confidence_level", "UNKNOWN")
        print(f"\n[RESEARCHER] FINALIZED: {n} metrics with {confidence} confidence")

        # ── Coverage check — signal graph to retry if gaps remain ──────
        coverage     = self._assess_metric_coverage(final_metrics)
        should_retry = coverage["needs_retry"]

        if should_retry:
            print(f"[RESEARCHER]   Coverage gaps detected — graph will route back for retry")
        else:
            print(f"[RESEARCHER]   Coverage complete — both math and semantic types present")

        return {
            "researcher_output":    result,
            "researcher_retry":     should_retry,
            "researcher_iteration": 1,
            "errors": [],
        }
