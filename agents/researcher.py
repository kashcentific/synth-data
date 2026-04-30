import json
import sys
import os
import re
import requests
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, List, Tuple

from base import BaseAgent
from duckduckgo_search import DDGS                                   # direct — avoids backend rotation errors
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
import arxiv as arxiv_lib

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


class ResearchAgent(BaseAgent):
    """
    Agentic researcher that iterates until it produces a well-balanced set of
    metrics covering BOTH mathematical/statistical AND semantic/text-quality types.

    Key improvements over v1:
    - DuckDuckGo: multi-angle LLM-generated queries + structured results + real URL scraping
    - ArXiv: full PDF download via PyPDFLoader, top_k=5 papers, chunked + cited excerpts
    - Each metric is attributed to a specific paper with an exact supporting quote

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

        print("[RESEARCHER]   * DuckDuckGo (DDGS direct, multi-angle): OK")

        try:
            print("[RESEARCHER]   * Wikipedia: ", end="", flush=True)
            self.wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            self.wikipedia = None

        print("[RESEARCHER]   * ArXiv (deep PDF loader): OK")

    # ------------------------------------------------------------------
    # Wikipedia safe runner (kept for wiki only)
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
    # DuckDuckGo: LLM-generated multi-angle queries
    # ------------------------------------------------------------------

    def _generate_search_queries(self, domain: str, dataset_type: str, topic: str) -> List[str]:
        """
        Ask the LLM to produce 4 queries from distinct research angles so that
        each DDG run hits a different part of the web (practitioner blogs,
        academic papers, benchmark surveys, formula/algorithm explanations).
        Wrapping key terms in quotes forces exact-match results.
        """
        prompt = f"""Generate exactly 4 diverse DuckDuckGo search queries to find information about:
"{topic}"
Domain: {domain} | Dataset type: {dataset_type}

Each query must approach the topic from a DIFFERENT angle:
1. Implementation/practitioner: "how to compute" or "Python implementation" angle
2. Academic citation: formulated as a paper title fragment or key author keyword
3. Benchmark/survey: focus on comparative evaluation or survey papers
4. Formula/algorithm: mathematical definition or specific statistical test name

Rules:
- Wrap the most specific technical terms in double-quotes for exact matching
- Keep each query under 12 words
- Add "2022 OR 2023 OR 2024" to at least one query to bias toward recent work

Return ONLY a JSON array of exactly 4 query strings, no commentary:
["query1", "query2", "query3", "query4"]"""

        raw = self.call_llm(prompt, stream=False)
        try:
            parsed = self.parse_json(raw)
            if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                return parsed[:4]
            # parse_json returns dict when JSON is an object — check for embedded list
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list) and all(isinstance(q, str) for q in v):
                        return v[:4]
        except Exception:
            pass
        # Fallback: three hardcoded angles + original topic
        return [
            f'"{domain}" "{dataset_type}" data quality evaluation metrics',
            f'{domain} synthetic data quality benchmark survey 2023 OR 2024',
            topic,
        ]

    # ------------------------------------------------------------------
    # DuckDuckGo: multi-query runner with URL content fetching
    # ------------------------------------------------------------------

    def _multi_angle_duckduckgo(self, queries: List[str]) -> Tuple[str, List[Dict]]:
        """
        Run all angle-queries via DDGS (direct, no LangChain wrapper), deduplicate
        by URL, then scrape the top 3 unique URLs for full page text.
        Returns (formatted_text_for_prompt, structured_result_list).
        """
        seen_urls: set = set()
        structured: List[Dict] = []

        for query in queries:
            try:
                with DDGS() as ddgs:
                    raw = ddgs.text(query, max_results=8) or []
                for r in raw:
                    # DDGS returns: {"title", "href", "body"}
                    url = r.get("href", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        structured.append({
                            "title":        r.get("title", ""),
                            "snippet":      r.get("body", ""),
                            "url":          url,
                            "angle_query":  query,
                            "full_content": "",
                        })
            except Exception as exc:
                print(f"[RESEARCHER]      DDG error ({query[:50]}): {type(exc).__name__}")

        print(f"[RESEARCHER]      DDG: {len(structured)} unique URLs across {len(queries)} queries")

        # Scrape actual content from top 3 URLs
        scraped = 0
        for r in structured:
            if scraped >= 3:
                break
            content = self._fetch_url_content(r["url"])
            if content:
                r["full_content"] = content
                scraped += 1
                print(f"[RESEARCHER]      Scraped: {r['url'][:60]} ({len(content)} chars)")

        # Build text block for the LLM prompt
        parts = []
        for r in structured[:12]:
            block = (
                f"[{r['title']}]\n"
                f"URL: {r['url']}\n"
                f"Query angle: {r['angle_query']}\n"
                f"Snippet: {r['snippet']}\n"
            )
            if r["full_content"]:
                block += f"Full page excerpt:\n{r['full_content'][:1000]}\n"
            parts.append(block)

        return "\n---\n".join(parts), structured

    def _fetch_url_content(self, url: str, timeout: int = 10) -> str:
        """
        Fetch a page and strip HTML to plain text.
        Returns empty string on any error so callers can gracefully skip.
        """
        try:
            resp = requests.get(
                url, timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"},
            )
            resp.raise_for_status()
            text = resp.text
            # Remove script/style blocks first, then all remaining tags
            text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>",  " ", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>",                  " ", text)
            text = re.sub(r"&[a-zA-Z#0-9]+;",          " ", text)
            text = re.sub(r"\s+",                       " ", text).strip()
            return text[:3000]
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # ArXiv: full PDF loading with PyPDFLoader + relevant chunk extraction
    # ------------------------------------------------------------------

    def _load_arxiv_papers_deep(
        self, query: str, domain: str, top_k: int = 5
    ) -> List[Dict]:
        """
        Search ArXiv, download top_k PDFs, chunk each paper with
        RecursiveCharacterTextSplitter, and return only the chunks that
        contain 2+ domain-relevant keywords — with page-level citations.
        """
        papers: List[Dict] = []
        domain_keywords = {
            domain.lower(), "metric", "evaluation", "quality", "measure",
            "score", "benchmark", "dataset", "statistical", "semantic",
        }

        print(f"\n[RESEARCHER] ── ArXiv ────────────────────────────────────────")
        print(f"[RESEARCHER]   Query  : \"{query}\"")
        print(f"[RESEARCHER]   top_k  : {top_k}  (full PDF download + chunking)")

        try:
            client = arxiv_lib.Client()
            search = arxiv_lib.Search(
                query=query,
                max_results=top_k,
                sort_by=arxiv_lib.SortCriterion.Relevance,
            )

            results_iter = list(client.results(search))
            if not results_iter:
                print("[RESEARCHER]   ArXiv returned 0 results for this query.")
                return papers

            print(f"[RESEARCHER]   Found {len(results_iter)} papers:")
            for i, paper in enumerate(results_iter, 1):
                arxiv_id = paper.get_short_id() if hasattr(paper, "get_short_id") else paper.entry_id.split("/")[-1]
                print(f"[RESEARCHER]   [{i}] {paper.title[:70]}")
                print(f"[RESEARCHER]       ID: {arxiv_id}  | Year: {paper.published.year if paper.published else '?'}")
                print(f"[RESEARCHER]       Abstract: {paper.summary[:200].replace(chr(10),' ')}...")

                paper_data: Dict = {
                    "title":             paper.title,
                    "authors":           [a.name for a in paper.authors[:4]],
                    "year":              paper.published.year if paper.published else "?",
                    "arxiv_id":          arxiv_id,
                    "abstract":          paper.summary[:600],
                    "pdf_url":           paper.pdf_url,
                    "relevant_excerpts": [],
                    "pdf_loaded":        False,
                }

                print(f"[RESEARCHER]       Downloading PDF...", end="", flush=True)
                try:
                    resp = requests.get(
                        paper.pdf_url, timeout=40,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    resp.raise_for_status()

                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(resp.content)
                        tmp_path = tmp.name

                    loader = PyPDFLoader(tmp_path)
                    pages  = loader.load()[:20]

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800, chunk_overlap=100,
                        separators=["\n\n", "\n", ". ", " "],
                    )
                    chunks = splitter.split_documents(pages)

                    relevant_chunks = []
                    for chunk in chunks:
                        text_lower = chunk.page_content.lower()
                        hits = sum(1 for kw in domain_keywords if kw in text_lower)
                        if hits >= 2:
                            relevant_chunks.append({
                                "text": chunk.page_content.strip()[:600],
                                "page": chunk.metadata.get("page", "?"),
                            })

                    paper_data["relevant_excerpts"] = relevant_chunks[:5]
                    paper_data["pdf_loaded"]        = True
                    os.unlink(tmp_path)

                    print(f" {len(pages)}pp, {len(relevant_chunks)} relevant chunks extracted")

                except Exception as exc:
                    print(f" FAILED ({type(exc).__name__}) — abstract only")

                papers.append(paper_data)

            pdf_ok = sum(1 for p in papers if p["pdf_loaded"])
            print(f"[RESEARCHER]   ArXiv done: {pdf_ok}/{len(papers)} PDFs loaded, "
                  f"{sum(len(p['relevant_excerpts']) for p in papers)} total relevant chunks")

        except Exception as exc:
            print(f"[RESEARCHER]   ArXiv search failed: {exc}")

        return papers

    def _format_arxiv_papers(self, papers: List[Dict]) -> str:
        """Format loaded papers into a citation-rich block for the LLM prompt."""
        if not papers:
            return "No ArXiv papers loaded."

        parts = []
        for i, p in enumerate(papers, 1):
            header = (
                f"Paper {i}: \"{p['title']}\"\n"
                f"  Authors : {', '.join(p['authors'])}\n"
                f"  Year    : {p['year']}   ArXiv ID: {p['arxiv_id']}\n"
                f"  Abstract: {p['abstract'][:400]}\n"
            )
            if p["relevant_excerpts"]:
                header += "  Relevant excerpts from full PDF:\n"
                for ex in p["relevant_excerpts"]:
                    header += f"    [page {ex['page']}] \"{ex['text']}\"\n"
            elif not p["pdf_loaded"]:
                header += "  (PDF unavailable — abstract only)\n"
            else:
                header += "  (No high-relevance chunks found in PDF)\n"
            parts.append(header)

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Initial context gathering — now uses multi-angle DDG + deep ArXiv
    # ------------------------------------------------------------------

    def _gather_context(self, domain: str, dataset_type: str, column_summary: str) -> Dict[str, Any]:
        topic        = f"{domain} {dataset_type} data quality evaluation metrics"
        arxiv_query  = f"synthetic data evaluation metrics {domain} machine learning fidelity quality"
        wiki_query   = f"{domain} machine learning dataset evaluation benchmark"

        # ── 1. Multi-angle DuckDuckGo ──────────────────────────────────
        print("\n[RESEARCHER]   Generating diverse DuckDuckGo search angles...")
        queries = self._generate_search_queries(domain, dataset_type, topic)
        for i, q in enumerate(queries, 1):
            print(f"[RESEARCHER]     Angle {i}: {q}")

        print("\n[RESEARCHER]   Running multi-angle DuckDuckGo search + URL scraping...")
        web_text, web_results = self._multi_angle_duckduckgo(queries)

        # ── 2. Wikipedia ───────────────────────────────────────────────
        wiki_full, wiki_excerpt = self._safe_run(
            self.wikipedia, wiki_query, "Wikipedia", char_limit=1200
        )

        # ── 3. Deep ArXiv — download PDFs, chunk, extract excerpts ─────
        arxiv_papers = self._load_arxiv_papers_deep(arxiv_query, domain, top_k=5)
        arxiv_text   = self._format_arxiv_papers(arxiv_papers)

        combined_text = (
            f"## Web Search Results (Multi-Angle DuckDuckGo + Page Scraping)\n{web_text}\n\n"
            f"## Wikipedia Summary\n{wiki_full}\n\n"
            f"## Deep ArXiv Paper Analysis (Full PDF, top-{len(arxiv_papers)} papers)\n{arxiv_text}\n\n"
            f"## Dataset Column Profile\n{column_summary}"
        )

        sources = {
            "duckduckgo": {
                "queries":       queries,
                "unique_urls":   len(web_results),
                "scraped_pages": sum(1 for r in web_results if r.get("full_content")),
            },
            "wikipedia":    {"query": wiki_query, "excerpt": wiki_excerpt},
            "arxiv_papers": [
                {
                    "title":          p["title"],
                    "arxiv_id":       p["arxiv_id"],
                    "authors":        p["authors"],
                    "year":           p["year"],
                    "pdf_loaded":     p["pdf_loaded"],
                    "excerpts_count": len(p["relevant_excerpts"]),
                }
                for p in arxiv_papers
            ],
        }

        return {
            "text":         combined_text,
            "sources":      sources,
            "arxiv_papers": arxiv_papers,
        }

    # ------------------------------------------------------------------
    # Prompt builder — includes paper citation instructions
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        thinker_output: Dict[str, Any],
        external_context: str,
        arxiv_papers: List[Dict] = None,
    ) -> str:
        domain       = thinker_output.get("domain", {}).get("name", "general")
        dataset_type = thinker_output.get("dataset_type", "tabular")

        paper_names = ""
        if arxiv_papers:
            paper_names = "\n".join(
                f'  - Paper {i}: "{p["title"]}" [{p["arxiv_id"]}]'
                for i, p in enumerate(arxiv_papers, 1)
            )

        return f"""You are the Research Agent in a Data Quality Audit System.

You have gathered real research context from multi-angle DuckDuckGo searches (with full page content),
Wikipedia, and {len(arxiv_papers) if arxiv_papers else 0} fully-loaded ArXiv PDFs with extracted excerpts.
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

External Research Context (web search + Wikipedia + full ArXiv PDFs):
{external_context}

{'Available ArXiv papers (cite these by title + ID):' + chr(10) + paper_names if paper_names else ''}

HARD CITATION REQUIREMENT — READ CAREFULLY:
Every metric MUST have at least 2 entries in "paper_citations".
Each citation must reference a REAL paper from the ArXiv list above (use its exact title and arxiv_id).
Include the EXACT sentence or phrase from the paper excerpt that justifies this metric.
If you cannot find 2 supporting papers for a metric from the evidence above, DO NOT include that metric.
A metric with 0 or 1 citations will be automatically discarded by the pipeline.

Return ONLY valid JSON — no markdown:

{{
  "proposed_metrics": [
    {{
      "metric_name": "<short name>",
      "metric_type": "<distribution|label_noise|semantic_consistency|text_quality|domain_specific|utility|statistical|other>",
      "description": "<what this metric measures>",
      "reasoning": "<why this metric matters for data quality of this specific dataset>",
      "source_influence": "<which tool/URL/paper influenced this metric>",
      "execution_hint": "<concrete Python/pandas/scipy steps with actual column names>",
      "paper_citations": [
        {{
          "title": "<exact paper title from the list above>",
          "arxiv_id": "<arxiv id, e.g. 2301.12345>",
          "authors": "<First Author et al.>",
          "year": "<year>",
          "supporting_text": "<exact sentence from the paper excerpt supporting this metric>"
        }},
        {{
          "title": "<second paper title>",
          "arxiv_id": "<second arxiv id>",
          "authors": "<authors>",
          "year": "<year>",
          "supporting_text": "<exact sentence from this paper>"
        }}
      ]
    }}
  ],
  "research_summary": "<2-3 sentence summary citing the most relevant findings and papers>"
}}"""

    # ------------------------------------------------------------------
    # Citation gate — enforce minimum 2 paper citations per metric
    # ------------------------------------------------------------------

    _CITATION_SCHEMA = {
        "title": "", "arxiv_id": "", "authors": "", "year": "", "supporting_text": ""
    }

    def _filter_by_citations(self, metrics: list, min_citations: int = 2) -> list:
        """
        Drop any metric that cannot show at least `min_citations` distinct paper
        citations.  Also normalises the field name: merges a legacy single
        paper_citation into paper_citations if needed.
        """
        kept, dropped = [], []
        for m in metrics:
            cites = m.get("paper_citations", [])

            # Back-compat: single paper_citation field
            if not cites:
                single = m.get("paper_citation", {})
                if single.get("arxiv_id") or single.get("title"):
                    cites = [single]

            # Keep only citations that have at least an ID or a title
            valid = [c for c in cites
                     if isinstance(c, dict) and (c.get("arxiv_id") or c.get("title"))]

            if len(valid) >= min_citations:
                m["paper_citations"] = valid
                m.pop("paper_citation", None)   # remove legacy field
                kept.append(m)
            else:
                dropped.append(m.get("metric_name", "?"))

        if dropped:
            print(f"[RESEARCHER]   ✗ Dropped {len(dropped)} under-cited metrics "
                  f"(need ≥{min_citations} papers): {dropped}")
        print(f"[RESEARCHER]   ✓ {len(kept)} metrics passed the 2-paper citation gate")
        return kept

    # ------------------------------------------------------------------
    # Relevance scoring
    # ------------------------------------------------------------------

    def _evaluate_metric_relevance(self, metrics: list, data_profile: Dict) -> list:
        profile_str = json.dumps(data_profile, indent=2)
        metrics_str = json.dumps(metrics[:10], indent=2)

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
    # Phase 3: deep evidence per top metric — multi-angle DDG + quick ArXiv
    # ------------------------------------------------------------------

    def _search_for_refinement(self, high_value_metrics: list, domain: str) -> Dict[str, Any]:
        print(f"\n[RESEARCHER] Phase 3: Deep-diving into top metrics (multi-angle DDG)...")
        evidence = {}

        for metric in high_value_metrics[:6]:
            metric_name = metric.get("metric_name", "")
            print(f"[RESEARCHER]   -> Researching '{metric_name}'...")

            topic = f"{domain} {metric_name} data quality evaluation"

            # Multi-angle DDG for this specific metric
            queries = self._generate_search_queries(domain, "tabular", topic)
            web_text, _ = self._multi_angle_duckduckgo(queries)

            # Quick ArXiv (top 2 papers, abstract only — no PDF download for speed)
            # Returns both a text blob AND structured paper records so the
            # final consolidation prompt can offer them as citable sources.
            arxiv_text   = ""
            quick_papers: List[Dict] = []
            try:
                client = arxiv_lib.Client()
                search = arxiv_lib.Search(
                    query=f"{metric_name} data quality evaluation {domain} NLP machine learning",
                    max_results=2,
                    sort_by=arxiv_lib.SortCriterion.Relevance,
                )
                found_text = []
                for p in client.results(search):
                    pid = p.get_short_id() if hasattr(p, "get_short_id") else p.entry_id.split("/")[-1]
                    print(f"[RESEARCHER]      ArXiv: \"{p.title[:60]}\" [{pid}]")
                    found_text.append(
                        f'"{p.title}" [{pid}] ({p.published.year if p.published else "?"})\n'
                        f'Abstract: {p.summary[:400]}'
                    )
                    quick_papers.append({
                        "title":    p.title,
                        "arxiv_id": pid,
                        "authors":  [a.name for a in p.authors[:3]],
                        "year":     p.published.year if p.published else "?",
                        "abstract": p.summary[:500],
                        "source":   "phase3_quick",
                    })
                arxiv_text = "\n\n".join(found_text)
                if not found_text:
                    print(f"[RESEARCHER]      ArXiv: no papers found")
            except Exception as exc:
                print(f"[RESEARCHER]      ArXiv quick search failed: {type(exc).__name__}")

            # Wikipedia — always anchor to "data quality NLP" to avoid off-topic pages
            wiki_result = ""
            if self.wikipedia:
                wiki_query = f"{metric_name} data quality NLP machine learning evaluation"
                wiki_result, _ = self._safe_run(
                    self.wikipedia,
                    wiki_query,
                    f"Wikipedia: {metric_name}",
                )

            evidence[metric_name] = {
                "web_multi_angle":  web_text[:1200],
                "arxiv_quick":      arxiv_text[:800],
                "arxiv_quick_papers": quick_papers,   # structured, citable
                "wiki":             wiki_result[:600],
            }

        return evidence

    # ------------------------------------------------------------------
    # Coverage assessment — are both math AND semantic types present?
    # ------------------------------------------------------------------

    def _assess_metric_coverage(self, metrics: list) -> Dict[str, Any]:
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

            if mtype in excluded_type_tags or any(kw in blob for kw in excluded_keywords):
                continue

            is_sem  = mtype in sem_type_tags  or any(kw in blob for kw in sem_keywords)
            is_math = mtype in math_type_tags or any(kw in blob for kw in math_keywords)

            if is_sem:
                sem_metrics.append(m)
            elif is_math:
                math_metrics.append(m)
            else:
                math_metrics.append(m)

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
    # Targeted retry search — multi-angle DDG for missing types
    # ------------------------------------------------------------------

    def _targeted_retry_search(
        self, domain: str, missing_types: List[str], dataset_type: str,
        column_summary: str, iteration: int
    ) -> str:
        print(f"\n[RESEARCHER] Iteration {iteration}: Targeted search for missing types: {missing_types}")
        contexts = []

        if "mathematical_statistical" in missing_types:
            topics = [
                f"{domain} KL divergence Wasserstein Jensen-Shannon statistical distance {dataset_type}",
                f"chi-square entropy distribution evaluation {domain} data quality",
            ]
            for topic in topics:
                queries  = self._generate_search_queries(domain, dataset_type, topic)
                web_text, _ = self._multi_angle_duckduckgo(queries)
                if web_text:
                    contexts.append(f"## Math Metrics Research (iter {iteration})\n{web_text[:1500]}")

        if "semantic_textual" in missing_types:
            topics = [
                f"{domain} text semantic coherence consistency NLP evaluation metrics quality",
                f"semantic similarity sentence embeddings text quality {domain} dataset benchmark",
            ]
            for topic in topics:
                queries  = self._generate_search_queries(domain, dataset_type, topic)
                web_text, _ = self._multi_angle_duckduckgo(queries)
                if web_text:
                    contexts.append(f"## Semantic Metrics Research (iter {iteration})\n{web_text[:1500]}")

            wiki_result = ""
            if self.wikipedia:
                wiki_result, _ = self._safe_run(
                    self.wikipedia,
                    f"semantic similarity NLP text quality metrics",
                    f"Wiki [Sem iter={iteration}]",
                )
            if wiki_result:
                contexts.append(f"## Wikipedia Semantic Context\n{wiki_result}")

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

New research evidence:
{new_context[:2500]}

Already proposed metrics (DO NOT repeat these):
{json.dumps(sorted(existing_names), indent=2)}

Generate 4-6 NEW metrics covering ONLY the missing types: {type_desc}

Rules:
- Mathematical metrics: include scipy/numpy formulas in execution_hint
- Semantic metrics: include TF-IDF cosine or sentence-transformers approach in execution_hint
- Each metric must be clearly different from the existing ones above
- Each metric MUST have at least 2 entries in paper_citations — cite specific papers from the research evidence
- If you cannot find 2 supporting papers for a metric, DO NOT include that metric

Return ONLY valid JSON array (no markdown):
[
  {{
    "metric_name": "...",
    "metric_type": "distribution|semantic_consistency|text_quality|statistical|...",
    "description": "...",
    "reasoning": "Why this fills the data quality gap",
    "source_influence": "...",
    "execution_hint": "<concrete Python steps>",
    "relevance_score": 0.80,
    "paper_citations": [
      {{"title": "...", "arxiv_id": "...", "authors": "...", "year": "...", "supporting_text": "..."}},
      {{"title": "...", "arxiv_id": "...", "authors": "...", "year": "...", "supporting_text": "..."}}
    ]
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
                "researcher_output":    {},
                "researcher_retry":     False,
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
            print("[RESEARCHER] AGENTIC RESEARCH PHASE — Multi-Angle DDG + Deep ArXiv PDF")
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
Carry forward any paper_citations lists from the input metrics.

HARD CITATION REQUIREMENT: every metric must have at least 2 entries in paper_citations.
Omit any metric you cannot back with 2 real papers from the evidence above.

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
      "supporting_evidence": "...",
      "paper_citations": [
        {{"title": "...", "arxiv_id": "...", "authors": "...", "year": "...", "supporting_text": "..."}},
        {{"title": "...", "arxiv_id": "...", "authors": "...", "year": "...", "supporting_text": "..."}}
      ]
    }}
  ],
  "research_summary": "...",
  "confidence_level": "<HIGH|MEDIUM|LOW>"
}}"""

            final_response = self.call_llm(retry_prompt, stream=True)
            result = self.parse_json(final_response)

            if result.get("_parse_error"):
                result = existing_output

            result["research_context"]       = existing_output.get("research_context", {})
            result["deep_research_evidence"] = existing_output.get("deep_research_evidence", {})

            # Citation gate: drop metrics with < 2 paper citations
            raw_retry = result.get("final_metrics", [])
            result["final_metrics"] = self._filter_by_citations(raw_retry, min_citations=2)

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
        print(f"\n[RESEARCHER] Phase 1: Multi-angle DuckDuckGo + Deep ArXiv PDF loading...")
        context_data = self._gather_context(domain, dataset_type, column_summary)

        print(f"\n[RESEARCHER] Phase 1b: Generating 15 initial metric proposals with paper citations...")
        prompt = self.build_prompt(
            thinker_out,
            context_data["text"],
            arxiv_papers=context_data.get("arxiv_papers", []),
        )
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
                    metric["relevance_score"]     = score_entry.get("relevance_score", 0.5)
                    metric["relevance_reasoning"] = score_entry.get("reasoning", "")
                    break

        ranked_metrics = sorted(all_metrics, key=lambda m: m.get("relevance_score", 0), reverse=True)
        print(f"[RESEARCHER] Ranked metrics by relevance:")
        for i, m in enumerate(ranked_metrics[:8], 1):
            score = m.get("relevance_score", 0)
            cites = m.get("paper_citations", [])
            if not cites and m.get("paper_citation", {}).get("arxiv_id"):
                cites = [m["paper_citation"]]
            cited = f"📄x{len(cites)}" if len(cites) >= 2 else ("📄x1" if cites else "   ")
            print(f"[RESEARCHER]   {cited} {i}. {m.get('metric_name')} (relevance: {score:.2f})")

        # ── Phase 3: Multi-angle deep evidence for top metrics ─────────
        print(f"\n[RESEARCHER] Phase 3: Multi-angle research on top {min(6, len(ranked_metrics))} metrics...")
        top_metrics   = ranked_metrics[:6]
        deep_evidence = self._search_for_refinement(top_metrics, domain)

        # ── Phase 4: Consolidation ─────────────────────────────────────
        print(f"\n[RESEARCHER] Phase 4: Consolidating evidence and finalizing metrics...")

        arxiv_papers  = context_data.get("arxiv_papers", [])
        paper_summary = self._format_arxiv_papers(arxiv_papers)

        # Build a unified pool of all citable papers (Phase 1 full-PDFs + Phase 3 quick)
        phase3_quick_papers: List[Dict] = []
        for ev in deep_evidence.values():
            phase3_quick_papers.extend(ev.get("arxiv_quick_papers", []))
        # Deduplicate by arxiv_id
        seen_ids: set = set()
        all_citable: List[Dict] = []
        for p in arxiv_papers + phase3_quick_papers:
            pid = p.get("arxiv_id", "")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_citable.append(p)

        all_paper_names = "\n".join(
            f'  Paper {i}: "{p["title"]}" [{p["arxiv_id"]}]  '
            f'({", ".join(p.get("authors", [])[:2])} {p.get("year", "")})'
            for i, p in enumerate(all_citable, 1)
        )

        final_prompt = f"""You are the Research Agent finalizing metric selection for a DATA QUALITY pipeline.

Dataset Domain: {domain}
Dataset Type: {dataset_type}

SCOPE — DATA QUALITY ONLY. Exclude any metric related to:
privacy, PII, security, compliance, GDPR, sensitive data, anonymisation.

REQUIRED COVERAGE:
- At least 3 mathematical/statistical metrics (entropy, KL-divergence, distribution tests, etc.)
- At least 3 semantic/text-quality metrics (semantic coherence, embedding similarity, readability, etc.)

HARD CITATION REQUIREMENT — TWO-PAPER MINIMUM:
Every metric in "final_metrics" MUST have at least 2 entries in "paper_citations".
Each citation must reference a REAL paper from the list below (use its exact title and arxiv_id).
Include the EXACT sentence or phrase from the paper excerpt that justifies the metric.
If you cannot find 2 supporting papers for a metric, OMIT that metric entirely.
Metrics with 0 or 1 citations are automatically discarded.

All citable papers ({len(all_citable)} total — Phase 1 full PDFs + Phase 3 quick search):
{all_paper_names}

Top-ranked metrics (carry their existing citations forward, add more if possible):
{json.dumps(top_metrics, indent=2)}

Multi-angle web research evidence (by metric):
{json.dumps(deep_evidence, indent=2)}

Full ArXiv paper excerpts (use exact quotes for supporting_text):
{paper_summary[:3000]}

Output a FINAL set of 10-12 data quality metrics. Only include metrics you can back with ≥2 papers.

Return ONLY valid JSON (no markdown):
{{
  "final_metrics": [
    {{
      "metric_name": "...",
      "metric_type": "distribution|label_noise|semantic_consistency|text_quality|domain_specific|utility|statistical|other",
      "description": "...",
      "reasoning": "...",
      "source_influence": "<cite specific paper title or URL>",
      "execution_hint": "...",
      "relevance_score": 0.85,
      "supporting_evidence": "...",
      "paper_citations": [
        {{
          "title": "<exact paper title>",
          "arxiv_id": "<arxiv id>",
          "authors": "<First Author et al.>",
          "year": "<year>",
          "supporting_text": "<exact sentence from the paper excerpt>"
        }},
        {{
          "title": "<second paper title>",
          "arxiv_id": "<second arxiv id>",
          "authors": "<authors>",
          "year": "<year>",
          "supporting_text": "<exact sentence>"
        }}
      ]
    }}
  ],
  "research_summary": "<comprehensive summary citing paper titles>",
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

        # ── Citation gate: drop anything with < 2 paper citations ──────
        raw_metrics   = result.get("final_metrics", [])
        final_metrics = self._filter_by_citations(raw_metrics, min_citations=2)
        result["final_metrics"] = final_metrics

        n          = len(final_metrics)
        confidence = result.get("confidence_level", "UNKNOWN")
        fully_cited = sum(1 for m in final_metrics if len(m.get("paper_citations", [])) >= 2)
        print(f"\n[RESEARCHER] FINALIZED: {n} metrics, {fully_cited} with ≥2 paper citations, {confidence} confidence")

        coverage     = self._assess_metric_coverage(final_metrics)
        should_retry = coverage["needs_retry"]

        if should_retry:
            print(f"[RESEARCHER]   Coverage gaps — graph will route back for retry")
        else:
            print(f"[RESEARCHER]   Coverage complete — both math and semantic types present")

        return {
            "researcher_output":    result,
            "researcher_retry":     should_retry,
            "researcher_iteration": 1,
            "errors": [],
        }
