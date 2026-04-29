#!/usr/bin/env python
"""Test which research tools are available."""

import sys

print("[TEST] Checking DuckDuckGo...")
try:
    from duckduckgo_search import DDGS
    print("[TEST] ✓ duckduckgo_search installed")
except ImportError as e:
    print(f"[TEST] ✗ duckduckgo_search missing: {e}")

print("\n[TEST] Checking Wikipedia...")
try:
    import wikipedia
    print("[TEST] ✓ wikipedia installed")
except ImportError as e:
    print(f"[TEST] ✗ wikipedia missing: {e}")

print("\n[TEST] Checking Arxiv...")
try:
    import arxiv
    print("[TEST] ✓ arxiv installed")
except ImportError as e:
    print(f"[TEST] ✗ arxiv missing: {e}")

print("\n[TEST] Checking LangChain Community tools...")
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    print("[TEST] ✓ DuckDuckGoSearchRun available")
except Exception as e:
    print(f"[TEST] ✗ DuckDuckGoSearchRun: {e}")

try:
    from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
    print("[TEST] ✓ WikipediaQueryRun available")
except Exception as e:
    print(f"[TEST] ✗ WikipediaQueryRun: {e}")

try:
    from langchain_community.tools.arxiv.tool import ArxivQueryRun
    print("[TEST] ✓ ArxivQueryRun available")
except Exception as e:
    print(f"[TEST] ✗ ArxivQueryRun: {e}")
