import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, DEFAULT_MODEL, TEMPERATURE


class BaseAgent(ABC):
    """
    Shared LLM interface using OpenRouter.
    Handles common failure modes: looping content, auth errors, timeouts.
    """

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = TEMPERATURE):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=OPENAI_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

    def call_llm(self, prompt: str, fallback: Optional[str] = None, stream: bool = True) -> str:
        """
        Invoke the LLM with graceful error handling.
        
        If stream=True, outputs tokens in real-time as they arrive.
        If stream=False, waits for full response then returns it.

        OpenRouter-specific errors handled:
          • 400 "looping content" — strip repetitive sections and retry once
          • 401 auth errors       — re-raise with a clear message
          • Any other exception   — return fallback or empty string
        """
        try:
            if stream:
                return self._call_llm_streaming(prompt)
            else:
                response = self.llm.invoke(prompt)
                return response.content

        except Exception as exc:
            err_str = str(exc).lower()

            # ── OpenRouter loop-detection (400) ─────────────────────
            if "looping" in err_str or "loop" in err_str:
                print(
                    "\n[BASE] ⚠  OpenRouter loop-detection triggered. "
                    "Retrying with condensed prompt..."
                )
                condensed = self._condense_prompt(prompt)
                try:
                    if stream:
                        return self._call_llm_streaming(condensed)
                    else:
                        response = self.llm.invoke(condensed)
                        return response.content
                except Exception as exc2:
                    print(f"[BASE] ✗ Retry also failed: {exc2}")
                    return fallback or ""

            # ── Auth errors — nothing we can do except tell the user ─
            if "401" in err_str or "authentication" in err_str or "user not found" in err_str:
                raise  # Let this bubble up — user must fix the API key

            # ── All other errors (rate limit, timeout, etc.) ─────────
            print(f"[BASE] ✗ LLM call failed: {exc}")
            return fallback or ""

    def _call_llm_streaming(self, prompt: str) -> str:
        """
        Stream LLM output token-by-token, printing as it arrives.
        Collects and returns the full response.
        """
        full_response = ""
        print("\n[THINKING] 🤔 Agent is reasoning...\n")
        
        try:
            for chunk in self.llm.stream(prompt):
                token = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if token:
                    print(token, end="", flush=True)
                    full_response += token
            print("\n")  # newline after streaming completes
            return full_response
        except Exception as e:
            print(f"\n[BASE] Streaming failed, falling back to invoke: {e}")
            response = self.llm.invoke(prompt)
            return response.content

    def _condense_prompt(self, prompt: str) -> str:
        """
        Remove repetitive sections from a prompt that triggered loop detection.
        Specifically:
          - Truncates long JSON blocks
          - Removes duplicate row narratives
          - Keeps the instruction part intact
        """
        lines = prompt.splitlines()
        seen   = set()
        deduped = []
        for line in lines:
            stripped = line.strip()
            # Drop near-duplicate lines (e.g. repeated "Row N | ..." lines)
            key = re.sub(r"\b\d+\b", "N", stripped)  # normalise numbers
            if key in seen and stripped:
                continue
            seen.add(key)
            deduped.append(line)

        condensed = "\n".join(deduped)

        # Hard cap on total length
        if len(condensed) > 6000:
            condensed = condensed[:6000] + "\n...[truncated for brevity]"

        return condensed

    def parse_json(self, raw: str) -> Dict[str, Any]:
        """Robust JSON extraction from model output."""

        if not raw:
            return {"_parse_error": True, "raw": ""}

        # Strip markdown fences
        cleaned = re.sub(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```",
            r"\1",
            raw,
            flags=re.DOTALL,
        ).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find a JSON object anywhere in the text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {
            "_parse_error": True,
            "raw": raw[:1000],
        }

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass