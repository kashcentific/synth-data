import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, DEFAULT_MODEL, TEMPERATURE


class BaseAgent(ABC):
    """
    Shared LLM interface using OpenRouter
    """

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = TEMPERATURE):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=OPENAI_API_KEY,

            # IMPORTANT: OpenRouter endpoint
            base_url="https://openrouter.ai/api/v1",
        )

    def call_llm(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content

    def parse_json(self, raw: str) -> Dict[str, Any]:
        """
        Robust JSON extraction from model output
        """

        cleaned = re.sub(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```",
            r"\1",
            raw,
            flags=re.DOTALL
        ).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {
            "_parse_error": True,
            "raw": raw[:1000]
        }

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass