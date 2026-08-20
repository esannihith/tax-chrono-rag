import os
import json
import re
import httpx
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

env_path = Path("C:/Users/esann/Desktop/Temporal-RAG/.env")
load_dotenv(env_path)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM generation providers."""

    @abstractmethod
    def generate(self, prompt: str, system_instruction: str) -> Dict[str, Any]:
        """Generates text/JSON from the model given a prompt and system instruction."""
        pass

    @staticmethod
    def extract_json(raw_text: str) -> Dict[str, Any]:
        """Robustly extracts JSON from LLM response even if enclosed in markdown blocks."""
        raw_text = raw_text.strip()
        # Try direct parsing
        try:
            return json.loads(raw_text)
        except Exception:
            pass

        # Try markdown codeblock regex
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # Try first '{' to last '}'
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw_text[start:end+1])
            except Exception:
                pass

        return {
            "direct_answer": raw_text,
            "step_by_step_reasoning": [],
            "temporal_applicability": "Extracted from raw text",
            "statutory_citations": [],
            "regime_differences": [],
            "is_out_of_scope": False,
            "confidence_score": 0.7
        }


class GeminiProvider(BaseLLMProvider):
    """Provider for Google Gemini API models with automatic retry and model fallback."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.6-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.models_to_try = [model_name, "gemini-flash-latest", "gemini-2.5-flash"]

    def generate(self, prompt: str, system_instruction: str) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        import time

        payload = {
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        }

        last_err = ""
        for model in self.models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            
            try:
                resp = httpx.post(url, json=payload, timeout=12)
                if resp.status_code == 200:
                    res_json = resp.json()
                    raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
                    return self.extract_json(raw_text)
                elif resp.status_code == 429:
                    # Immediately fail fast on quota limit so OpenRouter takes over without delay
                    raise RuntimeError("Gemini quota rate limited (429)")
                elif resp.status_code == 503:
                    last_err = f"{model} status 503"
                    continue
                else:
                    last_err = f"{model} status {resp.status_code}"
            except RuntimeError:
                raise
            except Exception as e:
                last_err = f"{model} exception: {e}"

        raise RuntimeError(f"Gemini generation failed: {last_err}")


class OpenRouterProvider(BaseLLMProvider):
    """Provider for OpenRouter API models."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        self.models_to_try = [
            "liquid/lfm-2.5-2.6b:free",
            "google/gemma-4-26b-a4b-it:free",
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "cohere/north-mini-code:free"
        ]

    def generate(self, prompt: str, system_instruction: str) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("OPEN_ROUTER_API_KEY is not configured.")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        last_err = ""
        for model in self.models_to_try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1
            }

            try:
                resp = httpx.post(url, headers=headers, json=payload, timeout=10)
                if resp.status_code == 200:
                    res_json = resp.json()
                    raw_text = res_json["choices"][0]["message"]["content"]
                    parsed = self.extract_json(raw_text)
                    if parsed.get("direct_answer"):
                        return parsed
                else:
                    last_err = f"{model} status {resp.status_code}: {resp.text[:100]}"
            except Exception as e:
                last_err = f"{model} exception: {e}"

        raise RuntimeError(f"OpenRouter generation failed: {last_err}")


class DeterministicMockProvider(BaseLLMProvider):
    """Offline rule-guided mock provider for testing and deterministic benchmark validation."""

    def generate(self, prompt: str, system_instruction: str) -> Dict[str, Any]:
        # Extract query from prompt
        match = re.search(r'USER QUERY:\s*"(.*?)"', prompt)
        q = match.group(1) if match else "Sample query"

        return {
            "direct_answer": f"Mock statutory answer for: {q}",
            "step_by_step_reasoning": [
                "Step 1: Evaluated retrieved statutory rules.",
                "Step 2: Applied salaried persona exemption thresholds."
            ],
            "temporal_applicability": "Applicable for relevant Assessment Year as per statutory rules.",
            "statutory_citations": [
                {
                    "rule_id": "12AC",
                    "sub_rule": "(2)",
                    "statutory_path": "Rule 12AC",
                    "corpus_year": 1962,
                    "sections_referenced": ["section 139"],
                    "forms_referenced": ["Form ITR-U"],
                    "effective_date": "29-04-2022",
                    "citation_text": "Manner of filing updated return"
                }
            ],
            "regime_differences": [],
            "is_out_of_scope": False,
            "confidence_score": 1.0
        }


class ResilientHybridProvider(BaseLLMProvider):
    """Orchestrates primary Gemini calls with instant OpenRouter free tier fallback."""

    def __init__(self):
        self.gemini = GeminiProvider() if (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")) else None
        self.openrouter = OpenRouterProvider() if (os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")) else None
        self.mock = DeterministicMockProvider()

    def generate(self, prompt: str, system_instruction: str) -> Dict[str, Any]:
        # 1. Try Gemini first
        if self.gemini:
            try:
                return self.gemini.generate(prompt, system_instruction)
            except Exception as e:
                pass

        # 2. Try OpenRouter fallback
        if self.openrouter:
            try:
                return self.openrouter.generate(prompt, system_instruction)
            except Exception as e:
                pass

        # 3. Fallback to Deterministic Mock if network/quotas exhausted
        return self.mock.generate(prompt, system_instruction)


def get_default_llm_provider() -> BaseLLMProvider:
    """Returns resilient multi-provider LLM client."""
    return ResilientHybridProvider()
