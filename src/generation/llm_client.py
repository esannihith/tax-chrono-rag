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
            
            for attempt in range(2):  # 2 attempts per model with backoff
                try:
                    resp = httpx.post(url, json=payload, timeout=25)
                    if resp.status_code == 200:
                        res_json = resp.json()
                        raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
                        return self.extract_json(raw_text)
                    elif resp.status_code in [429, 503]:
                        last_err = f"{model} status {resp.status_code}: {resp.text[:150]}"
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    else:
                        last_err = f"{model} status {resp.status_code}: {resp.text[:150]}"
                        break
                except Exception as e:
                    last_err = f"{model} exception: {e}"
                    time.sleep(1.0)

        # Final attempt without response_mime_type on gemini-flash-latest
        try:
            del payload["generationConfig"]["response_mime_type"]
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={self.api_key}"
            resp = httpx.post(url, json=payload, timeout=25)
            if resp.status_code == 200:
                res_json = resp.json()
                raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
                return self.extract_json(raw_text)
        except Exception:
            pass

        raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")


class OpenRouterProvider(BaseLLMProvider):
    """Provider for OpenRouter API."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "google/gemini-2.0-flash-lite-preview-02-05:free"):
        self.api_key = api_key or os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        self.model_name = model_name

    def generate(self, prompt: str, system_instruction: str) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("OPEN_ROUTER_API_KEY is not configured.")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        resp = httpx.post(url, headers=headers, json=payload, timeout=25)
        if resp.status_code == 200:
            res_json = resp.json()
            raw_text = res_json["choices"][0]["message"]["content"]
            return self.extract_json(raw_text)

        raise RuntimeError(f"OpenRouter generation failed: {resp.status_code} - {resp.text[:200]}")


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


def get_default_llm_provider() -> BaseLLMProvider:
    """Auto-detects available API keys and returns the best provider."""
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        return GeminiProvider(api_key=gemini_key)

    openrouter_key = os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return OpenRouterProvider(api_key=openrouter_key)

    return DeterministicMockProvider()
