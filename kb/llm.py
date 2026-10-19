"""
Thin wrapper around the Gemini API for text generation.

The LLM is optional: when no GEMINI_API_KEY is set, every function returns
None and the pipeline falls back to non-LLM behavior (keyword labels,
extractive answers).
"""

import json
from functools import lru_cache
from typing import Optional

from kb.config import settings


def enabled() -> bool:
    return settings.llm_enabled


@lru_cache(maxsize=1)
def _client():
    from google import genai
    return genai.Client(api_key=settings.gemini_api_key)


def generate(prompt: str) -> Optional[str]:
    """Generate text. Returns None if the LLM is disabled or the call fails."""
    if not enabled():
        return None
    try:
        response = _client().models.generate_content(model=settings.gemini_llm_model, contents=prompt)
        return response.text
    except Exception as e:
        print(f"[llm] generation failed: {e}")
        return None


def generate_json(prompt: str) -> Optional[dict]:
    """Generate a JSON object. Returns None if disabled, failed, or not valid JSON."""
    if not enabled():
        return None
    try:
        from google.genai import types
        response = _client().models.generate_content(
            model=settings.gemini_llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"[llm] JSON generation failed: {e}")
        return None
