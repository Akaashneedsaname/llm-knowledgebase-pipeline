"""
Thin wrapper around the Gemini API for text generation.

The LLM is optional: when no GEMINI_API_KEY is set, every function returns
None and the pipeline falls back to non-LLM behavior (keyword labels,
extractive answers).

Cost controls:
- Only two kinds of call exist: one per document (labels) and one per "Ask".
  Search, chunking, and local embeddings never call the LLM.
- "Thinking" is disabled on Flash models; these are short extraction and
  summarization tasks that don't need it, and thinking tokens are billed.
- Output length is capped per call.
- A per-process call limit stops runaway usage (LLM_MAX_CALLS).
- No automatic retries: a failed call falls back instead of repeating.
- Token usage is counted so the UI and CLI can show it.
"""

import json
import threading
from functools import lru_cache
from typing import Optional

from kb.config import settings

_lock = threading.Lock()
usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "blocked": 0}
last_error: Optional[str] = None  # most recent failed call, shown in the UI


def enabled() -> bool:
    return settings.llm_enabled


def usage_summary() -> dict:
    with _lock:
        return {**usage, "total_tokens": usage["input_tokens"] + usage["output_tokens"]}


@lru_cache(maxsize=1)
def _client():
    from google import genai
    return genai.Client(api_key=settings.gemini_api_key)


def _config(max_output_tokens: int, json_mode: bool = False):
    from google.genai import types
    kwargs = {"max_output_tokens": max_output_tokens, "temperature": 0.2}
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    if "flash" in settings.gemini_llm_model:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)


def _call(prompt: str, max_output_tokens: int, json_mode: bool = False) -> Optional[str]:
    """Make one LLM call, enforcing the call limit and recording token usage."""
    if not enabled():
        return None
    with _lock:
        if usage["calls"] >= settings.llm_max_calls:
            usage["blocked"] += 1
            print(f"[llm] call limit reached ({settings.llm_max_calls}); falling back without the LLM")
            return None
        usage["calls"] += 1

    try:
        response = _client().models.generate_content(
            model=settings.gemini_llm_model,
            contents=prompt,
            config=_config(max_output_tokens, json_mode),
        )
    except Exception as e:
        global last_error
        last_error = str(e)[:300]
        print(f"[llm] call failed: {e}")
        return None

    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        with _lock:
            usage["input_tokens"] += getattr(meta, "prompt_token_count", 0) or 0
            usage["output_tokens"] += (getattr(meta, "candidates_token_count", 0) or 0) + \
                                      (getattr(meta, "thoughts_token_count", 0) or 0)
    return response.text


def generate(prompt: str, max_output_tokens: int = 800) -> Optional[str]:
    """Generate text. Returns None if the LLM is disabled, over the limit, or the call fails."""
    return _call(prompt, max_output_tokens)


def generate_json(prompt: str, max_output_tokens: int = 400) -> Optional[dict]:
    """Generate a JSON object. Returns None if disabled, failed, or not valid JSON."""
    text = _call(prompt, max_output_tokens, json_mode=True)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("[llm] response was not valid JSON; using keyword labels instead")
        return None
