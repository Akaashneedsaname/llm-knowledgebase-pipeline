"""
Cost controls for the Gemini integration, tested with a fake client (no API key or network).
"""

from types import SimpleNamespace

import pytest

from kb import llm, metadata, qa
from kb.config import settings


class FakeModels:
    def __init__(self, text='{"summary": "s", "topics": ["a"], "genre": "g", "language": "en"}'):
        self.calls = []
        self.text = text

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(
            text=self.text,
            usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=0),
        )


@pytest.fixture
def fake(monkeypatch):
    models = FakeModels()
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_llm_model", "gemini-2.5-flash")
    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(models=models))
    monkeypatch.setattr(llm, "usage", {"calls": 0, "input_tokens": 0, "output_tokens": 0, "blocked": 0})
    qa._answer_cache.clear()
    return models


def test_thinking_disabled_and_output_capped(fake):
    llm.generate("hello", max_output_tokens=123)
    config = fake.calls[0]["config"]
    assert config.thinking_config.thinking_budget == 0
    assert config.max_output_tokens == 123


def test_usage_is_counted(fake):
    llm.generate("a")
    llm.generate_json("b")
    summary = llm.usage_summary()
    assert summary["calls"] == 2
    assert summary["total_tokens"] == 240


def test_call_limit_blocks_further_calls(fake, monkeypatch):
    monkeypatch.setattr(settings, "llm_max_calls", 2)
    results = [llm.generate("x") for _ in range(4)]
    assert results[:2] == [fake.text, fake.text]
    assert results[2:] == [None, None]
    assert len(fake.calls) == 2
    assert llm.usage_summary()["blocked"] == 2


def test_label_excerpt_is_bounded_for_long_documents(fake, monkeypatch):
    monkeypatch.setattr(settings, "llm_label_chars", 3000)
    metadata.llm_labels("word " * 200_000)  # about a 1,000-page book
    prompt = fake.calls[0]["contents"]
    assert len(prompt) < 3600


def test_repeated_question_is_answered_from_cache(fake, monkeypatch):
    chunk = {"chunk_text": "Weena was lost in the forest fire.", "page_start": 3, "page_end": 3,
             "chapter": "Chapter XII", "title": "The Time Machine", "source_file": "tm.pdf", "similarity": 0.8}
    monkeypatch.setattr(qa, "search", lambda query, top_k=None: [chunk])
    fake.text = "She was lost in the fire [1]."

    first = qa.answer("What happened to Weena?")
    second = qa.answer("what happened to weena? ")
    assert first["mode"] == "llm" and second.get("cached") is True
    assert len(fake.calls) == 1


def test_no_key_means_no_calls(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    assert llm.generate("x") is None
    assert llm.generate_json("x") is None
