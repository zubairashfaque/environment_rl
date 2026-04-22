"""Tests for PromptDistiller."""

import pytest

from env_rl.harness.prompt_tuning.distiller import PromptDistiller
from tests.unit.harness._stubs import StubOpenAIClient


class _PlainTextStubResponse:
    """A stub response whose message.content is a plain string (not JSON)."""
    def __init__(self, content: str) -> None:
        self.choices = [_StubChoice(content)]
        self.usage = None


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message = _StubMessage(content)


class _StubMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _PlainTextStubClient:
    """Variant of StubOpenAIClient that doesn't JSON-serialize the response."""
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list = []
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _PlainTextStubResponse(self._text)


def test_distiller_returns_prompt_edit_with_distillation_technique() -> None:
    bloated = "A " * 500  # 1000 chars of repeated noise
    client = _PlainTextStubClient("A concise version")
    d = PromptDistiller(client=client, model="gpt-4o-mini")
    edit = d.distill(bloated)
    assert edit.technique == "distillation"
    assert "A concise version" in edit.addition
    assert "shorter" in edit.rationale or "chars" in edit.rationale


def test_distiller_uses_given_model_and_temperature() -> None:
    client = _PlainTextStubClient("short")
    d = PromptDistiller(client=client, model="gpt-4o", temperature=0.0)
    d.distill("bloated prompt")
    assert client.calls[0]["model"] == "gpt-4o"
    assert client.calls[0]["temperature"] == 0.0
