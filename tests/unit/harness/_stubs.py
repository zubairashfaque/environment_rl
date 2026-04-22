"""Stand-ins for the OpenAI client — tests never hit the real API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class _StubMessage:
    content: str


@dataclass
class _StubChoice:
    message: _StubMessage


@dataclass
class _StubResponse:
    choices: list[_StubChoice]


class _CompletionsStub:
    def __init__(self, parent: "StubOpenAIClient") -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> _StubResponse:
        self._parent.calls.append(kwargs)
        payload = self._parent._next_payload()
        content = json.dumps(payload) if isinstance(payload, dict) else str(payload)
        return _StubResponse(choices=[_StubChoice(_StubMessage(content=content))])


class _ChatStub:
    def __init__(self, parent: "StubOpenAIClient") -> None:
        self.completions = _CompletionsStub(parent)


class StubOpenAIClient:
    """Mimics the subset of ``openai.OpenAI`` the policy uses."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = _ChatStub(self)

    def _next_payload(self) -> dict[str, Any]:
        if not self._responses:
            raise RuntimeError("StubOpenAIClient: no more canned responses")
        return self._responses.pop(0)
