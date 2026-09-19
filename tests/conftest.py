"""Shared fixtures: make project modules importable and provide a fake Ollama client."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ollama_client import OllamaTruncatedError  # noqa: E402

TRUNCATED = object()


class FakeOllamaClient:
    """Scripted client: each generate() call pops the next scripted outcome."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.calls.append((prompt, max_tokens))
        if not self.outcomes:
            raise AssertionError("FakeOllamaClient ran out of scripted outcomes")
        outcome = self.outcomes.pop(0)
        if outcome is TRUNCATED:
            raise OllamaTruncatedError(max_tokens)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def fake_client_factory():
    return FakeOllamaClient
