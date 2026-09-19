"""Shared fixtures: make project modules importable and provide a fake OpenAI client."""

import sys
from pathlib import Path

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai_client import OpenAITruncatedError  # noqa: E402

TRUNCATED = object()


@pytest.fixture(autouse=True)
def block_real_http(monkeypatch):
    """Tests must never use a real API key or incur API charges."""
    def unexpected_request(*args, **kwargs):
        raise AssertionError("Real HTTP requests are disabled in tests")

    monkeypatch.setattr(requests.sessions.Session, "request", unexpected_request)


class FakeOpenAIClient:
    """Scripted client: each generate() call pops the next scripted outcome."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate(self, prompt: str, max_tokens: int) -> str:
        self.calls.append((prompt, max_tokens))
        if not self.outcomes:
            raise AssertionError("FakeOpenAIClient ran out of scripted outcomes")
        outcome = self.outcomes.pop(0)
        if outcome is TRUNCATED:
            raise OpenAITruncatedError(max_tokens)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def fake_client_factory():
    return FakeOpenAIClient
