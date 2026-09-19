"""Ollama transport: response contract, request failures, truncation detection."""

from unittest.mock import Mock

import pytest
import requests

import ollama_client
from ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTruncatedError,
)


def make_response(status_code=200, body=None, text="", json_error=False):
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.text = text
    if json_error:
        response.json.side_effect = ValueError("bad json")
    else:
        response.json.return_value = body
    return response


@pytest.fixture
def client():
    return OllamaClient("http://ollama.test/api/generate", "model", 5)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (make_response(body={"response": "  hello  ", "done_reason": "stop"}), "hello"),
        (make_response(body={"response": "no reason field"}), "no reason field"),
    ],
)
def test_generate_returns_text(client, monkeypatch, response, expected):
    monkeypatch.setattr(ollama_client.requests, "post", Mock(return_value=response))
    assert client.generate("prompt", 10) == expected


@pytest.mark.parametrize(
    ("response", "exception", "match"),
    [
        (make_response(status_code=500, text="boom"), OllamaResponseError, "HTTP 500"),
        (make_response(status_code=404, text="x"), OllamaResponseError, "HTTP 404"),
        (make_response(json_error=True), OllamaResponseError, "non-JSON"),
        (make_response(body=["list"]), OllamaResponseError, "missing"),
        (make_response(body={"other": 1}), OllamaResponseError, "missing"),
        (make_response(body={"response": 42}), OllamaResponseError, "empty"),
        (make_response(body={"response": None}), OllamaResponseError, "empty"),
        (make_response(body={"response": "   "}), OllamaResponseError, "empty"),
        (make_response(body={"response": "cut", "done_reason": "length"}), OllamaTruncatedError, "cut off"),
    ],
)
def test_generate_response_contract(client, monkeypatch, response, exception, match):
    monkeypatch.setattr(ollama_client.requests, "post", Mock(return_value=response))
    with pytest.raises(exception, match=match):
        client.generate("prompt", 10)


@pytest.mark.parametrize(
    ("raised", "match"),
    [
        (requests.exceptions.ConnectionError("refused"), "Cannot reach Ollama"),
        (requests.exceptions.Timeout("slow"), "did not respond within 5s"),
        (requests.exceptions.RequestException("odd"), "Request to Ollama failed"),
    ],
)
def test_request_failures_are_translated(client, monkeypatch, raised, match):
    monkeypatch.setattr(ollama_client.requests, "post", Mock(side_effect=raised))
    with pytest.raises(OllamaConnectionError, match=match):
        client.generate("prompt", 10)


def test_generate_sends_expected_payload(client, monkeypatch):
    post = Mock(return_value=make_response(body={"response": "ok"}))
    monkeypatch.setattr(ollama_client.requests, "post", post)
    client.generate("the prompt", 77)
    _, kwargs = post.call_args
    assert kwargs["json"] == {
        "model": "model",
        "prompt": "the prompt",
        "stream": False,
        "options": {"num_predict": 77},
    }
    assert kwargs["timeout"] == 5


@pytest.mark.parametrize(("prompt", "max_tokens"), [("", 10), ("   ", 10), ("ok", 0), ("ok", -1)])
def test_generate_rejects_bad_arguments(client, monkeypatch, prompt, max_tokens):
    post = Mock()
    monkeypatch.setattr(ollama_client.requests, "post", post)
    with pytest.raises(ValueError):
        client.generate(prompt, max_tokens)
    post.assert_not_called()


def test_error_body_is_escaped_and_bounded(client, monkeypatch):
    hostile = "\x1b[2J" + "A" * 500 + "\n"
    monkeypatch.setattr(
        ollama_client.requests, "post", Mock(return_value=make_response(status_code=502, text=hostile))
    )
    with pytest.raises(OllamaResponseError) as excinfo:
        client.generate("prompt", 10)
    message = str(excinfo.value)
    assert "\x1b" not in message and "\n" not in message
    assert "\\x1b[2J" in message
    assert message.count("A") <= 200
