"""OpenAI Responses contract, bearer authentication, errors, and truncation."""

from unittest.mock import Mock

import pytest
import requests

import openai_client
from openai_client import OpenAIClient, OpenAIConnectionError, OpenAIResponseError, OpenAITruncatedError


def completed(text="hello"):
    return {
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": text},
        ]}],
    }


def make_response(body=None, status=200):
    response = Mock(spec=requests.Response)
    response.status_code = status
    response.json.return_value = body
    return response


@pytest.fixture
def client():
    return OpenAIClient("test-key", "gpt-5.2", 5)


def test_generate_authenticates_and_sends_responses_payload(client, monkeypatch):
    response = make_response(completed("  hello  "))
    post = Mock(return_value=response)
    monkeypatch.setattr(openai_client.requests, "post", post)
    assert client.generate("the prompt", 77) == "hello"
    post.assert_called_once_with(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": "Bearer test-key"},
        json={"model": "gpt-5.2", "input": "the prompt", "max_output_tokens": 77,
              "reasoning": {"effort": "none"}, "stream": False, "store": False},
        timeout=5,
        allow_redirects=False,
    )
    response.close.assert_called_once()


def test_extracts_all_text_after_reasoning(client, monkeypatch):
    body = completed("first ")
    body["output"].insert(0, {"type": "reasoning", "summary": []})
    body["output"][1]["content"].append({"type": "output_text", "text": "second"})
    body["output"].append(completed(" third")["output"][0])
    monkeypatch.setattr(openai_client.requests, "post", Mock(return_value=make_response(body)))
    assert client.generate("prompt", 30) == "first second third"


@pytest.mark.parametrize(("body", "exception", "match"), [
    ([], OpenAIResponseError, "malformed"),
    ({}, OpenAIResponseError, "did not complete"),
    ({"status": "failed", "error": {"message": "failure"}}, OpenAIResponseError, "did not complete"),
    ({"status": "in_progress"}, OpenAIResponseError, "did not complete"),
    ({"status": "completed", "error": {"message": "failure"}}, OpenAIResponseError, "did not complete"),
    ({"status": "completed"}, OpenAIResponseError, "output list"),
    ({"status": "completed", "output": {}}, OpenAIResponseError, "output list"),
    ({"status": "completed", "output": []}, OpenAIResponseError, "empty"),
    ({"status": "completed", "output": [None]}, OpenAIResponseError, "malformed"),
    (completed(None), OpenAIResponseError, "non-text"),
    (completed(42), OpenAIResponseError, "non-text"),
    (completed("  "), OpenAIResponseError, "empty"),
    ({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []},
     OpenAITruncatedError, "cut off"),
    ({"status": "incomplete", "incomplete_details": {"reason": "content_filter"}},
     OpenAIResponseError, "incomplete"),
    ({"status": "incomplete", "incomplete_details": None}, OpenAIResponseError, "incomplete"),
])
def test_generate_response_contract(client, monkeypatch, body, exception, match):
    response = make_response(body)
    monkeypatch.setattr(openai_client.requests, "post", Mock(return_value=response))
    with pytest.raises(exception, match=match):
        client.generate("prompt", 30)
    response.close.assert_called_once()


@pytest.mark.parametrize("content", [None, [None], [{"type": "refusal", "refusal": "No"}]])
def test_rejects_malformed_or_refused_messages(client, monkeypatch, content):
    body = completed()
    body["output"][0]["content"] = content
    monkeypatch.setattr(openai_client.requests, "post", Mock(return_value=make_response(body)))
    with pytest.raises(OpenAIResponseError):
        client.generate("prompt", 30)


def test_invalid_json(client, monkeypatch):
    response = make_response()
    response.json.side_effect = ValueError("bad json")
    monkeypatch.setattr(openai_client.requests, "post", Mock(return_value=response))
    with pytest.raises(OpenAIResponseError, match="non-JSON"):
        client.generate("prompt", 30)
    response.close.assert_called_once()


@pytest.mark.parametrize("status", [302, 401, 403, 404, 429, 500])
def test_http_errors_do_not_expose_keys_or_raw_body(client, monkeypatch, caplog, status):
    response = make_response(status=status)
    response.text = "Invalid API key test-key\x1b[2J\nforged log line"
    monkeypatch.setattr(openai_client.requests, "post", Mock(return_value=response))
    with pytest.raises(OpenAIResponseError, match=f"HTTP {status}") as error:
        client.generate("prompt", 30)
    assert "test-key" not in str(error.value) + caplog.text
    assert "\x1b" not in str(error.value)
    response.close.assert_called_once()


@pytest.mark.parametrize(("raised", "match"), [
    (requests.exceptions.ConnectionError("test-key"), "Cannot reach OpenAI"),
    (requests.exceptions.Timeout("test-key"), "within 5s"),
    (requests.exceptions.RequestException("test-key"), "Request to OpenAI failed"),
])
def test_request_failures_are_translated(client, monkeypatch, caplog, raised, match):
    monkeypatch.setattr(openai_client.requests, "post", Mock(side_effect=raised))
    with pytest.raises(OpenAIConnectionError, match=match) as error:
        client.generate("prompt", 30)
    assert "test-key" not in str(error.value) + caplog.text


@pytest.mark.parametrize(("prompt", "max_tokens"), [("", 30), ("   ", 30), ("ok", 0), ("ok", -1)])
def test_generate_rejects_bad_arguments(client, monkeypatch, prompt, max_tokens):
    post = Mock()
    monkeypatch.setattr(openai_client.requests, "post", post)
    with pytest.raises(ValueError):
        client.generate(prompt, max_tokens)
    post.assert_not_called()
