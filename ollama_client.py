"""HTTP transport for the Ollama generate API with typed exceptions."""

import logging

import requests

logger = logging.getLogger("ollama_client")


class OllamaError(Exception):
    """Base class for Ollama transport failures."""


class OllamaConnectionError(OllamaError):
    """Ollama is unreachable or timed out."""


class OllamaResponseError(OllamaError):
    """Ollama answered with an error status or malformed body."""


class OllamaTruncatedError(OllamaError):
    """Generation stopped because the token budget ran out."""

    def __init__(self, max_tokens: int) -> None:
        super().__init__(f"Ollama output was cut off at the {max_tokens}-token limit")
        self.max_tokens = max_tokens


DONE_REASON_LENGTH = "length"


class OllamaClient:
    """Thin wrapper around POST /api/generate."""

    def __init__(self, url: str, model: str, timeout_seconds: int) -> None:
        self.url = url
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Return the model's full response text for a prompt."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        logger.info("ollama_request model=%s max_tokens=%d", self.model, max_tokens)

        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout_seconds)
        except requests.exceptions.ConnectionError as error:
            logger.error("ollama_unreachable url=%s", self.url)
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.url}. Is it running?"
            ) from error
        except requests.exceptions.Timeout as error:
            logger.error("ollama_timeout url=%s timeout=%d", self.url, self.timeout_seconds)
            raise OllamaConnectionError(
                f"Ollama did not respond within {self.timeout_seconds}s"
            ) from error
        except requests.exceptions.RequestException as error:
            logger.error("ollama_request_failed url=%s error=%s", self.url, error)
            raise OllamaConnectionError(f"Request to Ollama failed: {error}") from error

        return _extract_response_text(response, max_tokens)


def _extract_response_text(response: requests.Response, max_tokens: int) -> str:
    """Validate the HTTP response and pull out the generated text."""
    if response.status_code != 200:
        logger.error("ollama_bad_status status=%d", response.status_code)
        raise OllamaResponseError(
            f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
        )
    try:
        body = response.json()
    except ValueError as error:
        logger.error("ollama_invalid_json")
        raise OllamaResponseError("Ollama returned a non-JSON body") from error

    if not isinstance(body, dict) or "response" not in body:
        raise OllamaResponseError("Ollama response is missing the 'response' field")
    text = body["response"]
    if not isinstance(text, str) or not text.strip():
        raise OllamaResponseError("Ollama returned an empty response")
    if body.get("done_reason") == DONE_REASON_LENGTH:
        logger.warning("ollama_truncated max_tokens=%d", max_tokens)
        raise OllamaTruncatedError(max_tokens)
    return text.strip()
