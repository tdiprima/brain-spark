"""HTTP transport for the OpenAI Responses API with typed exceptions."""

import logging

import requests

logger = logging.getLogger("openai_client")
RESPONSES_URL = "https://api.openai.com/v1/responses"


class OpenAIError(Exception):
    """Base class for OpenAI request failures."""


class OpenAIConnectionError(OpenAIError):
    """OpenAI is unreachable or the request timed out."""


class OpenAIResponseError(OpenAIError):
    """OpenAI returned an error, refusal, or malformed response."""


class OpenAITruncatedError(OpenAIError):
    """Generation exhausted its output token budget."""

    def __init__(self, max_tokens: int) -> None:
        super().__init__(f"OpenAI output was cut off at the {max_tokens}-token limit")
        self.max_tokens = max_tokens


class OpenAIClient:
    """Generate text using an API key supplied by the application's configuration."""

    def __init__(self, api_key: str, model: str, timeout_seconds: int) -> None:
        if not api_key or any(not 33 <= ord(char) <= 126 for char in api_key):
            raise ValueError("OPENAI_API_KEY must be a non-empty ASCII token without whitespace")
        self._api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, max_tokens: int) -> str:
        """Return completed assistant text; incomplete output is never accepted."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        payload = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": "none"},
            "stream": False,
            "store": False,
        }
        logger.info("openai_request model=%s max_tokens=%d", self.model, max_tokens)
        try:
            response = requests.post(
                RESPONSES_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=self.timeout_seconds,
                allow_redirects=False,
            )
        except requests.exceptions.Timeout as error:
            raise OpenAIConnectionError(
                f"OpenAI did not respond within {self.timeout_seconds}s"
            ) from error
        except requests.exceptions.ConnectionError as error:
            raise OpenAIConnectionError("Cannot reach OpenAI. Check your network connection.") from error
        except requests.exceptions.RequestException as error:
            # Request exceptions can include headers; never expose their raw text.
            raise OpenAIConnectionError("Request to OpenAI failed") from error

        try:
            return _extract_response_text(response, max_tokens)
        finally:
            response.close()


def _extract_response_text(response: requests.Response, max_tokens: int) -> str:
    """Read assistant message content, skipping reasoning and other output items."""
    if response.status_code != 200:
        # API errors may echo part of an invalid key. Do not log or display the body.
        hints = {
            401: "Check OPENAI_API_KEY.",
            403: "Check API key permissions and model access.",
            429: "Check your API quota or retry later.",
        }
        hint = hints.get(response.status_code, "Please retry or check your OpenAI configuration.")
        raise OpenAIResponseError(f"OpenAI returned HTTP {response.status_code}. {hint}")
    try:
        body = response.json()
    except ValueError as error:
        raise OpenAIResponseError("OpenAI returned a non-JSON body") from error
    if not isinstance(body, dict):
        raise OpenAIResponseError("OpenAI returned a malformed response")

    if body.get("status") == "incomplete":
        details = body.get("incomplete_details")
        if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
            raise OpenAITruncatedError(max_tokens)
        raise OpenAIResponseError("OpenAI returned incomplete output")
    if body.get("status") != "completed" or body.get("error") is not None:
        raise OpenAIResponseError("OpenAI did not complete the response")

    output = body.get("output")
    if not isinstance(output, list):
        raise OpenAIResponseError("OpenAI response is missing the output list")
    parts = []
    for item in output:
        if not isinstance(item, dict):
            raise OpenAIResponseError("OpenAI returned a malformed output item")
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if item.get("role") != "assistant" or not isinstance(content, list):
            raise OpenAIResponseError("OpenAI returned a malformed assistant message")
        for part in content:
            if not isinstance(part, dict):
                raise OpenAIResponseError("OpenAI returned malformed message content")
            if part.get("type") == "refusal":
                raise OpenAIResponseError("OpenAI declined to generate this content")
            if part.get("type") == "output_text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise OpenAIResponseError("OpenAI returned non-text output")
                parts.append(text)
    text = "".join(parts).strip()
    if not text:
        raise OpenAIResponseError("OpenAI returned an empty response")
    return text
