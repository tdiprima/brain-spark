import requests
from config import OLLAMA_URL, MODEL


class OllamaConnectionError(RuntimeError):
    pass


class OllamaTimeoutError(RuntimeError):
    pass


class OllamaHTTPError(RuntimeError):
    pass


def query_ollama(prompt, max_tokens=1500):
    """Send prompt to Ollama and return text response."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.ConnectionError as err:
        raise OllamaConnectionError("Can't reach Ollama. Is it running?") from err
    except requests.Timeout as err:
        raise OllamaTimeoutError("Ollama timed out.") from err
    except requests.HTTPError as err:
        raise OllamaHTTPError(f"Ollama returned {err}") from err
