"""ChatClient backed by a local Ollama server.

It forwards the prepared messages to Ollama's ``/api/chat`` endpoint and returns
only the reply text. JSON output is requested, but every reply is still
validated by ``RepairAgent``; this adapter never parses or trusts the action.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_HOST = "http://127.0.0.1:11434"  # Local server only; no remote endpoints.
DEFAULT_TIMEOUT_SECONDS = 300


class OllamaError(RuntimeError):
    """Raised when the local Ollama server cannot produce a reply."""


class OllamaClient:
    """``ChatClient`` that asks a local Ollama model for the next action."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self.url = host.rstrip("/") + "/api/chat"
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send ``messages`` to Ollama and return the reply text unchanged."""
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            hint = f" Try: ollama pull {self.model}" if exc.code == 404 else ""
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}.{hint}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            reason = getattr(exc, "reason", exc)
            raise OllamaError(f"Cannot reach Ollama at {self.url} ({reason}). Is `ollama serve` running?") from None
        except json.JSONDecodeError:
            raise OllamaError("Ollama sent a response that is not JSON.") from None

        content = (data.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise OllamaError(f"Ollama response has no message content: {data.get('error', data)}")
        return content
