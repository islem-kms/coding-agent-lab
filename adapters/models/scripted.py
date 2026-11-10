"""A deterministic stand-in for a language model: a stack of answer cards.

Each ``chat`` call returns the next pre-written reply. It never calls a model,
Docker, or the Internet.
"""

from __future__ import annotations

from collections.abc import Iterable


class ScriptedClientExhausted(RuntimeError):
    """Raised when ``chat`` is called after every scripted reply was used."""


class ScriptedClient:
    """``ChatClient`` that returns pre-written replies in order."""

    def __init__(self, replies: Iterable[str]) -> None:
        self._replies = list(replies)
        self._index = 0
        self.calls: list[list[dict[str, str]]] = []

    @property
    def remaining(self) -> int:
        """Number of replies not yet returned."""
        return len(self._replies) - self._index

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Record ``messages`` and return the next scripted reply."""
        self.calls.append(list(messages))
        if self._index >= len(self._replies):
            raise ScriptedClientExhausted(
                f"ScriptedClient has no replies left (all {len(self._replies)} were used)."
            )
        reply = self._replies[self._index]
        self._index += 1
        return reply
