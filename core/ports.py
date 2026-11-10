"""Protocols describing the outside services the repair agent depends on."""

from __future__ import annotations

from typing import Any, Protocol

from core.models import TestResult


class ChatClient(Protocol):
    """A language model client that answers a list of chat messages with text."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Return the model's reply to the given conversation."""
        ...


class RepairToolsPort(Protocol):
    """The only operations the agent may perform on the project under repair."""

    def read_file(self, path: str) -> str:
        """Return the text of a file inside the workspace."""
        ...

    def edit_file(self, path: str, content: str) -> str:
        """Replace a workspace file's content and return a short observation."""
        ...

    def run_tests(self) -> TestResult:
        """Run the project's tests and return the result."""
        ...


class TraceRecorder(Protocol):
    """Optional sink for trace events emitted during a repair attempt."""

    def record(self, event: str, payload: dict[str, Any]) -> None:
        """Record one named event with its payload."""
        ...
