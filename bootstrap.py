"""Assembly point: connect the core repair rules to concrete adapters.

This is the only module that imports both ``core`` and ``adapters``. It also
holds the scripted answer-card sequences, kept apart from the repair loop.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from adapters.models.scripted import ScriptedClient
from adapters.sandbox import runner
from adapters.sandbox.repair_tools import RepairTools
from core.ports import ChatClient
from core.repair_agent import RepairAgent
from core.repair_service import RepairService

PROVIDERS = ("scripted",)

EventCallback = Callable[[str, dict[str, Any]], None]


def _card(tool: str, **fields: str) -> str:
    """Serialize one answer card as the JSON reply a model would send."""
    return json.dumps({"tool": tool, **fields})


def _sequence(test_file: str, source_file: str, fixed_source: str) -> list[str]:
    """Build a read -> failing test -> edit -> passing test answer-card sequence."""
    return [
        _card("read_file", path=test_file, summary="Read the tests"),
        _card("read_file", path=source_file, summary="Read the code under test"),
        _card("run_tests", summary="Reproduce the failure"),
        _card("edit_file", path=source_file, content=fixed_source, summary="Apply a focused fix"),
        _card("run_tests", summary="Verify the repair"),
    ]


# Fixed sources prepared by reading each fixture under evals/.
SCRIPTED_REPLIES: dict[str, list[str]] = {
    "arithmetic": _sequence("test_calculator.py", "calculator.py", '''\
def add(left, right):
    """Return the sum of two numbers."""
    return left + right
'''),
    "balanced_brackets": _sequence("test_brackets.py", "brackets.py", '''\
def is_balanced(text):
    """Return whether parentheses, brackets, and braces are properly nested."""
    opening = "([{"
    closing = ")]}"
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []

    for character in text:
        if character in opening:
            stack.append(character)
        elif character in closing:
            if not stack or stack.pop() != pairs[character]:
                return False

    return not stack
'''),
    "median": _sequence("test_statistics_tools.py", "statistics_tools.py", '''\
def median(values):
    """Return the median value without changing the input sequence."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
'''),
    "multi_function": _sequence("test_text_tools.py", "text_tools.py", '''\
def normalize_name(name):
    """Trim surrounding whitespace and lowercase a supplied name."""
    return name.strip().lower()


def greeting(name):
    """Create a greeting from a normalized name."""
    return f"Hello, {normalize_name(name).title()}!"


def initials(name):
    """Return uppercase initials for the whitespace-separated parts of a name."""
    return "".join(part[0].upper() for part in name.split())
'''),
    "unique_words": _sequence("test_word_tools.py", "word_tools.py", '''\
def unique_words(text):
    """Return lowercase words once, in the order first seen."""
    return list(dict.fromkeys(text.lower().split()))
'''),
}


class CallbackRecorder:
    """TraceRecorder that forwards each event to a callback as it occurs."""

    def __init__(self, callback: EventCallback) -> None:
        self._callback = callback

    def record(self, event: str, payload: dict[str, Any]) -> None:
        """Forward one trace event."""
        self._callback(event, payload)


def make_client(provider: str, case: str) -> ChatClient:
    """Create the chat client for ``provider`` and the selected case."""
    if provider == "scripted":
        if case not in SCRIPTED_REPLIES:
            raise ValueError(f"No scripted answer cards for case: {case}")
        return ScriptedClient(SCRIPTED_REPLIES[case])
    raise ValueError(f"Unknown provider: {provider}. Choose one of: {', '.join(PROVIDERS)}")


def build_service(provider: str, on_event: EventCallback | None = None) -> RepairService:
    """Wire the runner, safe tools, chat client and agent into a RepairService."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Choose one of: {', '.join(PROVIDERS)}")
    recorder = CallbackRecorder(on_event) if on_event else None

    def make_agent(case: str) -> RepairAgent:
        return RepairAgent(make_client(provider, case), recorder=recorder)

    return RepairService(
        evals_dir=runner.EVALS_DIR,
        create_workspace=runner.create_workspace,
        make_tools=RepairTools,
        make_agent=make_agent,
    )
