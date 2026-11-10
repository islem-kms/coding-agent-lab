"""Tests for the bounded repair loop, using ScriptedClient and fake tools only."""

from __future__ import annotations

import json

import pytest

from adapters.models.scripted import ScriptedClient
from core.models import TestResult
from core.repair_agent import InvalidReply, RepairAgent, parse_action

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
FAIL = TestResult(False, "1 failed", 1)
PASS = TestResult(True, "1 passed", 0)


class FakeTools:
    """In-memory RepairToolsPort that records calls and returns queued test results."""

    def __init__(self, test_results: list[TestResult]) -> None:
        self.files = {"calculator.py": BUGGY, "test_calculator.py": "def test_add(): ..."}
        self.test_results = list(test_results)
        self.calls: list[tuple[str, ...]] = []

    def read_file(self, path: str) -> str:
        self.calls.append(("read_file", path))
        if path not in self.files:
            raise FileNotFoundError(f"No such file in workspace: {path}")
        return self.files[path]

    def edit_file(self, path: str, content: str) -> str:
        self.calls.append(("edit_file", path))
        self.files[path] = content
        return f"Wrote {len(content)} characters to {path}."

    def run_tests(self) -> TestResult:
        self.calls.append(("run_tests",))
        return self.test_results.pop(0)


def card(tool: str, **fields: str) -> str:
    """Build one answer-card JSON reply."""
    return json.dumps({"tool": tool, **fields})


READ = card("read_file", path="calculator.py", summary="Read the code")
TEST = card("run_tests", summary="Run tests")
EDIT = card("edit_file", path="calculator.py", content=FIXED)


def run(replies: list[str], tools: FakeTools, budget: int = 8):
    agent = RepairAgent(ScriptedClient(replies), max_decisions=budget)
    return agent.run("arithmetic", ["calculator.py", "test_calculator.py"], tools)


def events(result) -> list[str]:
    return [entry["event"] for entry in result.trace]


def test_successful_read_fail_edit_pass_sequence() -> None:
    """read → failing test → edit → passing test is reported as a repair."""
    tools = FakeTools([FAIL, PASS])
    result = run([READ, TEST, EDIT, TEST], tools)

    assert result.passed is True
    assert result.changed_files == ["calculator.py"]
    assert result.last_test_result == PASS
    assert tools.files["calculator.py"] == FIXED
    assert tools.calls == [("read_file", "calculator.py"), ("run_tests",),
                           ("edit_file", "calculator.py"), ("run_tests",)]
    assert events(result)[0] == "repair_started"
    assert events(result)[-1] == "repair_finished"
    assert events(result).count("model_response") == 4
    assert events(result).count("tool_result") == 4
    assert result.trace[1]["summary"] == "Read the code"


def test_conversation_history_grows_with_replies_and_observations() -> None:
    """Each accepted reply and each tool observation is added to the history."""
    client = ScriptedClient([READ, TEST, EDIT, TEST])
    RepairAgent(client).run("arithmetic", ["calculator.py"], FakeTools([FAIL, PASS]))

    last = client.calls[-1]
    assert [m["role"] for m in last] == ["system", "user"] + ["assistant", "user"] * 3
    assert "arithmetic" in last[1]["content"]
    assert BUGGY in last[3]["content"]


@pytest.mark.parametrize("raw", [
    "I will read the file first.",
    '```json\n{"tool":"run_tests"}\n```',
    '[{"tool":"run_tests"}]',
    '{"tool":"run_tests"',
    '{"tool":"run_tests","tool":"read_file"}',
    '{"tool":"shell","command":"ls"}',
    '{"tool":"read_file"}',
    '{"tool":"run_tests","path":"x.py"}',
    '{"tool":"run_tests"} {"tool":"run_tests"}',
])
def test_invalid_reply_calls_no_tool(raw: str) -> None:
    """Prose, fences, arrays, bad JSON, duplicates, unknown tools and bad fields fail."""
    with pytest.raises(InvalidReply):
        parse_action(raw)

    tools = FakeTools([])
    result = run([raw], tools)
    assert result.passed is False
    assert tools.calls == []
    assert "Invalid model reply" in result.trace[-1]["reason"]


def test_edit_before_failing_test_is_rejected_with_visible_policy_error() -> None:
    """An early edit is not executed; the model sees a policy error and continues."""
    tools = FakeTools([FAIL, PASS])
    client = ScriptedClient([READ, EDIT, TEST, EDIT, TEST])
    result = RepairAgent(client).run("arithmetic", ["calculator.py"], tools)

    assert tools.calls[:2] == [("read_file", "calculator.py"), ("run_tests",)]
    rejected = result.trace[4]
    assert rejected["event"] == "tool_result" and rejected["policy_error"] is True
    assert "Policy error" in client.calls[2][-1]["content"]
    assert result.passed is True


def test_budget_exhaustion_fails() -> None:
    """The loop stops after the decision budget without a verified repair."""
    tools = FakeTools([FAIL] * 10)
    result = run([TEST] * 10, tools, budget=8)

    assert result.passed is False
    assert tools.calls.count(("run_tests",)) == 8
    assert "budget" in result.trace[-1]["reason"]
    assert result.trace[-1]["decisions"] == 8


def test_passing_baseline_is_not_a_repair() -> None:
    """Tests that pass before any edit end as a failed outcome."""
    tools = FakeTools([PASS])
    result = run([READ, TEST], tools)

    assert result.passed is False
    assert result.changed_files == []
    assert "no reproducible failure" in result.trace[-1]["reason"]


def test_client_error_still_returns_result() -> None:
    """A client that runs out of replies yields a failed RepairResult."""
    tools = FakeTools([FAIL])
    result = run([READ, TEST], tools)

    assert result.passed is False
    assert result.last_test_result == FAIL
    assert "Model client error" in result.trace[-1]["reason"]


def test_tool_error_becomes_visible_observation() -> None:
    """A tool exception is shown to the model and the loop keeps going."""
    tools = FakeTools([FAIL, PASS])
    missing = card("read_file", path="nope.py")
    result = run([missing, READ, TEST, EDIT, TEST], tools)

    error = result.trace[2]
    assert error["event"] == "tool_result" and error["ok"] is False
    assert "FileNotFoundError" in error["observation"]
    assert result.passed is True
