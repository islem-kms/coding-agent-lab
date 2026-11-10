"""Tests for the Langfuse trajectory recorder, using a fake client only."""

from __future__ import annotations

import json
import logging

import pytest

import bootstrap
import main
from adapters.models.scripted import ScriptedClient
from adapters.observability import langfuse as lf
from adapters.observability.langfuse import LangfuseTrajectoryRecorder, ObservabilityConfigError
from core.models import TestResult
from core.repair_agent import RepairAgent

BUGGY = "def add(a, b):\n    return a - b  # BUGGY-SOURCE\n"
FIXED = "def add(a, b):\n    return a + b  # FIXED-SOURCE\n"
FAIL = TestResult(False, "E   assert -1 == 5  FULL-TEST-OUTPUT", 1)
PASS = TestResult(True, "2 passed FULL-TEST-OUTPUT", 0)
SECRETS = ("BUGGY-SOURCE", "FIXED-SOURCE", "FULL-TEST-OUTPUT", "You are a careful", "Repair the project")


class FakeObservation:
    """Records the calls a Langfuse observation would receive."""

    def __init__(self, log: list, name: str, **kwargs) -> None:
        self.log, self.name, self.kwargs, self.ended = log, name, kwargs, False
        log.append(("start", name, kwargs))

    def start_observation(self, *, name: str, **kwargs) -> FakeObservation:
        return FakeObservation(self.log, name, **kwargs)

    def update(self, **kwargs) -> FakeObservation:
        self.log.append(("update", self.name, kwargs))
        return self

    def end(self) -> FakeObservation:
        self.ended = True
        self.log.append(("end", self.name, {}))
        return self


class FakeLangfuse:
    """Stand-in for ``langfuse.Langfuse``; never touches the network."""

    def __init__(self) -> None:
        self.log: list = []

    def start_observation(self, *, name: str, **kwargs) -> FakeObservation:
        return FakeObservation(self.log, name, **kwargs)

    def flush(self) -> None:
        self.log.append(("flush", "", {}))


class BrokenLangfuse(FakeLangfuse):
    """A client whose every call fails, as with bad credentials or no network."""

    def start_observation(self, **kwargs):
        raise ConnectionError("Langfuse unreachable")

    def flush(self) -> None:
        raise ConnectionError("Langfuse unreachable")


class FakeTools:
    """In-memory sandbox tools with queued test results."""

    def __init__(self) -> None:
        self.files = {"calculator.py": BUGGY}
        self.results = [FAIL, PASS]

    def read_file(self, path: str) -> str:
        return self.files[path]

    def edit_file(self, path: str, content: str) -> str:
        self.files[path] = content
        return "ok"

    def run_tests(self) -> TestResult:
        return self.results.pop(0)


def card(tool: str, **fields: str) -> str:
    return json.dumps({"tool": tool, **fields})


REPLIES = [
    card("read_file", path="calculator.py", summary="Read the code"),
    card("run_tests", summary="Reproduce the failure"),
    card("edit_file", path="calculator.py", content=FIXED, summary="Fix addition"),
    card("run_tests", summary="Verify the repair"),
]


def repair(recorder):
    return RepairAgent(ScriptedClient(REPLIES), recorder=recorder).run(
        "arithmetic", ["calculator.py"], FakeTools())


@pytest.fixture(autouse=True)
def no_real_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any attempt to build a real Langfuse client fail the test."""
    def refuse():
        raise AssertionError("real Langfuse client requested")
    monkeypatch.setattr(bootstrap, "create_langfuse_client", refuse)


def test_event_order_and_flush_on_repair_finished() -> None:
    """One root agent observation, closed children in order, then end and flush."""
    client = FakeLangfuse()
    result = repair(LangfuseTrajectoryRecorder(client))
    assert result.passed is True

    calls = [(kind, name) for kind, name, _ in client.log]
    assert calls == [
        ("start", "repair"),
        ("start", "model_response #1"), ("end", "model_response #1"),
        ("start", "tool_result #1 read_file"), ("end", "tool_result #1 read_file"),
        ("start", "model_response #2"), ("end", "model_response #2"),
        ("start", "tool_result #2 run_tests"), ("end", "tool_result #2 run_tests"),
        ("start", "model_response #3"), ("end", "model_response #3"),
        ("start", "tool_result #3 edit_file"), ("end", "tool_result #3 edit_file"),
        ("start", "model_response #4"), ("end", "model_response #4"),
        ("start", "tool_result #4 run_tests"), ("end", "tool_result #4 run_tests"),
        ("update", "repair"), ("end", "repair"),
        ("flush", ""),
    ]
    root = client.log[0][2]
    assert root["as_type"] == "agent" and root["input"] == {"case": "arithmetic"}
    final = client.log[-3][2]["output"]
    assert final == {"outcome": "repaired", "passed": True, "decisions": 4, "changed_files": ["calculator.py"]}


def test_payload_is_small_and_public() -> None:
    """Steps, tools, summaries, status and return codes are sent; content never is."""
    client = FakeLangfuse()
    repair(LangfuseTrajectoryRecorder(client))

    sent = json.dumps([kwargs for _, _, kwargs in client.log])
    for secret in SECRETS:
        assert secret not in sent
    outputs = {name: kwargs.get("output") for kind, name, kwargs in client.log if kind == "start"}
    assert outputs["model_response #3"] == {"step": 3, "valid": True, "tool": "edit_file", "summary": "Fix addition"}
    assert outputs["tool_result #2 run_tests"] == {
        "step": 2, "tool": "run_tests", "ok": True, "passed": False, "returncode": 1,
    }
    assert outputs["tool_result #3 edit_file"]["changed_file"] == "calculator.py"


def test_langfuse_failures_are_logged_and_do_not_change_the_repair(caplog: pytest.LogCaptureFixture) -> None:
    """A failing client changes neither the result nor the visible trace."""
    baseline = repair(None)
    with caplog.at_level(logging.WARNING, logger=lf.__name__):
        observed = repair(LangfuseTrajectoryRecorder(BrokenLangfuse()))

    assert observed.passed is baseline.passed is True
    assert observed.changed_files == baseline.changed_files
    assert [e["event"] for e in observed.trace] == [e["event"] for e in baseline.trace]
    messages = [r.getMessage() for r in caplog.records]
    assert any("Langfuse recording failed for repair_started" in m for m in messages)
    assert any("Langfuse flush failed" in m for m in messages)


def test_failure_mid_repair_still_flushes() -> None:
    """If a child observation fails, later events and the final flush still happen."""
    client = FakeLangfuse()
    original = FakeObservation.start_observation

    def flaky(self, *, name, **kwargs):
        if name == "tool_result #2 run_tests":
            raise RuntimeError("boom")
        return original(self, name=name, **kwargs)

    FakeObservation.start_observation = flaky
    try:
        result = repair(LangfuseTrajectoryRecorder(client))
    finally:
        FakeObservation.start_observation = original

    assert result.passed is True
    assert client.log[-1][0] == "flush"
    assert ("end", "repair", {}) in client.log


def test_observe_is_off_by_default() -> None:
    """build_service without observe never asks for a Langfuse client."""
    bootstrap.build_service("scripted")
    assert main.build_parser().parse_args(["repair", "evals/arithmetic"]).observe is False


@pytest.mark.parametrize("missing", list(lf.REQUIRED_ENV))
def test_observe_requires_all_three_variables(monkeypatch, capsys, missing: str) -> None:
    """--observe with a missing variable stops with a clear error before any repair."""
    for name in lf.REQUIRED_ENV:
        monkeypatch.setenv(name, "value")
    monkeypatch.delenv(missing)
    monkeypatch.setattr(bootstrap, "create_langfuse_client", lf.create_client)

    def no_repair(*args, **kwargs):
        raise AssertionError("repair started")

    monkeypatch.setattr(bootstrap.runner, "create_workspace", no_repair)
    with pytest.raises(ObservabilityConfigError, match=missing):
        lf.create_client()
    assert main.main(["repair", "evals/arithmetic", "--observe"]) == 2
    assert missing in capsys.readouterr().err
