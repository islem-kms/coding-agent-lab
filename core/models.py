"""Shared data containers passed between the repair agent, its model, and its tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    """Outcome of one test run: pass/fail flag, captured output, and exit code."""

    __test__ = False  # Stop pytest from collecting this class as a test case.

    passed: bool
    output: str
    returncode: int


@dataclass
class ModelAction:
    """One tool call requested by the model, with an optional public summary."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None


@dataclass
class RepairResult:
    """Final outcome of a repair attempt, including its trace and last test result."""

    passed: bool
    trace: list[dict[str, Any]] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    workspace: Path | None = None
    last_test_result: TestResult | None = None
