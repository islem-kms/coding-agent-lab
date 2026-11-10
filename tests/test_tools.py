"""Focused tests for the workspace-confined repair tools."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from adapters.sandbox import runner
from adapters.sandbox.repair_tools import RepairTools, UnsafePathError, safe_path
from core.models import TestResult


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A small workspace with one source file and a sibling secret outside it."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "calc.py").write_bytes(b"def add(a, b):\n    return a - b\n")
    (tmp_path / "secret.txt").write_text("do not read", encoding="utf-8")
    return ws


def test_reads_normal_workspace_file(workspace: Path) -> None:
    """A relative path inside the workspace returns the file's text."""
    tools = RepairTools(workspace, test_runner=lambda _: TestResult(True, "", 0))
    assert tools.read_file("calc.py") == "def add(a, b):\n    return a - b\n"


@pytest.mark.parametrize("bad", ["/etc/passwd", "C:\\Windows\\win.ini", "\\\\server\\share\\x"])
def test_rejects_absolute_paths(workspace: Path, bad: str) -> None:
    """Absolute paths in either POSIX or Windows form are refused."""
    with pytest.raises(UnsafePathError):
        safe_path(workspace, bad)


@pytest.mark.parametrize("bad", ["../secret.txt", "sub/../../secret.txt", ".."])
def test_rejects_parent_traversal(workspace: Path, bad: str) -> None:
    """Paths that resolve outside the workspace are refused for reads and edits."""
    tools = RepairTools(workspace, test_runner=lambda _: TestResult(True, "", 0))
    with pytest.raises(UnsafePathError):
        tools.read_file(bad)
    with pytest.raises(UnsafePathError):
        tools.edit_file(bad, "pwned")
    assert (workspace.parent / "secret.txt").read_text(encoding="utf-8") == "do not read"


def test_edit_is_saved_and_recorded_once(workspace: Path) -> None:
    """Edits are written to disk and each changed path is listed only once."""
    tools = RepairTools(workspace, test_runner=lambda _: TestResult(True, "", 0))
    fixed = "def add(a, b):\n    return a + b\n"

    tools.edit_file("calc.py", "draft\n")
    tools.edit_file("calc.py", fixed)

    assert (workspace / "calc.py").read_bytes() == fixed.encode("utf-8")
    assert tools.changed_files == ["calc.py"]


def test_run_tests_delegates_to_runner(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_tests hands the workspace to the runner and never launches a host process."""

    def forbidden(*args, **kwargs):
        raise AssertionError("host process launched")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)

    calls: list[Path] = []
    expected = TestResult(False, "1 failed", 1)

    def fake_runner(ws: Path) -> TestResult:
        calls.append(ws)
        return expected

    tools = RepairTools(workspace, test_runner=fake_runner)
    assert tools.run_tests() is expected
    assert calls == [workspace.resolve()]


def test_docker_command_is_isolated(workspace: Path) -> None:
    """The container has no network and mounts only the workspace, read-only."""
    cmd = runner.docker_command(workspace)
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert cmd[cmd.index("--network") + 1] == "none"
    mounts = [cmd[i + 1] for i, arg in enumerate(cmd) if arg in ("--mount", "-v", "--volume")]
    assert mounts == [f"type=bind,source={workspace.resolve()},target=/workspace,readonly"]
    assert cmd[-4:] == ["pytest", "-q", "-p", "no:cacheprovider"]
