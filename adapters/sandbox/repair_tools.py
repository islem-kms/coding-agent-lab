"""The three workspace-confined tools the repair agent may use.

Only ``read_file``, ``edit_file`` and ``run_tests`` are exposed. There is no
shell, network, upload, or arbitrary-path capability, and tests run only
through the Docker-backed runner.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable

from adapters.sandbox import runner
from core.models import TestResult

TestRunner = Callable[[Path], TestResult]


class UnsafePathError(ValueError):
    """Raised when a requested path is absolute or escapes the workspace."""


def safe_path(workspace: Path, requested_path: str) -> Path:
    """Resolve ``requested_path`` inside ``workspace`` or raise UnsafePathError."""
    if not requested_path or not requested_path.strip():
        raise UnsafePathError("A relative file path is required.")
    # Check both flavours so "/etc/passwd" is rejected on Windows and "C:\\x" on POSIX.
    if PurePosixPath(requested_path).is_absolute() or PureWindowsPath(requested_path).anchor:
        raise UnsafePathError(f"Absolute paths are not allowed: {requested_path}")

    root = Path(workspace).resolve()
    candidate = (root / requested_path).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise UnsafePathError(f"Path escapes the workspace: {requested_path}")
    return candidate


class RepairTools:
    """Concrete ``RepairToolsPort`` bound to one disposable workspace."""

    def __init__(self, workspace: Path, test_runner: TestRunner = runner.run_tests) -> None:
        self.workspace = Path(workspace).resolve()
        self._test_runner = test_runner
        self.changed_files: list[str] = []

    def read_file(self, path: str) -> str:
        """Return the UTF-8 text of a file inside the workspace."""
        target = safe_path(self.workspace, path)
        if not target.is_file():
            raise FileNotFoundError(f"No such file in workspace: {path}")
        return target.read_text(encoding="utf-8", newline="")

    def edit_file(self, path: str, content: str) -> str:
        """Overwrite a workspace file with UTF-8 ``content`` and record the change."""
        target = safe_path(self.workspace, path)
        if not target.parent.is_dir():
            raise FileNotFoundError(f"No such directory in workspace: {path}")
        target.write_text(content, encoding="utf-8", newline="")
        relative = target.relative_to(self.workspace).as_posix()
        if relative not in self.changed_files:
            self.changed_files.append(relative)
        return f"Wrote {len(content)} characters to {relative}."

    def run_tests(self) -> TestResult:
        """Run the workspace tests through the Docker-backed runner."""
        return self._test_runner(self.workspace)
