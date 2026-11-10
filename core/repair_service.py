"""Use case: repair one supplied evaluation project in a disposable workspace.

The service validates the selection, asks an injected factory for a workspace
copy and tools, and runs the repair agent. It knows nothing about the command
line, Docker, or a particular model provider.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath, PureWindowsPath

from core.models import RepairResult
from core.ports import RepairToolsPort
from core.repair_agent import RepairAgent

_SKIPPED_PARTS = {"__pycache__", ".pytest_cache"}


class InvalidSelection(ValueError):
    """Raised when the selected project is absolute, outside evals/, or unknown."""


class RepairService:
    """Validate a selected eval project, copy it, and run one repair attempt."""

    def __init__(
        self,
        evals_dir: Path,
        create_workspace: Callable[[str], Path],
        make_tools: Callable[[Path], RepairToolsPort],
        make_agent: Callable[[str], RepairAgent],
    ) -> None:
        self.evals_dir = Path(evals_dir).resolve()
        self._create_workspace = create_workspace
        self._make_tools = make_tools
        self._make_agent = make_agent

    def cases(self) -> list[str]:
        """Return the names of the supplied evaluation projects."""
        return sorted(
            p.name for p in self.evals_dir.iterdir()
            if p.is_dir() and not p.name.startswith((".", "_"))
        )

    def resolve_case(self, selected: str) -> str:
        """Map ``evals/<case>`` (or a bare ``<case>``) to a known case name."""
        text = (selected or "").strip()
        if not text:
            raise InvalidSelection("Select a project under evals/.")
        if PurePosixPath(text).is_absolute() or PureWindowsPath(text).anchor:
            raise InvalidSelection(f"Absolute paths are not allowed: {selected}")

        parts = PurePosixPath(text.replace("\\", "/")).parts
        relative = Path(*parts) if parts[0] == self.evals_dir.name else Path(self.evals_dir.name, *parts)
        candidate = (self.evals_dir.parent / relative).resolve()
        if candidate.parent != self.evals_dir:
            raise InvalidSelection(f"Select a project directly under evals/: {selected}")
        if candidate.name not in self.cases():
            raise InvalidSelection(f"Unknown project: {selected}. Choose one of: {', '.join(self.cases())}")
        return candidate.name

    @staticmethod
    def list_files(workspace: Path) -> list[str]:
        """Return the workspace's files as sorted relative POSIX paths."""
        return sorted(
            path.relative_to(workspace).as_posix()
            for path in workspace.rglob("*")
            if path.is_file() and not _SKIPPED_PARTS.intersection(path.relative_to(workspace).parts)
        )

    def repair(self, selected: str) -> RepairResult:
        """Copy the selected project to a fresh workspace and run the repair agent."""
        case = self.resolve_case(selected)
        workspace = self._create_workspace(case)
        tools = self._make_tools(workspace)
        agent = self._make_agent(case)
        return agent.run(case, self.list_files(workspace), tools, workspace=workspace)
