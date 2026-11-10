"""Create disposable workspaces and run their tests inside a Docker container.

The host only copies files and starts Docker. Evaluation code and pytest run
exclusively inside a short-lived container with networking disabled and the
workspace mounted read-only.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from core.models import TestResult

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS_DIR = REPO_ROOT / "evals"
IMAGE = "coding-agent-lab-runner"
CONTAINER_WORKSPACE = "/workspace"
DEFAULT_TIMEOUT_SECONDS = 120

# Kept under the home directory, never the system temp directory: Colima,
# Rancher Desktop and Podman share only $HOME with their virtual machine, so a
# mount of anything else would appear empty inside the container.
WORKSPACE_ROOT = Path.home() / ".cache" / "coding-agent-lab" / "workspaces"

_IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")


def list_cases() -> list[str]:
    """Return the names of the supplied evaluation projects."""
    return sorted(p.name for p in EVALS_DIR.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))


def create_workspace(case: str, root: Path = WORKSPACE_ROOT) -> Path:
    """Copy ``evals/<case>`` into a new disposable folder and return its path."""
    if case not in list_cases():
        raise ValueError(f"Unknown evaluation case: {case!r}")
    root.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=f"{case}-", dir=root))
    shutil.copytree(EVALS_DIR / case, workspace, dirs_exist_ok=True, ignore=_IGNORED)
    return workspace


def docker_command(workspace: Path, image: str = IMAGE) -> list[str]:
    """Build the ``docker run`` command for one isolated, read-only test run."""
    mount = f"type=bind,source={Path(workspace).resolve()},target={CONTAINER_WORKSPACE},readonly"
    return [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "128",
        "--memory", "512m",
        "--mount", mount,
        "--workdir", CONTAINER_WORKSPACE,
        image,
        "python", "-m", "pytest", "-q", "-p", "no:cacheprovider",
    ]


def run_tests(workspace: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> TestResult:
    """Run pytest for ``workspace`` in Docker and return a structured result.

    Docker failures (not installed, not running, timeout) are reported as a
    failing ``TestResult`` rather than raised, so the agent can observe them.
    """
    try:
        completed = subprocess.run(
            docker_command(workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return TestResult(False, "Runner error: the docker command was not found.", -1)
    except subprocess.TimeoutExpired:
        return TestResult(False, f"Runner error: tests exceeded {timeout} seconds.", -1)

    output = (completed.stdout or "") + (completed.stderr or "")
    return TestResult(completed.returncode == 0, output.strip(), completed.returncode)
