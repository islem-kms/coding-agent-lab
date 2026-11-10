"""Best-effort Langfuse Cloud recorder for repair trajectories.

Implements the ``TraceRecorder`` port. One repair becomes one root ``agent``
observation with closed child observations for each model decision and tool
result. Only a small public payload is sent: case name, step number, tool
name, public summary, pass/fail, return code, changed file names, and outcome.
Prompts, model text, file contents, and test output are never sent. Every
Langfuse failure is caught and logged; it can never change or stop a repair.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

REQUIRED_ENV = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
MAX_SUMMARY_CHARS = 120


class ObservabilityConfigError(RuntimeError):
    """Raised when --observe is requested without the Langfuse settings."""


def missing_environment() -> list[str]:
    """Return the names of required Langfuse variables that are not set."""
    return [name for name in REQUIRED_ENV if not os.environ.get(name, "").strip()]


def create_client() -> Any:
    """Build a Langfuse client from the environment, or raise a clear error."""
    missing = missing_environment()
    if missing:
        raise ObservabilityConfigError(
            "--observe needs these environment variables: " + ", ".join(missing)
        )
    from langfuse import Langfuse  # Imported only when observation is requested.

    return Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ["LANGFUSE_BASE_URL"],
    )


def _compact(data: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is None."""
    return {key: value for key, value in data.items() if value is not None}


class LangfuseTrajectoryRecorder:
    """``TraceRecorder`` that mirrors repair events into Langfuse observations."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._root: Any = None
        self._step = 0
        self._tool = ""

    def record(self, event: str, payload: dict[str, Any]) -> None:
        """Record one event; never raises."""
        try:
            handler = getattr(self, f"_on_{event}", None)
            if handler is not None:
                handler(payload)
        except Exception:
            logger.warning("Langfuse recording failed for %s; the repair continues.", event, exc_info=True)
        if event == "repair_finished":
            self._flush()

    # -- event handlers (each receives the agent's payload) -------------------
    def _on_repair_started(self, payload: dict[str, Any]) -> None:
        self._step, self._tool = 0, ""
        self._root = self._client.start_observation(
            name="repair", as_type="agent", input={"case": str(payload.get("case", ""))},
        )

    def _on_model_response(self, payload: dict[str, Any]) -> None:
        self._step = int(payload.get("decision", self._step + 1))
        self._tool = str(payload.get("tool", "")) if payload.get("valid", True) else ""
        summary = payload.get("summary")
        output = _compact({
            "step": self._step,
            "valid": bool(payload.get("valid", True)),
            "tool": self._tool or None,
            "summary": str(summary)[:MAX_SUMMARY_CHARS] if summary else None,
        })
        self._child(f"model_response #{self._step}", "span", output)

    def _on_tool_result(self, payload: dict[str, Any]) -> None:
        tool = str(payload.get("tool", self._tool))
        output = _compact({
            "step": self._step,
            "tool": tool,
            "ok": bool(payload.get("ok", False)),
            "policy_error": True if payload.get("policy_error") else None,
            "passed": payload.get("passed"),
            "returncode": payload.get("returncode"),
            "changed_file": payload.get("path") if tool == "edit_file" and payload.get("ok") else None,
        })
        self._child(f"tool_result #{self._step} {tool}", "tool", output)

    def _on_repair_finished(self, payload: dict[str, Any]) -> None:
        if self._root is None:
            return
        passed = bool(payload.get("passed", False))
        self._root.update(output={
            "outcome": "repaired" if passed else "not_repaired",
            "passed": passed,
            "decisions": payload.get("decisions"),
            "changed_files": [str(name) for name in payload.get("changed_files", [])],
        })
        root, self._root = self._root, None
        root.end()

    # -- helpers -------------------------------------------------------------
    def _child(self, name: str, as_type: str, output: dict[str, Any]) -> None:
        """Create and immediately close one child observation under the root."""
        if self._root is None:
            return
        self._root.start_observation(name=name, as_type=as_type, output=output).end()

    def _flush(self) -> None:
        """Flush pending data to Langfuse; log instead of raising on failure."""
        try:
            self._client.flush()
        except Exception:
            logger.warning("Langfuse flush failed; the repair result is unaffected.", exc_info=True)
