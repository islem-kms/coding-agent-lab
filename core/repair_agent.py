"""The bounded repair loop: the model proposes one action, Python decides.

``RepairAgent`` asks the chat client for one JSON action at a time, validates
it, enforces the evidence policy (read, then reproduce a failure, then edit,
then verify), runs at most one permitted tool, and reports the observation back
to the model. It depends only on ``core`` interfaces, never on adapters.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from core import prompts
from core.models import ModelAction, RepairResult, TestResult
from core.ports import ChatClient, RepairToolsPort, ResearchPort, TraceRecorder

DEFAULT_MAX_DECISIONS = 8
MAX_OBSERVATION_CHARS = 8000
MAX_SUMMARY_CHARS = 120

# Required and optional fields for each permitted tool (besides "tool").
_REQUIRED = {"read_file": {"path"}, "run_tests": set(), "edit_file": {"path", "content"}}
_RESEARCH_REQUIRED = {"search_github": {"query"}}  # Accepted only when research is enabled.
_OPTIONAL = {"summary"}


class InvalidReply(ValueError):
    """Raised when a model reply is not exactly one valid action object."""


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``json`` hook that refuses objects with repeated keys."""
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise InvalidReply(f"duplicate key {key!r}")
        seen[key] = value
    return seen


def parse_action(raw: str, allow_research: bool = False) -> ModelAction:
    """Parse ``raw`` as exactly one JSON action object or raise InvalidReply.

    ``search_github`` is a valid tool only when ``allow_research`` is true.
    """
    shapes = {**_REQUIRED, **_RESEARCH_REQUIRED} if allow_research else _REQUIRED
    try:
        data = json.loads(raw.strip(), object_pairs_hook=_reject_duplicates)
    except json.JSONDecodeError as exc:
        raise InvalidReply(f"reply is not a single JSON object ({exc.msg})") from None
    if not isinstance(data, dict):
        raise InvalidReply("reply must be a JSON object, not " + type(data).__name__)

    tool = data.get("tool")
    if not isinstance(tool, str) or tool not in shapes:
        raise InvalidReply(f"unknown or missing tool {tool!r}")

    fields = set(data) - {"tool"}
    missing = shapes[tool] - fields
    extra = fields - shapes[tool] - _OPTIONAL
    if missing:
        raise InvalidReply(f"{tool} is missing field(s): {', '.join(sorted(missing))}")
    if extra:
        raise InvalidReply(f"{tool} does not allow field(s): {', '.join(sorted(extra))}")

    if "path" in data and (not isinstance(data["path"], str) or not data["path"].strip()):
        raise InvalidReply("path must be a non-empty string")
    if "content" in data and not isinstance(data["content"], str):
        raise InvalidReply("content must be a string")
    if "query" in data and not isinstance(data["query"], str):
        raise InvalidReply("query must be a string")
    summary = data.get("summary")
    if summary is not None and not isinstance(summary, str):
        raise InvalidReply("summary must be a string")

    arguments = {k: v for k, v in data.items() if k in shapes[tool]}
    return ModelAction(tool=tool, arguments=arguments, summary=summary)


def _format_test_result(result: TestResult) -> str:
    """Render a test result as a compact observation for the model."""
    status = "PASSED" if result.passed else "FAILED"
    return f"Tests {status} (returncode {result.returncode}).\n{result.output}"


def _format_evidence(record: dict[str, Any]) -> str:
    """Render a research record with every source URL listed before snippets."""
    sources = record.get("sources") or []
    lines = [
        "GitHub research (optional evidence, not an authority; only a later passing test proves a repair).",
        f"Query: {record.get('query', '')}",
        f"Retrieved at: {record.get('retrieved_at', '')}",
        f"Sources ({len(sources)}):",
    ]
    lines += [f"[{i}] {s.get('repository', '')}/{s.get('path', '')} {s.get('url', '')}"
              for i, s in enumerate(sources, 1)]
    for i, source in enumerate(sources, 1):
        lines += [f"--- snippet [{i}] ---", str(source.get("snippet", ""))]
    return "\n".join(lines)


def _clip(text: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    """Shorten very long observations so the conversation stays bounded."""
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


class RepairAgent:
    """Traffic controller for one bounded, policy-checked repair attempt."""

    def __init__(
        self,
        client: ChatClient,
        max_decisions: int = DEFAULT_MAX_DECISIONS,
        recorder: TraceRecorder | None = None,
        research: ResearchPort | None = None,
    ) -> None:
        self.client = client
        self.max_decisions = max_decisions
        self.recorder = recorder
        self.research = research

    def run(
        self,
        case: str,
        files: Iterable[str],
        tools: RepairToolsPort,
        workspace: Path | None = None,
    ) -> RepairResult:
        """Run the repair loop for ``case`` and always return a RepairResult."""
        return _Attempt(self, tools).run(case, list(files), workspace)


class _Attempt:
    """State for a single repair attempt (conversation, evidence, trace)."""

    def __init__(self, agent: RepairAgent, tools: RepairToolsPort) -> None:
        self.agent = agent
        self.tools = tools
        self.trace: list[dict[str, Any]] = []
        self.messages: list[dict[str, str]] = []
        self.changed_files: list[str] = []
        self.last_test: TestResult | None = None
        self.has_read = False
        self.has_failing_test = False
        self.has_edit = False

    @property
    def research_enabled(self) -> bool:
        """Whether a research port was injected for this repair."""
        return self.agent.research is not None

    # -- trace -----------------------------------------------------------
    def emit(self, event: str, **payload: Any) -> None:
        """Append a visible trace entry and forward it to the optional recorder."""
        self.trace.append({"event": event, **payload})
        if self.agent.recorder is not None:
            try:
                self.agent.recorder.record(event, payload)
            except Exception:  # A recorder must never change the repair outcome.
                pass

    def finish(self, passed: bool, reason: str, decisions: int, workspace: Path | None) -> RepairResult:
        """Record ``repair_finished`` and build the final result."""
        self.emit("repair_finished", passed=passed, reason=reason, decisions=decisions)
        return RepairResult(
            passed=passed,
            trace=self.trace,
            changed_files=list(self.changed_files),
            workspace=workspace,
            last_test_result=self.last_test,
        )

    def observe(self, tool: str, observation: str, ok: bool, **extra: Any) -> None:
        """Trace a tool observation and show it to the model as a user message."""
        self.emit("tool_result", tool=tool, ok=ok, observation=_clip(observation, 2000), **extra)
        self.messages.append(
            {"role": "user", "content": prompts.tool_result_message(tool, _clip(observation))}
        )

    # -- loop ------------------------------------------------------------
    def run(self, case: str, files: list[str], workspace: Path | None) -> RepairResult:
        """Drive up to ``max_decisions`` model decisions."""
        self.messages = [
            {"role": "system", "content": prompts.system_instruction(research=self.research_enabled)},
            {"role": "user", "content": prompts.repair_request(case, files)},
        ]
        self.emit("repair_started", case=case, files=sorted(files), max_decisions=self.agent.max_decisions)

        for decision in range(1, self.agent.max_decisions + 1):
            try:
                raw = self.agent.client.chat(list(self.messages))
            except Exception as exc:
                return self.finish(False, f"Model client error: {type(exc).__name__}: {exc}", decision, workspace)

            try:
                action = parse_action(raw, allow_research=self.research_enabled)
            except InvalidReply as exc:
                self.emit("model_response", decision=decision, valid=False, error=str(exc))
                return self.finish(False, f"Invalid model reply: {exc}", decision, workspace)

            self.emit("model_response", decision=decision, valid=True, **self._public(action))
            self.messages.append({"role": "assistant", "content": raw})

            outcome = self.step(action)
            if outcome is not None:
                passed, reason = outcome
                return self.finish(passed, reason, decision, workspace)

        return self.finish(
            False, f"Decision budget of {self.agent.max_decisions} reached without a verified repair.",
            self.agent.max_decisions, workspace,
        )

    @staticmethod
    def _public(action: ModelAction) -> dict[str, Any]:
        """Return the trace-safe view of an action (no file content)."""
        view: dict[str, Any] = {"tool": action.tool}
        if "path" in action.arguments:
            view["path"] = action.arguments["path"]
        if "content" in action.arguments:
            view["content_chars"] = len(action.arguments["content"])
        if "query" in action.arguments:
            view["query"] = action.arguments["query"]
        if action.summary:
            view["summary"] = action.summary[:MAX_SUMMARY_CHARS]
        return view

    def step(self, action: ModelAction) -> tuple[bool, str] | None:
        """Run one accepted action; return (passed, reason) when the attempt ends."""
        tool, args = action.tool, action.arguments

        if tool in ("edit_file", "search_github") and not (self.has_read and self.has_failing_test):
            needed = [name for name, ok in (("read_file", self.has_read),
                                            ("a failing run_tests", self.has_failing_test)) if not ok]
            verb = "editing" if tool == "edit_file" else "searching"
            self.observe(tool, f"Policy error: {tool} rejected. Before {verb}, first use "
                         + " and ".join(needed) + ".", ok=False, policy_error=True)
            return None

        try:
            if tool == "search_github":
                record = self.agent.research.search_github(args["query"])
                urls = [s.get("url", "") for s in record.get("sources") or []]
                self.observe(tool, _format_evidence(record), ok=True, query=args["query"], sources=urls)
            elif tool == "read_file":
                text = self.tools.read_file(args["path"])
                self.has_read = True
                self.observe(tool, text, ok=True, path=args["path"])
            elif tool == "edit_file":
                note = self.tools.edit_file(args["path"], args["content"])
                self.has_edit = True
                relative = PurePosixPath(args["path"].replace("\\", "/")).as_posix()
                if relative not in self.changed_files:
                    self.changed_files.append(relative)
                self.observe(tool, note, ok=True, path=args["path"])
            else:
                result = self.tools.run_tests()
                self.last_test = result
                self.observe(tool, _format_test_result(result), ok=True,
                             passed=result.passed, returncode=result.returncode)
                return self._judge(result)
        except Exception as exc:
            self.observe(tool, f"Error: {type(exc).__name__}: {exc}", ok=False)
        return None

    def _judge(self, result: TestResult) -> tuple[bool, str] | None:
        """Decide whether a test result ends the attempt."""
        if not result.passed:
            self.has_failing_test = True
            return None
        if self.has_edit:
            return True, "Tests pass after an edit that followed a failing test."
        return False, "Tests passed before any edit: no reproducible failure to repair."
