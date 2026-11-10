"""Command-line entry point: ``python main.py repair evals/<case> --provider scripted``."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import bootstrap
from core.repair_service import InvalidSelection

MAX_PRINTED_LINES = 12


def _indent(text: str, limit: int = MAX_PRINTED_LINES) -> str:
    """Indent ``text`` and keep only its first ``limit`` lines."""
    lines = str(text).splitlines() or [""]
    shown = lines[:limit]
    if len(lines) > limit:
        shown.append(f"... ({len(lines) - limit} more lines)")
    return "\n".join("      " + line for line in shown)


def print_event(event: str, payload: dict[str, Any]) -> None:
    """Print one visible trace event as soon as the agent emits it."""
    if event == "repair_started":
        print(f"[repair_started] case={payload['case']} files={', '.join(payload['files'])}")
    elif event == "model_response":
        if not payload.get("valid", True):
            print(f"[model_response #{payload['decision']}] INVALID: {payload['error']}")
            return
        target = f" {payload['path']}" if "path" in payload else ""
        label = f" - {payload['summary']}" if payload.get("summary") else ""
        print(f"[model_response #{payload['decision']}] {payload['tool']}{target}{label}")
    elif event == "tool_result":
        status = "ok" if payload.get("ok") else ("policy error" if payload.get("policy_error") else "error")
        if "passed" in payload:
            status = "tests passed" if payload["passed"] else "tests failed"
        print(f"[tool_result] {payload['tool']}: {status}")
        print(_indent(payload.get("observation", "")))
    elif event == "repair_finished":
        outcome = "PASSED" if payload["passed"] else "FAILED"
        print(f"[repair_finished] {outcome} after {payload['decisions']} decision(s): {payload['reason']}")
    sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the ``repair`` command."""
    parser = argparse.ArgumentParser(prog="main.py", description="Sandboxed Python repair agent.")
    commands = parser.add_subparsers(dest="command", required=True)
    repair = commands.add_parser("repair", help="Repair one supplied project under evals/.")
    repair.add_argument("project", help="Selected project, for example evals/arithmetic.")
    repair.add_argument("--provider", choices=bootstrap.PROVIDERS, default="scripted")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit status."""
    args = build_parser().parse_args(argv)
    service = bootstrap.build_service(args.provider, on_event=print_event)
    try:
        result = service.repair(args.project)
    except InvalidSelection as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"Workspace: {result.workspace}")
    print("Changed files: " + (", ".join(result.changed_files) or "(none)"))
    final = result.last_test_result
    if final is None:
        print("Final sandboxed test result: (tests were not run)")
    else:
        status = "PASSED" if final.passed else "FAILED"
        print(f"Final sandboxed test result: {status} (returncode {final.returncode})")
        print(_indent(final.output))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
