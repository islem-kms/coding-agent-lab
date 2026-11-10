"""Gradio presentation layer for the repair service.

The page only collects a case choice and displays what ``RepairService``
returns. It makes no repair decisions, builds no prompts, and never calls
Docker itself.
"""

from __future__ import annotations

import difflib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gradio as gr

from adapters.sandbox.runner import WORKSPACE_ROOT

if TYPE_CHECKING:
    from core.repair_service import RepairService

CASES = ["arithmetic", "balanced_brackets", "median", "multi_function", "unique_words"]


def format_trace(trace: list[dict[str, Any]]) -> str:
    """Render the visible trace entries as readable text."""
    lines: list[str] = []
    for entry in trace:
        event = entry["event"]
        if event == "repair_started":
            lines.append(f"[repair_started] case={entry['case']} files={', '.join(entry['files'])}")
        elif event == "model_response":
            if not entry.get("valid", True):
                lines.append(f"[model_response #{entry['decision']}] INVALID: {entry['error']}")
                continue
            target = f" {entry['path']}" if "path" in entry else ""
            label = f" - {entry['summary']}" if entry.get("summary") else ""
            lines.append(f"[model_response #{entry['decision']}] {entry['tool']}{target}{label}")
        elif event == "tool_result":
            if "passed" in entry:
                status = "tests passed" if entry["passed"] else "tests failed"
            else:
                status = "ok" if entry.get("ok") else ("policy error" if entry.get("policy_error") else "error")
            lines.append(f"[tool_result] {entry['tool']}: {status}")
            lines.extend("    " + line for line in str(entry.get("observation", "")).splitlines())
        elif event == "repair_finished":
            outcome = "PASSED" if entry["passed"] else "FAILED"
            lines.append(f"[repair_finished] {outcome} after {entry['decisions']} decision(s): {entry['reason']}")
    return "\n".join(lines)


def format_changes(original_dir: Path, workspace: Path, changed: list[str]) -> str:
    """Show each changed relative file as a unified diff against the original."""
    if not changed:
        return "_No files were changed._"
    blocks: list[str] = []
    for relative in changed:
        before_path = original_dir / relative
        before = before_path.read_text(encoding="utf-8").splitlines() if before_path.is_file() else []
        after = (workspace / relative).read_text(encoding="utf-8").splitlines()
        diff = "\n".join(difflib.unified_diff(
            before, after, f"original/{relative}", f"repaired/{relative}", lineterm="",
        ))
        blocks.append(f"### `{relative}`\n```diff\n{diff or '(no textual difference)'}\n```")
    return "\n\n".join(blocks)


def zip_workspace(workspace: Path) -> str:
    """Archive the disposable workspace next to it and return the ZIP path."""
    return shutil.make_archive(str(workspace), "zip", root_dir=workspace)


def create_demo(service: RepairService | None = None) -> gr.Blocks:
    """Build the Gradio page; defaults to the scripted provider's service."""
    if service is None:
        import bootstrap  # Deferred: the assembly point wires core to adapters.

        service = bootstrap.build_service("scripted")

    def run_repair(case: str):
        """Repair the selected case via the service and return display values."""
        if case not in CASES:
            return "**Error:** choose one of the listed projects.", "", "", "", None
        try:
            result = service.repair(case)
        except Exception as exc:  # Show the problem instead of crashing the page.
            return f"**Error:** {type(exc).__name__}: {exc}", "", "", "", None

        outcome = "PASSED" if result.passed else "FAILED"
        reason = result.trace[-1].get("reason", "") if result.trace else ""
        final = result.last_test_result
        test_output = "(tests were not run)" if final is None else (
            f"{'PASSED' if final.passed else 'FAILED'} (returncode {final.returncode})\n\n{final.output}"
        )
        changes = format_changes(service.evals_dir / case, result.workspace, result.changed_files)
        archive = zip_workspace(result.workspace) if result.workspace else None
        status = f"**Repair {outcome}** for `{case}`: {reason}"
        return status, format_trace(result.trace), changes, test_output, archive

    with gr.Blocks(title="Python Repair Agent") as demo:
        gr.Markdown(
            "# Python Repair Agent\n"
            "Pick a supplied project. The agent repairs a disposable copy; "
            "tests run only inside the Docker sandbox."
        )
        with gr.Row():
            case = gr.Dropdown(choices=CASES, value=CASES[0], label="Project", allow_custom_value=False)
            start = gr.Button("Start repair", variant="primary")
        status = gr.Markdown()
        trace = gr.Textbox(label="Visible trace", lines=18, max_lines=40, interactive=False)
        changes = gr.Markdown(label="Changed files")
        tests = gr.Textbox(label="Final sandboxed test output", lines=6, interactive=False)
        archive = gr.File(label="Repaired workspace (ZIP)", interactive=False)

        start.click(run_repair, inputs=case, outputs=[status, trace, changes, tests, archive])
    return demo


def launch(service: RepairService | None = None, host: str = "127.0.0.1", port: int = 7860) -> None:
    """Start the local web server; only the workspace folder is downloadable."""
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    create_demo(service).launch(server_name=host, server_port=port, allowed_paths=[str(WORKSPACE_ROOT)])
