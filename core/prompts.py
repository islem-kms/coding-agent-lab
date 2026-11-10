"""All text the repair agent sends to a model.

These helpers only build strings. Parsing and executing replies happens in the
repair loop (Task 4).
"""

from __future__ import annotations

from collections.abc import Iterable

SYSTEM_INSTRUCTION = """\
You are a careful Python repair agent working on a small local project.

Work in this order:
1. Inspect the local code and tests with read_file.
2. Reproduce a failing test with run_tests before making any edit.
3. Make one focused edit with edit_file that fixes the cause of the failure.
4. Verify the repair by running the tests again with run_tests.

Reply with exactly one JSON object and nothing else: no Markdown, no code
fences, no prose. Each reply is one action. The allowed actions are:
{"tool":"read_file","path":"test_calculator.py","summary":"Read the test"}
{"tool":"run_tests","summary":"Reproduce the failure"}
{"tool":"edit_file","path":"calculator.py","content":"..."}

Paths are relative to the project folder. edit_file replaces the whole file,
so "content" must be the complete new file text. The optional "summary" is a
short public label shown in the trace; do not put private reasoning in it.
"""


def system_instruction() -> str:
    """Return the system instruction describing the workflow and reply format."""
    return SYSTEM_INSTRUCTION


def repair_request(case: str, files: Iterable[str]) -> str:
    """Return the opening request naming the case and its relative file list."""
    listing = "\n".join(f"- {name}" for name in sorted(files)) or "- (no files)"
    return (
        f"Repair the project '{case}' so that its tests pass.\n"
        f"Files in the project:\n{listing}\n"
        "Reply with your first action as one JSON object."
    )


def tool_result_message(tool: str, observation: str) -> str:
    """Return the message that reports one tool's observation back to the model."""
    return (
        f"Result of {tool}:\n{observation}\n"
        "Reply with your next action as one JSON object."
    )
