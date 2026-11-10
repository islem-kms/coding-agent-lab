"""Tests for the optional GitHub research extension. Nothing here contacts GitHub."""

from __future__ import annotations

import io
import json
import urllib.request

import pytest

import bootstrap
from adapters.models.scripted import ScriptedClient
from adapters.research.github_research import (
    MAX_SNIPPET_CHARS, GitHubResearch, ResearchError,
)
from core import prompts
from core.models import TestResult
from core.repair_agent import InvalidReply, RepairAgent, parse_action

TOKEN = "test-token-not-real"
FIXED = "def add(a, b):\n    return a + b\n"
FAIL = TestResult(False, "1 failed", 1)
PASS = TestResult(True, "1 passed", 0)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any code path tries the real urllib opener."""
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def item(n: int, private: bool = False, fragment: str = "return sum(values) / 2") -> dict:
    """One fake GitHub code-search result item."""
    return {
        "path": f"src/stats{n}.py",
        "html_url": f"https://github.com/octo/repo{n}/blob/main/src/stats{n}.py",
        "repository": {"full_name": f"octo/repo{n}", "private": private},
        "text_matches": [{"fragment": fragment}],
    }


class FakeHTTP:
    """Stand-in for urlopen that returns canned JSON and records requests."""

    def __init__(self, items: list[dict]) -> None:
        self.items = items
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        return io.BytesIO(json.dumps({"items": self.items}).encode("utf-8"))


class FakeTools:
    """In-memory sandbox tools with queued test results."""

    def __init__(self, results: list[TestResult]) -> None:
        self.results = list(results)
        self.files = {"calculator.py": "def add(a, b):\n    return a - b\n"}

    def read_file(self, path: str) -> str:
        return self.files[path]

    def edit_file(self, path: str, content: str) -> str:
        self.files[path] = content
        return "ok"

    def run_tests(self) -> TestResult:
        return self.results.pop(0)


def card(tool: str, **fields: str) -> str:
    return json.dumps({"tool": tool, **fields})


READ = card("read_file", path="calculator.py")
TEST = card("run_tests")
SEARCH = card("search_github", query="Python add two numbers", summary="Find a public reference")
EDIT = card("edit_file", path="calculator.py", content=FIXED)


# -- disabled research ---------------------------------------------------

def test_disabled_research_rejects_search_and_makes_no_call() -> None:
    """Without a research port, search_github is an invalid reply and nothing is called."""
    with pytest.raises(InvalidReply):
        parse_action(SEARCH)
    result = RepairAgent(ScriptedClient([READ, TEST, SEARCH])).run(
        "arithmetic", ["calculator.py"], FakeTools([FAIL]))
    assert result.passed is False
    assert "Invalid model reply" in result.trace[-1]["reason"]
    assert "search_github" not in prompts.system_instruction()


def test_bootstrap_does_not_construct_research_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_service without research never reads the token or builds the adapter."""
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)

    def forbidden(*args, **kwargs):
        raise AssertionError("research adapter constructed")

    monkeypatch.setattr(bootstrap.GitHubResearch, "__init__", forbidden)
    monkeypatch.setattr(bootstrap.GitHubResearch, "token_from_environment", forbidden)
    service = bootstrap.build_service("scripted")
    assert service._make_agent("arithmetic").research is None


def test_research_flag_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """--research without GITHUB_TOKEN fails at composition time."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(ResearchError):
        bootstrap.build_service("scripted", research=True)


# -- evidence gate --------------------------------------------------------

@pytest.mark.parametrize("replies,results", [
    ([SEARCH], []),                 # nothing yet
    ([READ, SEARCH], []),           # read but no failing test
    ([TEST, SEARCH], [FAIL]),       # failing test but no read
])
def test_early_search_is_a_visible_policy_error_without_api_call(replies, results) -> None:
    """Search before local read + failing test is rejected and calls no API."""
    http = FakeHTTP([item(1)])
    agent = RepairAgent(ScriptedClient(replies), research=GitHubResearch(TOKEN, urlopen=http))
    result = agent.run("arithmetic", ["calculator.py"], FakeTools(results))

    rejected = [e for e in result.trace if e["event"] == "tool_result" and e["tool"] == "search_github"]
    assert rejected and rejected[0]["policy_error"] is True
    assert "Policy error" in rejected[0]["observation"]
    assert http.requests == []


# -- adapter budgets and record shape -------------------------------------

@pytest.mark.parametrize("query", ["", "   ", "two\nlines", "x" * 181])
def test_bad_queries_are_rejected_without_request(query: str) -> None:
    """Empty, multi-line and over-180-character queries never reach the API."""
    http = FakeHTTP([item(1)])
    with pytest.raises(ResearchError):
        GitHubResearch(TOKEN, urlopen=http).search_github(query)
    assert http.requests == []


def test_request_budget_is_three_per_repair() -> None:
    """A fourth search raises without making a request."""
    http = FakeHTTP([])
    research = GitHubResearch(TOKEN, urlopen=http)
    for _ in range(3):
        research.search_github("median even length")
    with pytest.raises(ResearchError, match="at most 3 searches"):
        research.search_github("median even length")
    assert len(http.requests) == 3


def test_result_budget_is_three_per_repair_and_private_results_are_dropped() -> None:
    """Private results are discarded and no more than three sources are returned overall."""
    http = FakeHTTP([item(0, private=True)] + [item(n) for n in range(1, 6)])
    research = GitHubResearch(TOKEN, urlopen=http)

    record = research.search_github("median")
    assert [s["repository"] for s in record["sources"]] == ["octo/repo1", "octo/repo2", "octo/repo3"]
    with pytest.raises(ResearchError, match="at most 3 results"):
        research.search_github("median again")
    assert len(http.requests) == 1


def test_record_is_small_and_never_leaks_the_token() -> None:
    """The record has only query, time and trimmed sources; the token is absent."""
    http = FakeHTTP([item(1, fragment="x" * 2000)])
    record = GitHubResearch(TOKEN, urlopen=http).search_github("median")

    assert set(record) == {"query", "retrieved_at", "sources"}
    assert set(record["sources"][0]) == {"repository", "path", "url", "snippet"}
    assert len(record["sources"][0]["snippet"]) <= MAX_SNIPPET_CHARS
    assert TOKEN not in json.dumps(record)
    sent = http.requests[0]
    assert sent.get_method() == "GET"
    assert sent.full_url.startswith("https://api.github.com/search/code?")


# -- allowed search -------------------------------------------------------

def test_allowed_search_shows_source_url_and_only_tests_finish_the_repair() -> None:
    """After read + failing test, the evidence keeps its GitHub URL; a later pass finishes."""
    http = FakeHTTP([item(1)])
    client = ScriptedClient([READ, TEST, SEARCH, EDIT, TEST])
    agent = RepairAgent(client, research=GitHubResearch(TOKEN, urlopen=http))
    result = agent.run("arithmetic", ["calculator.py"], FakeTools([FAIL, PASS]))

    url = "https://github.com/octo/repo1/blob/main/src/stats1.py"
    observed = [e for e in result.trace if e["event"] == "tool_result" and e["tool"] == "search_github"][0]
    assert observed["ok"] is True and observed["sources"] == [url]
    assert url in observed["observation"]
    assert url in client.calls[3][-1]["content"]
    assert "not an authority" in client.calls[0][0]["content"]
    assert len(http.requests) == 1
    assert result.passed is True
    assert result.trace[-1]["decisions"] == 5
