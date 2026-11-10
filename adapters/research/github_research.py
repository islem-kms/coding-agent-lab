"""Bounded GitHub code search, implemented with the standard library only.

The adapter calls one endpoint, ``GET https://api.github.com/search/code``, and
returns a small evidence record: query, retrieval time, and up to three public
sources (repository, path, GitHub URL, short snippet). It never returns the
token, request headers, full remote files, or any other fetched page.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

SEARCH_URL = "https://api.github.com/search/code"
MAX_QUERY_CHARS = 180
MAX_REQUESTS = 3
MAX_RESULTS = 3
MAX_SNIPPET_CHARS = 500
TIMEOUT_SECONDS = 20


class ResearchError(RuntimeError):
    """Raised for a rejected query, an exhausted budget, or a failed search."""


class GitHubResearch:
    """``ResearchPort`` with per-repair request and result budgets."""

    def __init__(self, token: str, urlopen: Callable[..., Any] | None = None) -> None:
        if not token:
            raise ResearchError("A GitHub token is required for --research.")
        self._token = token
        self._urlopen = urlopen  # None means urllib.request.urlopen, looked up per call.
        self.requests_made = 0
        self.results_returned = 0

    @staticmethod
    def token_from_environment() -> str:
        """Return ``GITHUB_TOKEN`` from the environment; raise if it is not set."""
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if not token:
            raise ResearchError("Set GITHUB_TOKEN in this terminal session to use --research.")
        return token

    @staticmethod
    def validate_query(query: str) -> str:
        """Return a cleaned query or raise ResearchError if it is not allowed."""
        if not isinstance(query, str) or not query.strip():
            raise ResearchError("Search query must not be empty.")
        if "\n" in query or "\r" in query:
            raise ResearchError("Search query must be a single line.")
        if len(query) > MAX_QUERY_CHARS:
            raise ResearchError(f"Search query must be at most {MAX_QUERY_CHARS} characters.")
        return query.strip()

    def search_github(self, query: str) -> dict[str, Any]:
        """Run one budgeted code search and return a small evidence record."""
        query = self.validate_query(query)
        if self.requests_made >= MAX_REQUESTS:
            raise ResearchError(f"Research budget used: at most {MAX_REQUESTS} searches per repair.")
        remaining = MAX_RESULTS - self.results_returned
        if remaining <= 0:
            raise ResearchError(f"Research budget used: at most {MAX_RESULTS} results per repair.")

        self.requests_made += 1
        items = self._request(query, remaining)
        sources = [source for item in items if (source := self._source(item))][:remaining]
        self.results_returned += len(sources)
        return {
            "query": query,
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": sources,
        }

    def _request(self, query: str, per_page: int) -> list[dict[str, Any]]:
        """Perform the single GET request and return the raw result items."""
        url = SEARCH_URL + "?" + urllib.parse.urlencode({"q": query, "per_page": per_page})
        request = urllib.request.Request(url, method="GET", headers={
            "Accept": "application/vnd.github.text-match+json",
            "Authorization": f"Bearer {self._token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "coding-agent-lab",
        })
        try:
            opener = self._urlopen or urllib.request.urlopen
            with opener(request, timeout=TIMEOUT_SECONDS) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ResearchError(f"GitHub search failed with HTTP {exc.code}.") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            raise ResearchError("GitHub search could not be reached.") from None
        except json.JSONDecodeError:
            raise ResearchError("GitHub search returned a response that is not JSON.") from None
        items = data.get("items") if isinstance(data, dict) else None
        return items if isinstance(items, list) else []

    @staticmethod
    def _source(item: Any) -> dict[str, str] | None:
        """Reduce one result to public provenance and a short snippet, or drop it."""
        if not isinstance(item, dict):
            return None
        repo = item.get("repository") or {}
        if repo.get("private") is not False:  # Drop private or unmarked results.
            return None
        url = item.get("html_url", "")
        if not isinstance(url, str) or not url.startswith("https://github.com/"):
            return None
        fragments = [m.get("fragment", "") for m in item.get("text_matches") or [] if isinstance(m, dict)]
        return {
            "repository": str(repo.get("full_name", "")),
            "path": str(item.get("path", "")),
            "url": url,
            "snippet": "\n...\n".join(f for f in fragments if isinstance(f, str))[:MAX_SNIPPET_CHARS],
        }
