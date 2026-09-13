from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from superclaw.runtime import clip
from superclaw.settings import DUCKDUCKGO_LOCALE, DUCKDUCKGO_SEARCH_URL, ENGINE_GOOGLE, GOOGLE_CX_ENV, GOOGLE_KEY_ENV, GOOGLE_SEARCH_URL, LIMITS, Settings
from superclaw.tools import Permission, Result, Safety, SideEffect, Tool, ToolContext
from superclaw.tools.budget import Category

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://duckduckgo.com/",
    "Origin": "https://html.duckduckgo.com",
    "Content-Type": "application/x-www-form-urlencoded",
}
REFUSAL_MARKER = "anomaly"
AD_MARKER = "duckduckgo.com/y.js"
REDIRECT_PARAM = "uddg"
now = time.monotonic
pause = time.sleep
_last_request = 0.0


@dataclass(frozen=True)
class Hit:
    title: str
    url: str
    snippet: str


class Refused(RuntimeError):
    pass


def fetch(url: str, headers: dict[str, str], data: bytes | None = None) -> bytes:
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=LIMITS.web_search_timeout_s) as response:
        return response.read(LIMITS.web_search_body_bytes)


def google(query: str, limit: int, key: str, cx: str) -> list[Hit]:
    params = urllib.parse.urlencode({"key": key, "cx": cx, "q": query, "num": limit})
    data = json.loads(fetch(f"{GOOGLE_SEARCH_URL}?{params}", {"Accept": "application/json"}))
    return [Hit(str(item.get("title") or ""), str(item["link"]), str(item.get("snippet") or "")) for item in data.get("items", []) if item.get("link")]


def unwrap(url: str) -> str:
    parsed = urllib.parse.urlparse(url if "://" in url else f"https:{url}")
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        return urllib.parse.parse_qs(parsed.query).get(REDIRECT_PARAM, [url])[0]
    return url


class _DuckParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hits: list[Hit] = []
        self._url = ""
        self._field = ""
        self._title: list[str] = []
        self._snippet: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        classes = (dict(attrs).get("class") or "").split()
        if "result__a" in classes:
            self._url, self._field, self._title, self._snippet = unwrap(dict(attrs).get("href") or ""), "title", [], []
        elif "result__snippet" in classes:
            self._field = "snippet"

    def handle_data(self, data: str) -> None:
        if self._field == "title":
            self._title.append(data)
        elif self._field == "snippet":
            self._snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._field:
            return
        if self._field == "snippet" and self._url and AD_MARKER not in self._url:
            self.hits.append(Hit(" ".join("".join(self._title).split()), self._url, " ".join("".join(self._snippet).split())))
            self._url = ""
        self._field = ""


def throttle() -> None:
    global _last_request
    wait = LIMITS.web_search_min_interval_s - (now() - _last_request)
    if wait > 0:
        pause(wait)
    _last_request = now()


def duckduckgo(query: str, limit: int) -> list[Hit]:
    form = urllib.parse.urlencode({"q": query, "b": "", "kl": DUCKDUCKGO_LOCALE}).encode()
    for attempt in range(LIMITS.web_search_attempts):
        if attempt:
            pause(LIMITS.web_search_retry_s)
        throttle()
        page = fetch(DUCKDUCKGO_SEARCH_URL, BROWSER_HEADERS, form).decode("utf-8", "replace")
        parser = _DuckParser()
        parser.feed(page)
        if parser.hits or REFUSAL_MARKER not in page:
            return parser.hits[:limit]
    raise Refused(f"DuckDuckGo refused the query as automated traffic; retry later or set {GOOGLE_KEY_ENV} and {GOOGLE_CX_ENV} to search with Google")


def search(settings: Settings, query: str, limit: int) -> list[Hit]:
    if settings.search_engine == ENGINE_GOOGLE:
        if not (settings.google_search_key and settings.google_search_cx):
            raise Refused(f"Google search needs {GOOGLE_KEY_ENV} and {GOOGLE_CX_ENV}")
        return google(query, limit, settings.google_search_key, settings.google_search_cx)
    return duckduckgo(query, limit)


def host_of(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")


def allowed(hit: Hit, domains: set[str]) -> bool:
    host = host_of(hit.url)
    return any(host == d or host.endswith(f".{d}") for d in domains)


def format_hits(hits: list[Hit]) -> str:
    lines = []
    for i, hit in enumerate(hits, 1):
        lines.append(f"{i}. {hit.title or '(untitled)'}\n   {hit.url}")
        if hit.snippet:
            lines.append(f"   {clip(hit.snippet, LIMITS.web_search_snippet_chars)}")
    return "\n".join(lines)


class WebSearch(Tool):
    name = "web_search"
    deferred = True
    description = (
        "Search the web and get ranked results as title, URL and snippet. Google when GOOGLE_API_KEY and GOOGLE_CSE_ID are set, "
        "DuckDuckGo otherwise. Read a result's page with bash curl only if the user allows network; quote the URL you relied on."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "limit": {"type": "integer", "description": "Maximum results.", "default": LIMITS.web_search_results, "minimum": 1, "maximum": LIMITS.web_search_results_max},
            "domains": {"type": "array", "items": {"type": "string"}, "description": "Optional hostnames; results from other hosts are dropped before you see them."},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.NETWORK, Permission.PROMPT, "Sends the model's query text to a web search engine outside the sandbox.")
    output_category = Category.SEARCH

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        query = str(args.get("query") or "").strip()
        if not query:
            return Result.error("Error: query must not be empty")
        limit = max(1, min(int(args.get("limit") or LIMITS.web_search_results), LIMITS.web_search_results_max))
        domains = {str(d).strip().lower().removeprefix("*.") for d in (args.get("domains") or []) if str(d).strip()}
        try:
            hits = search(self._settings, query, limit)
        except Refused as e:
            return Result.error(f"Error: {e}")
        except (OSError, ValueError) as e:
            return Result.error(f"Error: web search failed: {type(e).__name__}: {e}")
        if domains:
            hits = [h for h in hits if allowed(h, domains)]
        if not hits:
            return Result.success(f"No results for: {query}")
        return Result.success(f"Results from {self._settings.search_engine} for: {query}\n{format_hits(hits)}")
