from __future__ import annotations

import io
import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from http import HTTPStatus
from typing import Any

from superclaw.delegate import SPAWN_KEY
from superclaw.observations import ObservationStore
from superclaw.runtime import approx_tokens, clip
from superclaw.settings import BLOCKED_CODES, IMPERSONATE, LIMITS, READER_ENV, REDIRECT_CODES, Settings
from superclaw.tools import Permission, Result, Safety, SideEffect, Tool, ToolContext
from superclaw.tools.budget import Category

SCHEMES = ("http", "https")
FORMATS = ("auto", "raw", "markdown")
LOCAL_NAMES = ("localhost",)
DROP_TAGS = {"script", "style", "head", "noscript", "svg", "template", "iframe"}
HEADINGS = {f"h{level}": level for level in range(1, 7)}
BLOCK_TAGS = {"p", "div", "section", "article", "main", "header", "footer", "nav", "aside", "table", "tr", "ul", "ol",
              "blockquote", "hr", "dd", "dt", "figure", "figcaption", "form", "details", "summary"}
DEAD_LINKS = ("#", "javascript:", "mailto:")
HEADERS = {"User-Agent": "superclaw-web-fetch/1", "Accept": "text/*, application/json, application/xhtml+xml, application/xml;q=0.9, */*;q=0.5"}
HTML_STARTS = ("<!doctype html", "<html")
_SPACES = re.compile(r"\s+")
_BLANKS = re.compile(r"\n{3,}")


class Unsafe(ValueError):
    pass


@dataclass(frozen=True)
class Page:
    url: str
    status: int
    content_type: str
    body: bytes
    truncated: bool


def resolve(host: str) -> list[str]:
    return sorted({str(info[4][0]) for info in socket.getaddrinfo(host, None)})


def blocked(ip: str) -> str:
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv6Address) and (addr.ipv4_mapped or addr.sixtofour):
        addr = addr.ipv4_mapped or addr.sixtofour
    if addr.is_global:
        return ""
    if addr.is_loopback:
        return "loopback"
    if addr.is_link_local:
        return "link-local"
    if addr.is_multicast:
        return "multicast"
    return "private or special-use"


def validate(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in SCHEMES or not parsed.hostname:
        raise Unsafe(f"only public http and https URLs are fetched, not {url.strip()!r}")
    host = parsed.hostname.lower()
    if host in LOCAL_NAMES:
        raise Unsafe(f"{host} is not fetched; use bash with curl for local servers")
    try:
        addresses = [host] if _is_ip(host) else resolve(host)
    except OSError as e:
        raise Unsafe(f"{host} did not resolve: {e}") from e
    for address in addresses:
        if reason := blocked(address):
            where = f"{host} is" if address == host else f"{host} resolves to {address},"
            raise Unsafe(f"{where} a {reason} address; use bash with curl for local or private servers")
    return parsed


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


class _Redirects(urllib.request.HTTPRedirectHandler):
    max_redirections = LIMITS.web_fetch_redirects

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        validate(urllib.parse.urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_url(request: urllib.request.Request) -> Any:
    return urllib.request.build_opener(_Redirects()).open(request, timeout=LIMITS.web_fetch_timeout_s)


def browser_client() -> Any | None:
    try:
        import primp
    except ImportError:
        return None
    return primp.Client(impersonate=IMPERSONATE, follow_redirects=False, timeout=LIMITS.web_fetch_timeout_s)


def phrase(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "Error"


def fetch_as_browser(client: Any, url: str, max_bytes: int) -> Page:
    current = validate(url).geturl()
    for _ in range(LIMITS.web_fetch_redirects + 1):
        response = client.get(current)
        headers = {str(k).lower(): str(v) for k, v in dict(response.headers).items()}
        status = int(response.status_code)
        if status in REDIRECT_CODES and headers.get("location"):
            current = validate(urllib.parse.urljoin(current, headers["location"])).geturl()
            continue
        if status >= 400:
            raise urllib.error.HTTPError(current, status, phrase(status), headers, io.BytesIO(bytes(response.content)))
        raw = bytes(response.content)
        return Page(current, status, headers.get("content-type", ""), raw[:max_bytes], len(raw) > max_bytes)
    raise OSError(f"more than {LIMITS.web_fetch_redirects} redirects")


def fetch(url: str, max_bytes: int) -> Page:
    client = browser_client()
    if client is not None:
        return fetch_as_browser(client, url, max_bytes)
    parsed = validate(url)
    with open_url(urllib.request.Request(parsed.geturl(), headers=HEADERS)) as response:
        raw = response.read(max_bytes + 1)
        return Page(response.geturl(), int(response.status), str(response.headers.get("Content-Type") or ""), raw[:max_bytes], len(raw) > max_bytes)


def charset_of(content_type: str) -> str:
    match = re.search(r"charset=\"?([\w.-]+)", content_type, re.IGNORECASE)
    return match.group(1) if match else "utf-8"


def decode(page: Page) -> str:
    try:
        return page.body.decode(charset_of(page.content_type), errors="replace")
    except LookupError:
        return page.body.decode("utf-8", errors="replace")


def looks_like_html(content_type: str, text: str) -> bool:
    return "html" in content_type.lower() or text.lstrip()[: LIMITS.web_fetch_sniff_chars].lower().startswith(HTML_STARTS)


class _Markdown(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []
        self._dropping = 0
        self._pre = False
        self._href = ""
        self._label: list[str] | None = None

    def emit(self, text: str) -> None:
        (self._label if self._label is not None else self.out).append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in DROP_TAGS:
            self._dropping += 1
        if self._dropping:
            return
        if tag in HEADINGS:
            self.out.append(f"\n\n{'#' * HEADINGS[tag]} ")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag == "br":
            self.out.append("\n")
        elif tag == "pre":
            self._pre = True
            self.out.append("\n\n```\n")
        elif tag == "a":
            self._href, self._label = (dict(attrs).get("href") or "").strip(), []
        elif tag == "img":
            self.emit(dict(attrs).get("alt") or "")
        elif tag in BLOCK_TAGS:
            self.out.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in DROP_TAGS:
            self._dropping = max(0, self._dropping - 1)
            return
        if self._dropping:
            return
        if tag == "a" and self._label is not None:
            label = _SPACES.sub(" ", "".join(self._label)).strip()
            self.out.append(f"[{label}]({self._href})" if label and self._href and not self._href.lower().startswith(DEAD_LINKS) else label)
            self._href, self._label = "", None
        elif tag == "pre":
            self._pre = False
            self.out.append("\n```\n")
        elif tag in HEADINGS or tag in BLOCK_TAGS:
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._dropping:
            self.emit(data if self._pre else _SPACES.sub(" ", data))

    def text(self) -> str:
        lines, fenced = [], False
        for line in "".join(self.out).splitlines():
            if line.startswith("```"):
                fenced = not fenced
            lines.append(line.rstrip() if fenced else line.strip())
        return _BLANKS.sub("\n\n", "\n".join(lines)).strip()


def to_markdown(html: str) -> str:
    parser = _Markdown()
    parser.feed(html)
    parser.close()
    return parser.text()


def render(page: Page, format: str) -> tuple[list[str], str]:
    text = decode(page)
    converted = format == "markdown" or (format == "auto" and looks_like_html(page.content_type, text))
    body = to_markdown(text) if converted else text
    head = [f"URL: {page.url}", f"Status: {page.status}", f"Content-Type: {page.content_type or 'unknown'}",
            f"Bytes: {len(page.body):,}" + (", truncated at max_bytes; raise it for the rest" if page.truncated else "")]
    if converted:
        head.append('Converted: html to markdown (format "raw" keeps the html)')
    return head, body


def digest(body: str) -> list[str]:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    headings = [line for line in lines if line.startswith("#")]
    title = (headings[0] if headings else lines[0] if lines else "").lstrip("# ").strip()
    out = [f"Title: {title or '(none)'}", f"Size: {approx_tokens(body):,} tokens, {body.count('](')} links"]
    if headings:
        out += ["Outline:", *headings[: LIMITS.web_fetch_outline_lines]]
    return out


class WebFetch(Tool):
    name = "web_fetch"
    deferred = True
    description = (
        "Fetch a public http or https URL. The page is stored whole as a §ref (HTML as compact markdown) and you get its digest: title, size, outline. "
        "Pass prompt to have a child agent read the whole page in a fresh context and answer it; that is the way to extract facts without paying for the page. "
        "inline=true returns the text itself, budgeted. Loopback, private and link-local hosts are refused, use bash with curl for those."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Public http or https URL."},
            "prompt": {"type": "string", "description": "Question or extraction task a child agent answers from the whole page."},
            "inline": {"type": "boolean", "description": "Return the page text in this context instead of a digest.", "default": False},
            "max_bytes": {"type": "integer", "description": "Raw bytes to download before conversion.", "default": LIMITS.web_fetch_bytes, "minimum": 1, "maximum": LIMITS.web_fetch_bytes_max},
            "format": {"type": "string", "enum": list(FORMATS), "description": "auto converts HTML to markdown, raw never converts, markdown always does.", "default": "auto"},
        },
        "required": ["url"],
        "additionalProperties": False,
    }
    safety = Safety(SideEffect.NETWORK, Permission.PROMPT, "Requests a model-chosen URL from this host and stores the page for a week.")
    output_category = Category.DEFAULT

    def __init__(self, observations: ObservationStore | None = None, settings: Settings | None = None) -> None:
        self._store = observations
        self._reader = settings.reader_url if settings else ""

    def page(self, url: str, max_bytes: int) -> tuple[Page, str]:
        try:
            return fetch(url, max_bytes), ""
        except urllib.error.HTTPError as blocked:
            if not self._reader or blocked.code not in BLOCKED_CODES:
                raise
            try:
                return fetch(self._reader + url, max_bytes), self._reader
            except (Unsafe, urllib.error.HTTPError, OSError, ValueError):
                raise blocked from None

    def run(self, args: dict[str, Any], ctx: ToolContext) -> Result:
        url = str(args.get("url") or "").strip()
        format = str(args.get("format") or "auto")
        if not url:
            return Result.error("Error: url must not be empty")
        if format not in FORMATS:
            return Result.error(f"Error: format must be one of {', '.join(FORMATS)}")
        max_bytes = max(1, min(int(args.get("max_bytes") or LIMITS.web_fetch_bytes), LIMITS.web_fetch_bytes_max))
        try:
            page, via = self.page(url, max_bytes)
        except Unsafe as e:
            return Result.error(f"Error: {e}")
        except urllib.error.HTTPError as e:
            detail = clip(e.read(LIMITS.web_fetch_bytes).decode("utf-8", errors="replace").strip(), LIMITS.preview_error_chars)
            remedy = f"; a reader proxy in {READER_ENV} (for example https://r.jina.ai/) retries blocked pages" if e.code in BLOCKED_CODES and not self._reader else ""
            return Result.error(f"Error fetching URL: HTTP {e.code} {e.reason}{remedy}" + (f"\n{detail}" if detail else ""))
        except (OSError, ValueError) as e:
            return Result.error(f"Error fetching URL: {type(e).__name__}: {e}")
        head, body = render(page, format)
        if via:
            head.append(f"Via: {via}")
        stored = "\n".join([*head, "", body])
        if args.get("inline") or self._store is None:
            return Result.success(stored, truncated=page.truncated, meta={"full": stored})
        ref = self._store.save(ctx.session_id, self.name, "", stored, expires_days=LIMITS.web_raw_ttl_days)
        head.append(f"Stored: §{ref} for {LIMITS.web_raw_ttl_days} days; recall §{ref} reads it in chunks")
        prompt = str(args.get("prompt") or "").strip()
        spawn = ctx.state.get(SPAWN_KEY)
        if prompt and spawn is not None:
            child = spawn({"task": f"{prompt}\n\nThe page is stored as §{ref}; read all of it with recall before answering, and quote what you rely on.", "refs": [ref]})
            return Result(child.ok, "\n".join([*head, "", child.output]), truncated=page.truncated)
        if prompt:
            head.append("No child agent is available in this run; recall the §ref yourself")
        return Result.success("\n".join([*head, "", *digest(body)]), truncated=page.truncated)
