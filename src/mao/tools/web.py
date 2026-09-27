"""Web research tools: search (several backends) and URL fetching with SSRF protection."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
from collections.abc import Awaitable, Callable
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx
from pydantic import BaseModel

from mao.config.schema import WebToolConfig
from mao.core.errors import PermissionDeniedError, ToolError
from mao.core.text import truncate_end
from mao.security.permissions import Capability
from mao.security.risk import ActionKind, ProposedAction, RiskLevel
from mao.tools.base import Tool, ToolContext, ToolResult, object_schema

GroundedSearch = Callable[[str], Awaitable[tuple[str, list[dict[str, str]]]]]


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str = ""


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hits: list[SearchHit] = []
        self._capture: str | None = None
        self._buffer: list[str] = []
        self._href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            self._capture, self._buffer, self._href = "title", [], attributes.get("href") or ""
        elif tag in ("a", "div", "td") and "result__snippet" in classes:
            self._capture, self._buffer = "snippet", []

    def handle_endtag(self, tag: str) -> None:
        if self._capture == "title" and tag == "a":
            self.hits.append(SearchHit(title=" ".join("".join(self._buffer).split()), url=_ddg_target(self._href)))
            self._capture = None
        elif self._capture == "snippet" and tag in ("a", "div", "td"):
            if self.hits:
                self.hits[-1].snippet = " ".join("".join(self._buffer).split())
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._buffer.append(data)


def _ddg_target(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return href


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "template", "iframe", "canvas"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "section", "article", "header", "footer", "table", "ul", "ol"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(line for line in lines)).strip()


class WebSearcher:
    def __init__(self, config: WebToolConfig, http: httpx.AsyncClient, grounded_search: GroundedSearch | None = None) -> None:
        self.config = config
        self.http = http
        self.grounded_search = grounded_search

    def backend(self) -> str:
        if self.config.search_backend != "auto":
            return self.config.search_backend
        if os.environ.get(self.config.brave_api_key_env):
            return "brave"
        if os.environ.get(self.config.tavily_api_key_env):
            return "tavily"
        if self.config.searxng_url:
            return "searxng"
        return "duckduckgo"

    async def search(self, query: str, max_results: int) -> tuple[list[SearchHit], str, str | None]:
        backend = self.backend()
        headers = {"User-Agent": self.config.user_agent}
        timeout = self.config.timeout_s
        answer: str | None = None
        try:
            if backend == "brave":
                response = await self.http.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": max_results},
                    headers={**headers, "Accept": "application/json", "X-Subscription-Token": os.environ.get(self.config.brave_api_key_env, "")},
                    timeout=timeout,
                )
                response.raise_for_status()
                results = (response.json().get("web") or {}).get("results") or []
                hits = [SearchHit(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("description", "")) for r in results]
            elif backend == "tavily":
                response = await self.http.post(
                    "https://api.tavily.com/search",
                    json={"query": query, "max_results": max_results},
                    headers={**headers, "Authorization": f"Bearer {os.environ.get(self.config.tavily_api_key_env, '')}"},
                    timeout=timeout,
                )
                response.raise_for_status()
                hits = [SearchHit(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("content", "")) for r in response.json().get("results") or []]
            elif backend == "searxng":
                if not self.config.searxng_url:
                    raise ToolError("searxng_url is not configured")
                response = await self.http.get(
                    self.config.searxng_url.rstrip("/") + "/search",
                    params={"q": query, "format": "json"},
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                hits = [SearchHit(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("content", "")) for r in response.json().get("results") or []]
            elif backend == "gemini":
                if self.grounded_search is None:
                    raise ToolError("Gemini search unavailable (no Gemini key or model configured)")
                answer, sources = await self.grounded_search(query)
                hits = [SearchHit(title=s.get("title", ""), url=s.get("url", "")) for s in sources]
            else:
                response = await self.http.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query},
                    headers=headers,
                    timeout=timeout,
                    follow_redirects=True,
                )
                response.raise_for_status()
                parser = _DuckDuckGoParser()
                parser.feed(response.text)
                hits = parser.hits
                if not hits and re.search(r"anomaly|captcha|challenge", response.text, re.I):
                    raise ToolError(
                        "DuckDuckGo refused the automated request. For reliable research, please configure "
                        "BRAVE_API_KEY, TAVILY_API_KEY or searxng_url."
                    )
        except httpx.HTTPError as exc:
            raise ToolError(f"Web search ({backend}) failed: {type(exc).__name__}: {exc}") from exc
        hits = [h for h in hits if h.url.startswith(("http://", "https://"))][:max_results]
        return hits, backend, answer


async def ensure_public_url(url: str, allow_private: bool) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise PermissionDeniedError(f"Only http(s) URLs are allowed: {url}")
    if allow_private:
        return
    host = parsed.hostname
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except OSError as exc:
        raise ToolError(f"Host cannot be resolved: {host} ({exc})") from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0].split("%")[0])
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved or address.is_multicast or address.is_unspecified:
            raise PermissionDeniedError(f"Access to local or private network addresses blocked: {host} ({address})")


class WebSearchTool(Tool):
    name = "web_search"
    group = "web"
    description = "Search the internet. Returns titles, URLs and snippets. Use fetch_url to read a result."
    parameters = object_schema(
        {"query": {"type": "string"}, "max_results": {"type": "integer", "description": "1-10"}},
        ["query"],
    )
    required = frozenset({Capability.INTERNET})

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.web is None:
            return ToolResult.failure("Web search is not available")
        query = str(args["query"]).strip()
        if not query:
            return ToolResult.failure("Empty search query")
        await ctx.approval.require(ProposedAction(kind=ActionKind.INTERNET, agent=ctx.agent_name, target=f"search: {query}", risk=RiskLevel.LOW))
        limit = max(1, min(int(args.get("max_results") or ctx.tools_config.web.max_results), 10))
        hits, backend, answer = await ctx.web.search(query, limit)
        if ctx.collaboration is not None and hits:
            ctx.collaboration.record_sources(ctx.agent_name, query, [h.model_dump() for h in hits], ctx.node_id)
        lines = [f"Web search ({backend}): {query}"]
        if answer:
            lines.append("Summary:\n" + truncate_end(answer, 4_000))
        for index, hit in enumerate(hits, 1):
            lines.append(f"{index}. {hit.title}\n   {hit.url}\n   {truncate_end(hit.snippet, 300)}")
        if not hits:
            lines.append("No results.")
        return ToolResult.success("\n".join(lines), results=len(hits))


class FetchUrlTool(Tool):
    name = "fetch_url"
    group = "web"
    description = "Download a web page or text document and return its readable text (http/https only)."
    parameters = object_schema(
        {"url": {"type": "string"}, "max_chars": {"type": "integer"}},
        ["url"],
    )
    required = frozenset({Capability.INTERNET})

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.http is None:
            return ToolResult.failure("HTTP client unavailable")
        web = ctx.tools_config.web
        url = str(args["url"]).strip()
        await ctx.approval.require(ProposedAction(kind=ActionKind.INTERNET, agent=ctx.agent_name, target=url, risk=RiskLevel.LOW))
        max_chars = max(500, min(int(args.get("max_chars") or web.fetch_max_chars), web.fetch_max_chars))
        current = url
        for _redirect in range(6):
            await ensure_public_url(current, web.allow_private_networks)
            try:
                async with ctx.http.stream("GET", current, headers={"User-Agent": web.user_agent}, timeout=web.timeout_s, follow_redirects=False) as response:
                    if response.is_redirect and response.headers.get("location"):
                        current = urljoin(current, response.headers["location"])
                        continue
                    if response.status_code >= 400:
                        return ToolResult.failure(f"HTTP {response.status_code} for {current}")
                    content_type = response.headers.get("content-type", "")
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) >= web.fetch_max_bytes:
                            break
                    encoding = response.encoding or "utf-8"
            except httpx.HTTPError as exc:
                return ToolResult.failure(f"Fetch failed: {type(exc).__name__}: {exc}")
            break
        else:
            return ToolResult.failure("Too many redirects")
        text = bytes(body).decode(encoding, errors="replace")
        title = ""
        if "html" in content_type or text.lstrip().lower().startswith(("<!doctype html", "<html")):
            extractor = _TextExtractor()
            extractor.feed(text)
            title, text = extractor.title.strip(), extractor.text()
        elif not any(kind in content_type for kind in ("text", "json", "xml", "javascript", "markdown")) and content_type:
            return ToolResult.failure(f"Unsupported content type: {content_type}")
        if ctx.collaboration is not None:
            ctx.collaboration.record_sources(ctx.agent_name, f"fetch: {current}", [{"title": title or current, "url": current, "snippet": text[:300]}], ctx.node_id)
        truncated = len(text) > max_chars
        return ToolResult.success(
            f"{title or current}\nURL: {current}\n\n{text[:max_chars]}" + ("\n… (truncated)" if truncated else ""),
            url=current,
            title=title,
        )


TOOLS = [WebSearchTool, FetchUrlTool]
