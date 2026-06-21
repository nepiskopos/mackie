"""
Website scraping and web search for org research.

Owns: fetching and parsing org websites, searching the web via Tavily, and
combining results into a structured research dict. All I/O runs in parallel
via asyncio.gather. Does not own memory, tool dispatch, or the conversation loop.
"""
from __future__ import annotations

import asyncio
import ipaddress
import os
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from tavily import AsyncTavilyClient

_TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")

# Hostnames that must never be fetched — cloud metadata endpoints and localhost.
# IP-based checks below cover the rest of the private/loopback space.
_BLOCKED_HOSTS: frozenset[str] = frozenset({
    "localhost",
    "169.254.169.254",       # AWS / Azure / GCP instance metadata
    "metadata.google.internal",  # GCP metadata alternate hostname
})


def _is_safe_url(url: str) -> bool:
    """
    Return True only if the URL is safe to fetch (public internet, HTTPS or HTTP).

    Blocks:
    - Non-HTTP(S) schemes (file://, ftp://, etc.)
    - Known metadata hostnames
    - Loopback addresses (127.x.x.x, ::1)
    - Private/link-local ranges (10.x, 172.16-31.x, 192.168.x, 169.254.x)
    - Non-standard IP representations (hex "0x7f000001", decimal integer "2130706433")
      that ipaddress.ip_address() would not catch but some OS resolvers accept

    Args:
        url: Fully-qualified URL (https:// prefix already applied).

    Returns:
        True if the URL is safe to fetch, False otherwise.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    host = parsed.hostname or ""
    if not host:
        return False

    if host.lower() in _BLOCKED_HOSTS:
        return False

    # Try parsing as a standard IP address (dotted decimal or colon notation).
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_loopback or addr.is_private or addr.is_link_local or addr.is_reserved:
            return False
    except ValueError:
        pass  # host is a domain name — IP checks don't apply

    # Reject non-standard IP representations (hex "0x7f000001", decimal integer
    # "2130706433", octal "0177.0.0.1") that ipaddress.ip_address() won't parse
    # but some resolvers will treat as 127.0.0.1.
    if host.startswith("0x") or host.startswith("0X"):
        return False
    if host.isdigit():
        # Bare integer — could be a packed IPv4 address (e.g. 2130706433 → 127.0.0.1)
        try:
            addr = ipaddress.ip_address(int(host))
            if addr.is_loopback or addr.is_private or addr.is_link_local or addr.is_reserved:
                return False
        except ValueError:
            pass

    return True


async def scrape_website(url: str) -> str:
    """
    Fetch and extract plain text from a URL, stripping scripts, styles, and navigation.

    Rejects private/loopback/metadata URLs before making any network request to
    prevent SSRF attacks (e.g. fetching cloud instance metadata endpoints).

    Args:
        url: Website URL. An "https://" prefix is added if missing.

    Returns:
        Plain text content (up to 8000 chars), or an error message string on failure.
    """
    if not url.startswith("http"):
        url = "https://" + url

    if not _is_safe_url(url):
        return f"Error: URL '{url}' is not allowed (private or reserved address)."

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                timeout=15,
                follow_redirects=False,  # don't follow redirects — they can bypass the check above
                headers={"User-Agent": "Mozilla/5.0 (compatible; MackieBot/1.0)"},
            )
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:8000]
    except Exception as e:
        return f"Could not scrape {url}: {e}"


async def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Search the web via Tavily and return structured results.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, url, snippet. On error, returns
        a single-element list with an "error" key.
    """
    try:
        client = AsyncTavilyClient(api_key=_TAVILY_API_KEY)
        response = await client.search(query, max_results=max_results)
        return [
            {"title": r["title"], "url": r["url"], "snippet": r["content"]}
            for r in response.get("results", [])
        ]
    except Exception as e:
        return [{"error": str(e)}]


async def research_org(org_name: str, org_url: str = "") -> dict[str, Any]:
    """
    Research an org by scraping its website and running web searches in parallel.

    Args:
        org_name: Human-readable name used as the search query base.
        org_url:  Optional website URL. If provided, the page is scraped.

    Returns:
        Dict with keys: org_name, website_content, social_presence, news,
        similar_orgs, sources (list of dicts with type/url/title).
    """
    social_coro = search_web(
        f'"{org_name}" site:instagram.com OR site:facebook.com OR site:linkedin.com', 4
    )
    news_coro = search_web(f'"{org_name}" nonprofit 2025 OR 2026', 5)
    similar_coro = search_web(
        f'nonprofit similar to "{org_name}" social media content strategy', 3
    )

    if org_url:
        website_content, social_results, news_results, similar_results = await asyncio.gather(
            scrape_website(org_url), social_coro, news_coro, similar_coro
        )
    else:
        website_content = ""
        social_results, news_results, similar_results = await asyncio.gather(
            social_coro, news_coro, similar_coro
        )

    research: dict[str, Any] = {
        "org_name": org_name,
        "website_content": website_content,
        "social_presence": social_results,
        "news": news_results,
        "similar_orgs": similar_results,
        "sources": [],
    }

    if org_url:
        research["sources"].append({"type": "website", "url": org_url})
    for r in social_results:
        if "error" not in r:
            research["sources"].append({"type": "social", "url": r.get("url"), "title": r.get("title")})
    for r in news_results:
        if "error" not in r:
            research["sources"].append({"type": "news", "url": r.get("url"), "title": r.get("title")})

    return research
