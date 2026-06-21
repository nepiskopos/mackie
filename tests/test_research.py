"""Unit tests for agent/research.py — website scraping and web search."""
from unittest.mock import AsyncMock, MagicMock, patch


async def test_scrape_website_returns_text_content():
    from agent.research import scrape_website
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Mission: help kids.</p></body></html>"
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    with patch("agent.research.httpx.AsyncClient") as mock_class:
        mock_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_class.return_value.__aexit__ = AsyncMock(return_value=None)
        result = await scrape_website("https://example.com")
    assert "Mission: help kids" in result


async def test_scrape_website_strips_scripts_and_styles():
    from agent.research import scrape_website
    mock_response = MagicMock()
    mock_response.text = (
        "<html><body><p>Keep this</p>"
        "<script>remove this</script>"
        "<style>.remove { color: red; }</style></body></html>"
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    with patch("agent.research.httpx.AsyncClient") as mock_class:
        mock_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_class.return_value.__aexit__ = AsyncMock(return_value=None)
        result = await scrape_website("https://example.com")
    assert "Keep this" in result
    assert "remove this" not in result


async def test_scrape_website_prepends_https_when_missing():
    from agent.research import scrape_website
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Content</p></body></html>"
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    with patch("agent.research.httpx.AsyncClient") as mock_class:
        mock_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_class.return_value.__aexit__ = AsyncMock(return_value=None)
        await scrape_website("example.com")
    called_url = mock_client.get.call_args[0][0]
    assert called_url.startswith("https://")


async def test_scrape_website_returns_error_message_on_failure():
    from agent.research import scrape_website
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("timeout"))
    with patch("agent.research.httpx.AsyncClient") as mock_class:
        mock_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_class.return_value.__aexit__ = AsyncMock(return_value=None)
        result = await scrape_website("https://broken.example.com")
    assert "Could not scrape" in result


async def test_search_web_returns_formatted_results():
    from agent.research import search_web
    mock_response = {
        "results": [
            {"title": "Title 1", "url": "https://a.com", "content": "Snippet 1"},
            {"title": "Title 2", "url": "https://b.com", "content": "Snippet 2"},
        ]
    }
    with patch("agent.research.AsyncTavilyClient") as mock_class:
        mock_instance = AsyncMock()
        mock_instance.search = AsyncMock(return_value=mock_response)
        mock_class.return_value = mock_instance
        results = await search_web("test query", max_results=2)
    assert len(results) == 2
    assert results[0]["title"] == "Title 1"
    assert results[0]["url"] == "https://a.com"
    assert results[0]["snippet"] == "Snippet 1"


async def test_search_web_handles_exception():
    from agent.research import search_web
    with patch("agent.research.AsyncTavilyClient") as mock_class:
        mock_instance = AsyncMock()
        mock_instance.search = AsyncMock(side_effect=Exception("api error"))
        mock_class.return_value = mock_instance
        results = await search_web("test query")
    assert len(results) == 1
    assert "error" in results[0]


async def test_research_org_with_url_returns_structured_result():
    from agent.research import research_org
    search_result = [{"title": "T", "url": "https://x.com", "snippet": "S"}]
    with (
        patch("agent.research.scrape_website", new=AsyncMock(return_value="scraped content")) as mock_scrape,
        patch("agent.research.search_web", new=AsyncMock(return_value=search_result)) as mock_search,
    ):
        result = await research_org("Test Org", "https://test.org")

    for key in ("org_name", "website_content", "social_presence", "news", "similar_orgs", "sources"):
        assert key in result
    assert result["website_content"] == "scraped content"
    mock_scrape.assert_called_once_with("https://test.org")
    assert mock_search.call_count == 3


async def test_research_org_without_url_skips_scrape():
    from agent.research import research_org
    search_result = [{"title": "T", "url": "https://x.com", "snippet": "S"}]
    with (
        patch("agent.research.scrape_website", new=AsyncMock(return_value="scraped content")) as mock_scrape,
        patch("agent.research.search_web", new=AsyncMock(return_value=search_result)) as mock_search,
    ):
        result = await research_org("Test Org")

    mock_scrape.assert_not_called()
    assert mock_search.call_count == 3
    assert result["website_content"] == ""


def test_is_safe_url_allows_public_https():
    from agent.research import _is_safe_url
    assert _is_safe_url("https://brcastrong.org") is True
    assert _is_safe_url("http://example.com") is True


def test_is_safe_url_blocks_localhost():
    from agent.research import _is_safe_url
    assert _is_safe_url("http://localhost/admin") is False
    assert _is_safe_url("http://127.0.0.1:8080") is False
    assert _is_safe_url("http://127.255.255.255") is False


def test_is_safe_url_blocks_aws_metadata():
    from agent.research import _is_safe_url
    assert _is_safe_url("http://169.254.169.254/latest/meta-data/") is False


def test_is_safe_url_blocks_gcp_metadata():
    from agent.research import _is_safe_url
    assert _is_safe_url("http://metadata.google.internal/") is False


def test_is_safe_url_blocks_private_ranges():
    from agent.research import _is_safe_url
    assert _is_safe_url("http://10.0.0.1") is False
    assert _is_safe_url("http://192.168.1.1") is False
    assert _is_safe_url("http://172.16.0.1") is False


def test_is_safe_url_blocks_nonstandard_ip_representations():
    from agent.research import _is_safe_url
    # Hex notation — 0x7f000001 = 127.0.0.1
    assert _is_safe_url("http://0x7f000001/") is False
    # Decimal integer — 2130706433 = 127.0.0.1
    assert _is_safe_url("http://2130706433/") is False


def test_is_safe_url_blocks_non_http_schemes():
    from agent.research import _is_safe_url
    assert _is_safe_url("file:///etc/passwd") is False
    assert _is_safe_url("ftp://files.example.com") is False


async def test_scrape_website_rejects_blocked_url():
    from agent.research import scrape_website
    result = await scrape_website("http://169.254.169.254/")
    assert "not allowed" in result


async def test_scrape_website_rejects_localhost():
    from agent.research import scrape_website
    result = await scrape_website("http://localhost:6379/")
    assert "not allowed" in result


async def test_research_org_populates_sources():
    from agent.research import research_org
    social = [{"title": "FB Page", "url": "https://fb.com/org", "snippet": "Social"}]
    news = [{"title": "News Item", "url": "https://news.com/article", "snippet": "News"}]
    similar = [{"title": "Similar Org", "url": "https://sim.org", "snippet": "Similar"}]
    with (
        patch("agent.research.scrape_website", new=AsyncMock(return_value="content")),
        patch("agent.research.search_web", new=AsyncMock(side_effect=[social, news, similar])),
    ):
        result = await research_org("Test Org", "https://test.org")

    source_types = {s["type"] for s in result["sources"]}
    assert "website" in source_types
    assert "social" in source_types
    assert "news" in source_types
    website_src = next(s for s in result["sources"] if s["type"] == "website")
    assert website_src["url"] == "https://test.org"
