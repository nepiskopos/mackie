"""Unit tests for agent/tools.py — tool call handlers."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

_EMPTY_RESEARCH = {
    "org_name": "Test Org", "website_content": "", "social_presence": [],
    "news": [], "similar_orgs": [], "sources": [],
}


async def test_save_post_returns_post_id(tmp_data_dir):
    from agent.tools import handle_tool_call
    result = await handle_tool_call(
        "save_post",
        {"platform": "linkedin", "content": "Test post", "status": "draft"},
        "test-org",
    )
    assert "post_001" in result


async def test_save_post_default_status_is_suggested(tmp_data_dir):
    from agent.tools import handle_tool_call
    from agent.memory import get_ledger
    await handle_tool_call("save_post", {"platform": "linkedin", "content": "Hello"}, "test-org")
    assert get_ledger("test-org")[0]["status"] == "suggested"


async def test_update_post_status_existing_post(tmp_data_dir):
    from agent.memory import add_post
    from agent.tools import handle_tool_call
    add_post("test-org", {"platform": "linkedin", "content": "Hello"})
    result = await handle_tool_call(
        "update_post_status", {"post_id": "post_001", "status": "approved"}, "test-org"
    )
    assert "approved" in result


async def test_update_post_status_missing_post(tmp_data_dir):
    from agent.tools import handle_tool_call
    result = await handle_tool_call(
        "update_post_status", {"post_id": "post_999", "status": "approved"}, "test-org"
    )
    assert "not found" in result


async def test_save_preference_persists_to_profile(tmp_data_dir):
    from agent.tools import handle_tool_call
    from agent.memory import load_profile
    await handle_tool_call("save_preference", {"key": "voice", "value": "warm"}, "test-org")
    assert load_profile("test-org")["preferences"]["voice"] == "warm"


async def test_research_org_tool_saves_to_profile(tmp_data_dir):
    from agent.tools import handle_tool_call
    from agent.memory import load_profile

    mock_research = {
        "org_name": "Test NPO",
        "website_content": "We help kids.",
        "social_presence": [],
        "news": [],
        "similar_orgs": [],
        "sources": [{"type": "website", "url": "https://testnpo.org"}],
    }
    with patch("agent.tools.res.research_org", new=AsyncMock(return_value=mock_research)):
        await handle_tool_call(
            "research_org",
            {"org_name": "Test NPO", "org_url": "https://testnpo.org"},
            "test-org",
        )

    profile = load_profile("test-org")
    assert profile["research"] is not None
    assert profile["name"] == "Test NPO"
    assert profile["website"] == "https://testnpo.org"


async def test_research_org_tool_does_not_overwrite_existing_name_or_website(tmp_data_dir):
    from agent.tools import handle_tool_call
    from agent.memory import save_profile, load_profile

    save_profile("test-org", {
        "org_id": "test-org", "name": "Original Name", "website": "https://original.com",
        "research": None, "preferences": {}, "post_ledger": [],
    })
    with patch("agent.tools.res.research_org", new=AsyncMock(return_value=_EMPTY_RESEARCH)):
        await handle_tool_call(
            "research_org",
            {"org_name": "Different Name", "org_url": "https://different.com"},
            "test-org",
        )

    profile = load_profile("test-org")
    assert profile["name"] == "Original Name"
    assert profile["website"] == "https://original.com"
    assert profile["research"] is not None  # research itself IS updated


async def test_web_search_tool_formats_results(tmp_data_dir):
    from agent.tools import handle_tool_call

    mock_results = [
        {"title": "NPO News", "url": "https://example.com", "snippet": "Great work"}
    ]
    with patch("agent.tools.res.search_web", new=AsyncMock(return_value=mock_results)):
        result = await handle_tool_call("web_search", {"query": "nonprofit news"}, "test-org")
    assert "NPO News" in result
    assert "example.com" in result


async def test_unknown_tool_returns_error_message(tmp_data_dir):
    from agent.tools import handle_tool_call
    result = await handle_tool_call("nonexistent_tool", {}, "test-org")
    assert "Unknown" in result


async def test_refresh_research_updates_research_field(tmp_data_dir):
    from agent.tools import handle_tool_call
    from agent.memory import save_profile, load_profile

    save_profile("test-org", {
        "org_id": "test-org", "name": "Test NPO", "website": "",
        "research": {"website_content": "old content"},
        "preferences": {"voice": "warm"}, "post_ledger": [],
    })
    new_research = {
        "org_name": "Test NPO", "website_content": "new content",
        "social_presence": [], "news": [], "similar_orgs": [], "sources": [],
    }
    with patch("agent.tools.res.research_org", new=AsyncMock(return_value=new_research)):
        await handle_tool_call("refresh_research", {"org_name": "Test NPO"}, "test-org")

    profile = load_profile("test-org")
    assert profile["research"]["website_content"] == "new content"


async def test_refresh_research_preserves_preferences_and_ledger(tmp_data_dir):
    from agent.tools import handle_tool_call
    from agent.memory import save_profile, load_profile, add_post

    save_profile("test-org", {
        "org_id": "test-org", "name": "Test NPO", "website": "",
        "research": None, "preferences": {"voice": "warm"}, "post_ledger": [],
    })
    add_post("test-org", {"platform": "linkedin", "content": "Existing post"})

    new_research = {
        "org_name": "Test NPO", "website_content": "fresh",
        "social_presence": [], "news": [], "similar_orgs": [], "sources": [],
    }
    with patch("agent.tools.res.research_org", new=AsyncMock(return_value=new_research)):
        await handle_tool_call("refresh_research", {"org_name": "Test NPO"}, "test-org")

    profile = load_profile("test-org")
    assert profile["preferences"]["voice"] == "warm"
    assert len(profile["post_ledger"]) == 1


async def test_refresh_research_adds_timestamp(tmp_data_dir):
    from agent.tools import handle_tool_call
    from agent.memory import load_profile

    new_research = {
        "org_name": "Test NPO", "website_content": "",
        "social_presence": [], "news": [], "similar_orgs": [], "sources": [],
    }
    with patch("agent.tools.res.research_org", new=AsyncMock(return_value=new_research)):
        await handle_tool_call("refresh_research", {"org_name": "Test NPO"}, "test-org")

    assert "research_updated_at" in load_profile("test-org")


async def test_generate_calendar_creates_four_planned_posts(tmp_data_dir):
    from agent.tools import handle_tool_call
    from agent.memory import save_profile, get_ledger

    save_profile("test-org", {
        "org_id": "test-org", "name": "Test NPO", "website": "",
        "research": {"website_content": "We help people.", "news": [], "similar_orgs": [], "sources": []},
        "preferences": {}, "post_ledger": [],
    })
    week_json = json.dumps({
        "platform": "linkedin", "angle": "Highlight our impact",
        "pillar": "Community Impact", "rationale": "From website content",
    })
    mock_response = MagicMock(choices=[MagicMock(message=MagicMock(content=week_json))])
    with patch("agent.tools.litellm.acompletion", new=AsyncMock(return_value=mock_response)):
        result = await handle_tool_call("generate_calendar", {"month_year": "July 2026"}, "test-org")

    ledger = get_ledger("test-org")
    assert len(ledger) == 4
    assert all(p["status"] == "planned" for p in ledger)
    assert "July 2026" in result


async def test_generate_calendar_returns_markdown_table(tmp_data_dir):
    from agent.tools import handle_tool_call
    from agent.memory import save_profile

    save_profile("test-org", {
        "org_id": "test-org", "name": "Test NPO", "website": "",
        "research": None, "preferences": {}, "post_ledger": [],
    })
    week_json = json.dumps({
        "platform": "instagram", "angle": "Behind the scenes",
        "pillar": "Behind the Scenes", "rationale": "No research yet",
    })
    mock_response = MagicMock(choices=[MagicMock(message=MagicMock(content=week_json))])
    with patch("agent.tools.litellm.acompletion", new=AsyncMock(return_value=mock_response)):
        result = await handle_tool_call("generate_calendar", {"month_year": "August 2026"}, "test-org")

    assert "Content Calendar" in result
    assert "Week 1" in result
    assert "Week 4" in result


async def test_handle_tool_call_writes_trace_entry(tmp_data_dir):
    from agent.tools import handle_tool_call
    await handle_tool_call("save_post", {"platform": "linkedin", "content": "Test"}, "test-org")
    trace = tmp_data_dir / "test-org" / "trace.jsonl"
    assert trace.exists()
    entry = json.loads(trace.read_text().strip())
    assert entry["tool"] == "save_post"
    assert entry["latency_ms"] >= 0
    assert "input" in entry
    assert "result_preview" in entry


async def test_trace_appends_multiple_entries(tmp_data_dir):
    from agent.tools import handle_tool_call
    await handle_tool_call("save_post", {"platform": "linkedin", "content": "First"}, "test-org")
    await handle_tool_call("save_post", {"platform": "instagram", "content": "Second"}, "test-org")
    trace = tmp_data_dir / "test-org" / "trace.jsonl"
    entries = [json.loads(line) for line in trace.read_text().strip().splitlines()]
    assert len(entries) == 2
    assert entries[0]["tool"] == "save_post"
    assert entries[1]["tool"] == "save_post"


def test_format_research_summary_includes_org_name():
    from agent.tools import _format_research_summary
    result = {
        "website_content": "", "social_presence": [], "news": [],
        "similar_orgs": [], "sources": [],
    }
    summary = _format_research_summary("Test NPO", result)
    assert "Test NPO" in summary
    assert "Research complete" in summary


def test_format_research_summary_truncates_website_to_3000():
    from agent.tools import _format_research_summary
    result = {
        "website_content": "A" * 3000 + "B" * 2000,
        "social_presence": [], "news": [], "similar_orgs": [], "sources": [],
    }
    summary = _format_research_summary("Test NPO", result)
    assert "A" * 50 in summary
    assert "B" not in summary


def test_format_research_summary_limits_and_filters_results():
    from agent.tools import _format_research_summary
    social = [
        {"title": "Social A", "url": "https://a.com", "snippet": "About A"},
        {"title": "Social B", "url": "https://b.com", "snippet": "About B"},
        {"error": "fetch failed"},
        {"title": "Social D", "url": "https://d.com", "snippet": "About D"},
        {"title": "Social E", "url": "https://e.com", "snippet": "About E"},
    ]
    result = {
        "website_content": "", "social_presence": social,
        "news": [], "similar_orgs": [], "sources": [],
    }
    summary = _format_research_summary("Test NPO", result)
    assert "Social A" in summary
    assert "Social B" in summary
    assert "Social D" not in summary  # sliced off by [:3]
    assert "Social E" not in summary  # sliced off by [:3]


def test_format_research_summary_wraps_website_in_external_content_tags():
    from agent.tools import _format_research_summary
    result = {
        "website_content": "We help communities thrive.",
        "social_presence": [], "news": [], "similar_orgs": [], "sources": [],
    }
    summary = _format_research_summary("Test NPO", result)
    assert '<external_content source="website">' in summary
    assert "We help communities thrive." in summary
    assert "</external_content>" in summary


def test_format_research_summary_wraps_search_results_in_external_content_tags():
    from agent.tools import _format_research_summary
    result = {
        "website_content": "",
        "social_presence": [{"title": "Post", "url": "https://fb.com/p", "snippet": "Great event"}],
        "news": [], "similar_orgs": [], "sources": [],
    }
    summary = _format_research_summary("Test NPO", result)
    assert '<external_content source="web_search"' in summary
    assert "Great event" in summary
    assert "</external_content>" in summary


async def test_generate_calendar_handles_partial_failure(tmp_data_dir):
    from agent.tools import handle_tool_call
    from agent.memory import save_profile, get_ledger

    save_profile("test-org", {
        "org_id": "test-org", "name": "Test NPO", "website": "",
        "research": None, "preferences": {}, "post_ledger": [],
    })
    week_json = json.dumps({
        "platform": "linkedin", "angle": "Community story",
        "pillar": "Community Impact", "rationale": "From research",
    })
    mock_ok = MagicMock(choices=[MagicMock(message=MagicMock(content=week_json))])
    with patch("agent.tools.litellm.acompletion", new=AsyncMock(
        side_effect=[mock_ok, Exception("timeout"), mock_ok, mock_ok]
    )):
        result = await handle_tool_call("generate_calendar", {"month_year": "August 2026"}, "test-org")

    assert "3 of 4 weeks saved" in result
    assert "Error generating plan" in result
    assert len(get_ledger("test-org")) == 3
