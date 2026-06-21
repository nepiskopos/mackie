"""Unit tests for agent/assistant.py — system prompt construction and chat loop."""
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


async def _text_stream(text: str):
    """Async generator that yields one text chunk then a stop-reason sentinel."""
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=text, tool_calls=None), finish_reason=None)]
    yield chunk
    final = MagicMock()
    final.choices = [MagicMock(delta=MagicMock(content=None, tool_calls=None), finish_reason="stop")]
    yield final


async def _tool_call_stream(tc_id: str, name: str, arguments: str):
    """Async generator that yields a single tool call chunk then a tool_calls sentinel."""
    tc_delta = MagicMock()
    tc_delta.index = 0
    tc_delta.id = tc_id
    tc_delta.function.name = name
    tc_delta.function.arguments = arguments
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=None, tool_calls=[tc_delta]), finish_reason=None)]
    yield chunk
    final = MagicMock()
    final.choices = [MagicMock(delta=MagicMock(content=None, tool_calls=None), finish_reason="tool_calls")]
    yield final


def test_system_prompt_shows_no_research_warning(tmp_data_dir):
    from agent.memory import save_profile
    from agent.assistant import _build_system_prompt

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": None, "preferences": {}, "post_ledger": [],
    })
    prompt = _build_system_prompt("org")
    assert "Not yet done" in prompt


def test_system_prompt_shows_research_complete(tmp_data_dir):
    from agent.memory import save_profile
    from agent.assistant import _build_system_prompt

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": {"website_content": "some content"},
        "preferences": {}, "post_ledger": [],
    })
    prompt = _build_system_prompt("org")
    assert "Complete" in prompt


def test_system_prompt_includes_saved_preferences(tmp_data_dir):
    from agent.memory import save_profile
    from agent.assistant import _build_system_prompt

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": None,
        "preferences": {"voice": "warm and grassroots", "banned_topics": "politics"},
        "post_ledger": [],
    })
    prompt = _build_system_prompt("org")
    assert "warm and grassroots" in prompt
    assert "politics" in prompt


def test_system_prompt_includes_ledger_entries(tmp_data_dir):
    from agent.memory import save_profile
    from agent.assistant import _build_system_prompt

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": None, "preferences": {},
        "post_ledger": [{
            "id": "post_001", "platform": "linkedin", "status": "draft",
            "content": "Dignity matters for every patient.",
            "created_at": "2026-06-12T00:00:00",
        }],
    })
    prompt = _build_system_prompt("org")
    assert "post_001" in prompt
    assert "LINKEDIN" in prompt
    assert "draft" in prompt


def test_system_prompt_truncates_ledger_to_twenty(tmp_data_dir):
    from agent.memory import save_profile
    from agent.assistant import _build_system_prompt

    ledger = [
        {
            "id": f"post_{i:03d}", "platform": "linkedin", "status": "suggested",
            "content": f"Post number {i}.", "created_at": "2026-06-12T00:00:00",
        }
        for i in range(1, 26)  # 25 posts
    ]
    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": None, "preferences": {}, "post_ledger": ledger,
    })
    prompt = _build_system_prompt("org")
    assert "25 total" in prompt
    assert "5 older posts" in prompt
    assert "post_025" in prompt   # most recent is shown
    assert "post_001" not in prompt  # oldest is truncated


def test_system_prompt_includes_platform_skills(tmp_data_dir):
    from agent.memory import save_profile
    from agent.assistant import _build_system_prompt

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": None, "preferences": {}, "post_ledger": [],
    })
    prompt = _build_system_prompt("org")
    assert "LinkedIn" in prompt
    assert "Instagram" in prompt
    assert "Facebook" in prompt


def test_system_prompt_includes_content_pillars(tmp_data_dir):
    from agent.memory import save_profile
    from agent.assistant import _build_system_prompt

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": None, "preferences": {}, "post_ledger": [],
    })
    prompt = _build_system_prompt("org")
    assert "Content Pillars" in prompt
    assert "Community Impact" in prompt


def test_system_prompt_no_preferences_section_when_empty(tmp_data_dir):
    from agent.memory import save_profile
    from agent.assistant import _build_system_prompt

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": None, "preferences": {}, "post_ledger": [],
    })
    prompt = _build_system_prompt("org")
    assert "Learned Brand Preferences" not in prompt


async def test_chat_returns_assistant_reply(tmp_data_dir):
    from agent.assistant import Assistant

    with patch("agent.graph.litellm.acompletion", new=AsyncMock(return_value=_text_stream("Hello, I'm Mackie!"))):
        assistant = Assistant("org")
        reply = await assistant.chat("Hello")
    assert "Mackie" in reply


def test_system_prompt_shows_research_sources(tmp_data_dir):
    from agent.memory import save_profile
    from agent.assistant import _build_system_prompt

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": {
            "website_content": "some content",
            "sources": [
                {"type": "website", "url": "https://example.org"},
                {"type": "news", "url": "https://news.org/article", "title": "Great article"},
            ],
        },
        "preferences": {}, "post_ledger": [],
    })
    prompt = _build_system_prompt("org")
    assert "Research sources" in prompt
    assert "https://example.org" in prompt
    assert "Great article" in prompt


def test_system_prompt_no_sources_section_when_research_has_none(tmp_data_dir):
    from agent.memory import save_profile
    from agent.assistant import _build_system_prompt

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": {"website_content": "some content", "sources": []},
        "preferences": {}, "post_ledger": [],
    })
    prompt = _build_system_prompt("org")
    assert "Research sources" not in prompt


def test_system_prompt_includes_approval_guidance(tmp_data_dir):
    from agent.memory import save_profile
    from agent.assistant import _build_system_prompt

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": None, "preferences": {}, "post_ledger": [],
    })
    prompt = _build_system_prompt("org")
    assert "approved" in prompt
    assert "posted" in prompt


def test_system_prompt_shows_research_updated_at(tmp_data_dir):
    from agent.memory import save_profile
    from agent.assistant import _build_system_prompt

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": {"website_content": "some content"},
        "research_updated_at": "2026-06-12T10:00:00",
        "preferences": {}, "post_ledger": [],
    })
    prompt = _build_system_prompt("org")
    assert "2026-06-12T10:00:00" in prompt


def test_system_prompt_includes_security_policy(tmp_data_dir):
    from agent.memory import save_profile
    from agent.assistant import _build_system_prompt

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": None, "preferences": {}, "post_ledger": [],
    })
    prompt = _build_system_prompt("org")
    assert "Security Policy" in prompt
    assert "<external_content>" in prompt or "external_content" in prompt
    assert "Ignore previous instructions" in prompt  # named example of injection pattern


async def test_chat_persists_conversation_to_checkpoint(tmp_data_dir):
    from agent.assistant import Assistant
    from agent.graph import build_graph
    from langchain_core.messages import AIMessage, HumanMessage

    with patch("agent.graph.litellm.acompletion", new=AsyncMock(return_value=_text_stream("Reply text"))):
        assistant = Assistant("org")
        reply = await assistant.chat("Hello")

    assert reply == "Reply text"
    config = {"configurable": {"thread_id": assistant.thread_id}}
    async with build_graph(assistant.model) as graph:
        state = await graph.aget_state(config)
    msgs = state.values.get("messages", [])
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].content == "Reply text"


async def test_custom_thread_id_is_isolated_from_org_thread(tmp_data_dir):
    from agent.assistant import Assistant
    from agent.graph import build_graph
    from langchain_core.messages import HumanMessage

    with patch("agent.graph.litellm.acompletion", new=AsyncMock(return_value=_text_stream("Session reply"))):
        assistant = Assistant("org", thread_id="org-session-abc")
        await assistant.chat("Hello")

    # org default thread should be untouched
    async with build_graph(assistant.model) as graph:
        org_state = await graph.aget_state({"configurable": {"thread_id": "org"}})
    assert org_state.values.get("messages", []) == []

    # custom thread should have the conversation
    async with build_graph(assistant.model) as graph:
        session_state = await graph.aget_state({"configurable": {"thread_id": "org-session-abc"}})
    assert len(session_state.values.get("messages", [])) == 2


def test_load_platform_skills_skips_files_without_id(tmp_path):
    from agent.assistant import _load_platform_skills

    (tmp_path / "platform.yaml").write_text(
        "name: Test Platform\nid: test\ntone: casual\nmax_length: 500\n"
    )
    (tmp_path / "no_id.yaml").write_text("pillars:\n  - name: Some Pillar\n")

    with patch("agent.assistant.SKILLS_DIR", tmp_path):
        result = _load_platform_skills()

    assert "test" in result          # platform yaml included
    assert "Some Pillar" not in result  # file without id skipped


def test_system_prompt_uses_content_pillars_profile_override(tmp_data_dir):
    from agent.memory import save_profile
    from agent.assistant import _build_system_prompt

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": None, "preferences": {}, "post_ledger": [],
        "content_pillars": [{"name": "Custom Pillar", "description": "Our unique focus area"}],
    })
    prompt = _build_system_prompt("org")
    assert "Custom Pillar" in prompt
    assert "Community Impact" not in prompt


async def test_achat_stream_yields_text_event(tmp_data_dir):
    from agent.assistant import Assistant

    with patch("agent.graph.litellm.acompletion", new=AsyncMock(return_value=_text_stream("Hello from Mackie!"))):
        assistant = Assistant("org")
        events = [e async for e in assistant.achat_stream("Hello")]

    text_events = [e for e in events if e["type"] == "text"]
    assert len(text_events) == 1
    assert text_events[0]["content"] == "Hello from Mackie!"


async def test_achat_stream_yields_tool_step_then_text(tmp_data_dir):
    from agent.assistant import Assistant

    with patch(
        "agent.graph.litellm.acompletion",
        new=AsyncMock(side_effect=[
            _tool_call_stream("tc_001", "save_post", '{"platform": "linkedin", "content": "Test post"}'),
            _text_stream("Post saved!"),
        ]),
    ):
        assistant = Assistant("org")
        events = [e async for e in assistant.achat_stream("Save a post")]

    tool_events = [e for e in events if e["type"] == "tool_step"]
    text_events = [e for e in events if e["type"] == "text"]
    assert len(tool_events) == 1
    assert tool_events[0]["tool"] == "save_post"
    assert "input_preview" in tool_events[0]
    assert "result_preview" in tool_events[0]
    assert "latency_ms" in tool_events[0]
    assert len(text_events) == 1
    assert text_events[0]["content"] == "Post saved!"
    # tool_step must arrive before the final text reply
    assert events.index(tool_events[0]) < events.index(text_events[0])
