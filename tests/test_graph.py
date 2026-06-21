"""Unit tests for agent/graph.py — message conversion, graph routing, and prompt caching."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END


def test_to_litellm_converts_human_message():
    from agent.graph import _to_litellm
    result = _to_litellm([HumanMessage(content="Hello")])
    assert result == [{"role": "user", "content": "Hello"}]


def test_to_litellm_converts_ai_message_without_tool_calls():
    from agent.graph import _to_litellm
    result = _to_litellm([AIMessage(content="Here is my reply.")])
    assert result == [{"role": "assistant", "content": "Here is my reply."}]


def test_to_litellm_converts_ai_message_with_tool_calls():
    from agent.graph import _to_litellm
    msg = AIMessage(
        content="",
        tool_calls=[{
            "id": "tc_1", "name": "save_post",
            "args": {"platform": "linkedin", "content": "Hi"},
            "type": "tool_call",
        }],
    )
    result = _to_litellm([msg])
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert result[0]["content"] is None
    tc = result[0]["tool_calls"][0]
    assert tc["id"] == "tc_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "save_post"
    assert json.loads(tc["function"]["arguments"]) == {"platform": "linkedin", "content": "Hi"}


def test_to_litellm_converts_tool_message():
    from agent.graph import _to_litellm
    msg = ToolMessage(content="Saved as post_001", tool_call_id="tc_1", name="save_post")
    result = _to_litellm([msg])
    assert result == [{
        "role": "tool",
        "tool_call_id": "tc_1",
        "name": "save_post",
        "content": "Saved as post_001",
    }]


def test_to_litellm_handles_mixed_message_sequence():
    from agent.graph import _to_litellm
    msgs = [
        HumanMessage(content="Save a post"),
        AIMessage(
            content="",
            tool_calls=[{"id": "tc_1", "name": "save_post", "args": {"platform": "linkedin", "content": "Hi"}, "type": "tool_call"}],
        ),
        ToolMessage(content="Saved as post_001", tool_call_id="tc_1", name="save_post"),
        AIMessage(content="Done!"),
    ]
    result = _to_litellm(msgs)
    assert len(result) == 4
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert result[1]["tool_calls"] is not None
    assert result[2]["role"] == "tool"
    assert result[3]["role"] == "assistant"
    assert result[3]["content"] == "Done!"


def test_from_litellm_extracts_tool_calls():
    from agent.graph import _from_litellm
    tc = MagicMock()
    tc.id = "tc_1"
    tc.function.name = "save_post"
    tc.function.arguments = '{"platform": "linkedin", "content": "Hello"}'
    choice = MagicMock()
    choice.finish_reason = "tool_calls"
    choice.message.content = None
    choice.message.tool_calls = [tc]

    messages = _from_litellm(choice)
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert len(messages[0].tool_calls) == 1
    assert messages[0].tool_calls[0]["name"] == "save_post"
    assert messages[0].tool_calls[0]["id"] == "tc_1"
    assert messages[0].tool_calls[0]["args"] == {"platform": "linkedin", "content": "Hello"}


def test_from_litellm_extracts_text_reply():
    from agent.graph import _from_litellm
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = "Here is my response."

    messages = _from_litellm(choice)
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert messages[0].content == "Here is my response."
    assert not messages[0].tool_calls


def test_should_continue_returns_tools_when_tool_calls_present():
    from agent.graph import should_continue
    state = {
        "messages": [AIMessage(
            content="",
            tool_calls=[{"id": "tc_1", "name": "save_post", "args": {}, "type": "tool_call"}],
        )],
        "org_id": "test-org",
    }
    assert should_continue(state) == "tools"


def test_should_continue_returns_end_for_text_reply():
    from agent.graph import should_continue
    state = {
        "messages": [AIMessage(content="Here is my response.")],
        "org_id": "test-org",
    }
    assert should_continue(state) == END


def test_should_continue_returns_end_for_empty_tool_calls_list():
    from agent.graph import should_continue
    state = {
        "messages": [AIMessage(content="ok", tool_calls=[])],
        "org_id": "test-org",
    }
    assert should_continue(state) == END


async def _text_stream(text: str):
    """Async generator that yields one text chunk then a stop-reason sentinel."""
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=text, tool_calls=None), finish_reason=None)]
    yield chunk
    final = MagicMock()
    final.choices = [MagicMock(delta=MagicMock(content=None, tool_calls=None), finish_reason="stop")]
    yield final


async def test_call_model_applies_max_turns_sliding_window(tmp_data_dir):
    from agent.graph import _build_call_model, MAX_TURNS
    from agent.memory import save_profile

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": None, "preferences": {}, "post_ledger": [],
    })
    # Build a history longer than MAX_TURNS
    messages = []
    for i in range(MAX_TURNS + 5):
        messages.append(HumanMessage(content=f"msg {i}"))
        messages.append(AIMessage(content=f"reply {i}"))

    mock_completion = AsyncMock(return_value=_text_stream("ok"))

    with (
        patch("agent.graph.litellm.acompletion", mock_completion),
        patch("agent.graph.get_stream_writer", return_value=lambda x: None),
    ):
        call_model = _build_call_model("gpt-4o-mini")
        await call_model({"messages": messages, "org_id": "org"})

    sent = mock_completion.call_args.kwargs["messages"]
    # First entry is always the system message; the rest must be ≤ MAX_TURNS.
    assert sent[0]["role"] == "system"
    assert len(sent) - 1 <= MAX_TURNS


async def test_prompt_caching_adds_cache_control_for_anthropic_model(tmp_data_dir):
    from agent.graph import _build_call_model
    from agent.memory import save_profile

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": None, "preferences": {}, "post_ledger": [],
    })
    mock_completion = AsyncMock(return_value=_text_stream("hello"))

    with (
        patch("agent.graph.litellm.acompletion", mock_completion),
        patch("agent.graph.get_stream_writer", return_value=lambda x: None),
    ):
        call_model = _build_call_model("claude-haiku-4-5-20251001")
        await call_model({"messages": [HumanMessage(content="test")], "org_id": "org"})

    messages = mock_completion.call_args.kwargs["messages"]
    system_content = messages[0]["content"]
    assert isinstance(system_content, list)
    assert system_content[0]["type"] == "text"
    assert system_content[0]["cache_control"] == {"type": "ephemeral"}
    assert mock_completion.call_args.kwargs["extra_headers"] == {
        "anthropic-beta": "prompt-caching-2024-07-31"
    }


async def test_prompt_caching_skipped_for_non_anthropic_model(tmp_data_dir):
    from agent.graph import _build_call_model
    from agent.memory import save_profile

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": None, "preferences": {}, "post_ledger": [],
    })
    mock_completion = AsyncMock(return_value=_text_stream("hello"))

    with (
        patch("agent.graph.litellm.acompletion", mock_completion),
        patch("agent.graph.get_stream_writer", return_value=lambda x: None),
    ):
        call_model = _build_call_model("gpt-4o-mini")
        await call_model({"messages": [HumanMessage(content="test")], "org_id": "org"})

    messages = mock_completion.call_args.kwargs["messages"]
    system_content = messages[0]["content"]
    assert isinstance(system_content, str)
    assert mock_completion.call_args.kwargs["extra_headers"] is None


async def test_prompt_caching_with_anthropic_provider_prefix(tmp_data_dir):
    """anthropic/claude-* (LiteLLM provider prefix) must also trigger caching headers."""
    from agent.graph import _build_call_model
    from agent.memory import save_profile

    save_profile("org", {
        "org_id": "org", "name": "Test Org", "website": "",
        "research": None, "preferences": {}, "post_ledger": [],
    })
    mock_completion = AsyncMock(return_value=_text_stream("hello"))

    with (
        patch("agent.graph.litellm.acompletion", mock_completion),
        patch("agent.graph.get_stream_writer", return_value=lambda x: None),
    ):
        call_model = _build_call_model("anthropic/claude-haiku-4-5-20251001")
        await call_model({"messages": [HumanMessage(content="test")], "org_id": "org"})

    messages = mock_completion.call_args.kwargs["messages"]
    system_content = messages[0]["content"]
    assert isinstance(system_content, list)
    assert system_content[0]["cache_control"] == {"type": "ephemeral"}
    assert mock_completion.call_args.kwargs["extra_headers"] == {
        "anthropic-beta": "prompt-caching-2024-07-31"
    }
