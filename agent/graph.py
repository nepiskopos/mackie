"""
LangGraph agentic graph for Mackie.

Owns: the two-node (agent → tools) graph definition, message format conversion
between LangChain BaseMessage objects and LiteLLM-compatible dicts, async SQLite
conversation checkpointing via AsyncSqliteSaver, and the sliding window applied
before each LLM call. Does not own system prompt construction (assistant.py),
tool logic (tools.py), or memory I/O (memory.py).

The graph is compiled fresh per chat() call (cheap operation). Conversation
history is persisted in data/checkpoints.db, keyed by the thread_id in the
invoke config, so turns survive container restarts.

Both async nodes emit custom stream events via get_stream_writer():
  - tools node: {"type": "tool_step", "tool": ..., "input_preview": ...,
                  "result_preview": ..., "latency_ms": ...}
  - agent node: {"type": "text", "content": token}  — emitted once per chunk
                  because the LLM is called with stream=True; Chainlit calls
                  msg.stream_token() on each event so the reply renders
                  word-by-word as the model generates.
These are consumed by achat_stream() in assistant.py for real-time Chainlit display.
"""
from __future__ import annotations

import json
import operator
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, TypedDict

import litellm
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph

from .config import is_anthropic_model
from .tools import TOOL_DEFINITIONS, handle_tool_call

MAX_TURNS: int = 40
CHECKPOINTS_DB: Path = Path(__file__).parent.parent / "data" / "checkpoints.db"


class AgentState(TypedDict):
    """State accumulated across nodes within a single conversation thread."""

    messages: Annotated[list[BaseMessage], operator.add]
    org_id: str


def _to_litellm(messages: list[BaseMessage]) -> list[dict]:
    """
    Convert LangChain message objects to LiteLLM-compatible dicts.

    Args:
        messages: List of HumanMessage, AIMessage, or ToolMessage objects.

    Returns:
        List of role-keyed dicts accepted by litellm.acompletion().
    """
    result: list[dict] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                result.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
            else:
                result.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "name": msg.name,
                "content": msg.content,
            })
    return result


def _from_litellm(choice: Any) -> list[BaseMessage]:
    """
    Convert a non-streaming LiteLLM response choice to a list of LangChain messages.

    Not called by call_model at runtime (which now uses stream=True and builds
    AIMessage directly from accumulated chunks). Retained because test_graph.py
    exercises it directly and it remains a useful conversion utility.

    Args:
        choice: response.choices[0] from a non-streaming litellm.acompletion() call.

    Returns:
        A single-element list containing the appropriate BaseMessage subtype.
    """
    msg = choice.message
    if choice.finish_reason == "tool_calls":
        tool_calls = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments),
                "type": "tool_call",
            }
            for tc in msg.tool_calls
        ]
        return [AIMessage(content=msg.content or "", tool_calls=tool_calls)]
    return [AIMessage(content=msg.content or "")]


def _build_call_model(model: str) -> Callable[[AgentState], dict]:
    """
    Return an async call_model node function bound to the given model string.

    Applies the sliding window (MAX_TURNS) before calling the LLM so unbounded
    conversation history in the checkpoint does not overflow the context window.
    For Anthropic models (model starts with "claude" or "anthropic/"), wraps the
    system prompt in a content block with cache_control=ephemeral — this caches
    the prompt across turns for ~60% token cost reduction. Non-Anthropic providers
    receive a plain string system message, preserving full provider portability.
    Uses stream=True so each text token is emitted via the stream writer as it
    arrives, giving Chainlit a word-by-word display instead of a single pop-in.
    Tool call chunks are accumulated silently; text chunks each trigger a
    {"type": "text", "content": token} writer event.
    Side-effect: any text the model emits before invoking tools (e.g. "Let me
    research that…") is now visible in the UI rather than being silently
    dropped as it was in non-streaming mode. This is intentional — it gives
    users a real-time signal that the model is about to run a tool.
    """
    is_anthropic: bool = is_anthropic_model(model)

    async def call_model(state: AgentState) -> dict:
        # Lazy import breaks the circular dependency: graph → assistant → graph
        from . import assistant as _asst
        system_prompt = _asst._build_system_prompt(state["org_id"])
        recent = state["messages"][-MAX_TURNS:]
        system_content: list[dict] | str = (
            [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
            if is_anthropic else system_prompt
        )
        writer = get_stream_writer()
        text_parts: list[str] = []
        # tool_calls_raw maps chunk index → accumulated tool call fields
        tool_calls_raw: dict[int, dict] = {}

        stream = await litellm.acompletion(
            model=model,
            messages=[{"role": "system", "content": system_content}] + _to_litellm(recent),
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            max_tokens=4096,
            num_retries=3,
            stream=True,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"} if is_anthropic else None,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                text_parts.append(delta.content)
                writer({"type": "text", "content": delta.content})
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_raw:
                        tool_calls_raw[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                    if tc_delta.id:
                        tool_calls_raw[idx]["id"] = tc_delta.id
                    if tc_delta.function.name:
                        # name arrives fully in the first chunk; no concatenation needed
                        tool_calls_raw[idx]["function"]["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls_raw[idx]["function"]["arguments"] += tc_delta.function.arguments

        full_text = "".join(text_parts)
        if tool_calls_raw:
            tool_calls = [
                {
                    "id": tool_calls_raw[idx]["id"],
                    "name": tool_calls_raw[idx]["function"]["name"],
                    "args": json.loads(tool_calls_raw[idx]["function"]["arguments"]),
                    "type": "tool_call",
                }
                for idx in sorted(tool_calls_raw)
            ]
            return {"messages": [AIMessage(content=full_text, tool_calls=tool_calls)]}
        return {"messages": [AIMessage(content=full_text)]}
    return call_model


def _build_call_tools(step_log: list[dict] | None) -> Callable[[AgentState], dict]:
    """
    Return an async call_tools node that dispatches all tool calls in the last message.

    Emits {"type": "tool_step", ...} to the stream writer as each tool completes,
    enabling real-time display in Chainlit. Also appends to step_log if provided.
    """
    async def call_tools(state: AgentState) -> dict:
        last: AIMessage = state["messages"][-1]
        tool_messages: list[BaseMessage] = []
        writer = get_stream_writer()
        for tc in last.tool_calls:
            t0 = time.time()
            result = await handle_tool_call(tc["name"], tc["args"], state["org_id"])
            entry = {
                "tool": tc["name"],
                "input_preview": json.dumps(tc["args"])[:300],
                "result_preview": str(result)[:500],
                "latency_ms": round((time.time() - t0) * 1000),
            }
            writer({"type": "tool_step", **entry})
            if step_log is not None:
                step_log.append(entry)
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])
            )
        return {"messages": tool_messages}
    return call_tools


def should_continue(state: AgentState) -> str:
    """Route to 'tools' if the last message has tool calls, otherwise end the turn."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


@asynccontextmanager
async def build_graph(
    model: str, step_log: list[dict] | None = None
) -> AsyncIterator[Any]:
    """
    Build and yield a compiled LangGraph graph with async SQLite checkpointing.

    The graph has two async nodes — agent (calls LLM) and tools (dispatches tool
    calls) — connected in a loop until the LLM returns a text reply. Conversation
    history is persisted in CHECKPOINTS_DB keyed by the thread_id in the invoke
    config. Use astream(stream_mode="custom") to receive real-time tool and text
    events; use ainvoke() to get only the final state.

    Args:
        model:    LiteLLM model string, e.g. "claude-haiku-4-5-20251001".
        step_log: Optional list to also append tool trace entries to (for tests).

    Yields:
        A compiled CompiledGraph ready for ainvoke() and astream() calls.

    Example:
        config = {"configurable": {"thread_id": thread_id}}
        async with build_graph(model) as graph:
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=msg)], "org_id": org_id}, config
            )
        reply = result["messages"][-1].content
    """
    CHECKPOINTS_DB.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINTS_DB)) as checkpointer:
        builder: StateGraph = StateGraph(AgentState)
        builder.add_node("agent", _build_call_model(model))
        builder.add_node("tools", _build_call_tools(step_log))
        builder.set_entry_point("agent")
        builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")
        yield builder.compile(checkpointer=checkpointer)
