"""
Chainlit web UI entry point for Mackie.

Owns: session initialization, org identification prompt, and message routing.
Does not own conversation logic (assistant.py) or persistence (memory.py).
"""
from __future__ import annotations

import uuid
from pathlib import Path

import chainlit as cl
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from agent import memory as mem
from agent.assistant import Assistant


@cl.on_chat_start
async def start() -> None:
    """Initialize a new Chainlit session: ask for org name and URL, then create the Assistant."""
    res = await cl.AskUserMessage(
        content=(
            "Hi! I'm **Mackie**, your NPO social media assistant. "
            "What organization are you working with?\n\n"
            "Share the **name** and **website URL** (optional) — one per line or separated by a comma."
        ),
        timeout=120,
    ).send()

    if not res:
        await cl.Message(content="Session timed out. Please refresh to start again.").send()
        return

    text = res["output"].strip()

    if "," in text and "\n" not in text:
        parts = text.split(",", 1)
        org_name, org_url = parts[0].strip(), parts[1].strip()
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        org_name = lines[0]
        org_url = lines[1] if len(lines) > 1 else ""

    org_id = mem.slugify(org_name)
    profile = mem.load_profile(org_id)
    if not profile.get("name"):
        profile["name"] = org_name
        profile["website"] = org_url
        mem.save_profile(org_id, profile)

    # Each browser session gets its own thread_id so concurrent users on the
    # same org don't share or corrupt each other's conversation history.
    thread_id = f"{org_id}-{uuid.uuid4().hex[:8]}"
    cl.user_session.set("assistant", Assistant(org_id, thread_id=thread_id))

    await cl.Message(
        content=f"Got it! I'm set up for **{org_name}**. What would you like to work on?"
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Forward each user message to the Assistant, streaming tool steps and text in real time."""
    assistant: Assistant | None = cl.user_session.get("assistant")
    if not assistant:
        await cl.Message(content="Please refresh the page to start a new session.").send()
        return

    msg = cl.Message(content="")
    await msg.send()

    async for event in assistant.achat_stream(message.content):
        if event["type"] == "text":
            await msg.stream_token(event["content"])
        elif event["type"] == "tool_step":
            async with cl.Step(name=event["tool"]) as step:
                step.input = event["input_preview"]
                step.output = f"{event['result_preview']}  ({event['latency_ms']}ms)"

    await msg.update()
