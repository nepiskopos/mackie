"""
Tool definitions and handlers for Mackie's agentic loop.

Owns: the TOOL_DEFINITIONS list (OpenAI function-calling format consumed by
LiteLLM), handle_tool_call dispatch, per-tool trace logging to
data/{org_id}/trace.jsonl, and the async calendar generation helper.
Does not own the conversation loop (assistant.py) or low-level memory I/O
(memory.py).
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import litellm

from . import resolve_model, memory as mem
from . import research as res


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "research_org",
            "description": (
                "Research the organization before generating any content. "
                "Scrapes their website and searches the web for their social presence, "
                "recent news, and what similar orgs are doing on social media. "
                "ALWAYS call this first if org research has not been done yet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "org_name": {"type": "string"},
                    "org_url": {
                        "type": "string",
                        "description": "Organization website URL (optional)",
                    },
                },
                "required": ["org_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refresh_research",
            "description": (
                "Re-run research on the org to capture new programs, news, or campaigns. "
                "Preserves all learned preferences and the post ledger — only updates research data. "
                "Call when the user says 'update your research', 'we just launched X', or mentions recent changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "org_name": {"type": "string"},
                    "org_url": {
                        "type": "string",
                        "description": "Organization website URL (optional)",
                    },
                },
                "required": ["org_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for trends, news, or topics relevant to the org's content strategy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_post",
            "description": (
                "Save a post to the ledger. Call this every time you suggest or draft content — "
                "use status='suggested' for initial ideas and status='draft' for fully written posts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "description": "e.g. linkedin, instagram, facebook, general",
                    },
                    "content": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["suggested", "draft", "approved", "posted", "planned"],
                        "default": "suggested",
                    },
                    "source_summary": {
                        "type": "string",
                        "description": "What research or input informed this post",
                    },
                },
                "required": ["platform", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_post_status",
            "description": "Update the status of an existing post in the ledger.",
            "parameters": {
                "type": "object",
                "properties": {
                    "post_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["suggested", "draft", "approved", "posted", "planned"],
                    },
                },
                "required": ["post_id", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_preference",
            "description": (
                "Persist a learned brand preference or guideline for this org. "
                "Call this immediately when the user corrects tone, voice, or expresses any content preference. "
                "Examples of keys: 'voice', 'banned_topics', 'content_pillars', 'tone_notes'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_calendar",
            "description": (
                "Generate a 4-week content calendar for a given month. "
                "Plans one post per week (platform, angle, content pillar, rationale) grounded in org research, "
                "saves all four to the ledger with status 'planned', and returns a markdown summary table. "
                "Call when the user asks to plan a month's content or build a content calendar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "month_year": {
                        "type": "string",
                        "description": "Month and year to plan, e.g. 'July 2026'",
                    },
                },
                "required": ["month_year"],
            },
        },
    },
]


def _format_research_summary(org_name: str, result: dict[str, Any]) -> str:
    """
    Format a research result dict into a readable summary string for the LLM.

    All third-party content (website text, search snippets) is wrapped in
    <external_content> tags. This creates an explicit trust boundary so the model
    can distinguish legitimate instructions (system prompt + user messages) from
    untrusted third-party data, mitigating indirect prompt injection via malicious
    org websites or crafted search snippets.

    Args:
        org_name: Display name of the org, shown in the header line.
        result:   Dict returned by research_org(), with keys website_content,
                  social_presence, news, similar_orgs, sources.

    Returns:
        Multi-line summary with website content truncated to 3000 chars,
        social results limited to 3 entries, error entries filtered out,
        and all third-party text enclosed in <external_content> tags.
    """
    lines = [
        f"Research complete for {org_name}.",
        f"Sources collected: {len(result.get('sources', []))}",
        "",
    ]
    if result.get("website_content"):
        lines += [
            "Website content:",
            '<external_content source="website">',
            result["website_content"][:3000],
            "</external_content>",
            "",
        ]
    if result.get("social_presence"):
        lines.append("Social presence:")
        for r in result["social_presence"][:3]:
            if "error" not in r:
                lines.append(f'<external_content source="web_search" url="{r.get("url","")}">')
                lines.append(f'{r.get("title","")}: {r.get("snippet","")[:200]}')
                lines.append("</external_content>")
    if result.get("news"):
        lines.append("\nRecent news:")
        for r in result["news"][:4]:
            if "error" not in r:
                lines.append(f'<external_content source="web_search" url="{r.get("url","")}">')
                lines.append(f'{r.get("title","")}: {r.get("snippet","")[:200]}')
                lines.append("</external_content>")
    if result.get("similar_orgs"):
        lines.append("\nSimilar orgs on social:")
        for r in result["similar_orgs"][:2]:
            if "error" not in r:
                lines.append('<external_content source="web_search">')
                lines.append(f'{r.get("title","")}: {r.get("snippet","")[:200]}')
                lines.append("</external_content>")
    return "\n".join(lines)


async def handle_tool_call(tool_name: str, tool_input: dict[str, Any], org_id: str) -> str:
    """
    Dispatch a tool call by name, write a trace entry, and return the result.

    Args:
        tool_name:  Name matching one of the entries in TOOL_DEFINITIONS.
        tool_input: Parsed arguments dict from the LLM's tool call.
        org_id:     The active org identifier for memory operations.

    Returns:
        A plain-text result string to be sent back to the LLM as a tool result.
    """
    t0 = time.time()
    result = await _dispatch(tool_name, tool_input, org_id)
    _append_trace(org_id, tool_name, tool_input, result, round((time.time() - t0) * 1000))
    return result


async def _run_research(
    org_name: str, org_url: str, org_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Run org research and return (result, profile) with profile["research"] pre-set.

    The caller is responsible for any further profile mutations and for calling
    mem.save_profile() when done.
    """
    result = await res.research_org(org_name=org_name, org_url=org_url)
    profile = mem.load_profile(org_id)
    profile["research"] = result
    return result, profile


async def _dispatch(tool_name: str, tool_input: dict[str, Any], org_id: str) -> str:
    """Execute the named tool and return its result string."""
    if tool_name == "research_org":
        result, profile = await _run_research(
            tool_input["org_name"], tool_input.get("org_url", ""), org_id
        )
        if not profile["name"]:
            profile["name"] = tool_input["org_name"]
        if not profile["website"] and tool_input.get("org_url"):
            profile["website"] = tool_input["org_url"]
        mem.save_profile(org_id, profile)
        return _format_research_summary(tool_input["org_name"], result)

    elif tool_name == "refresh_research":
        result, profile = await _run_research(
            tool_input["org_name"], tool_input.get("org_url", ""), org_id
        )
        profile["research_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        mem.save_profile(org_id, profile)
        return (
            f"Research refreshed for {tool_input['org_name']}. "
            f"Sources: {len(result.get('sources', []))}. "
            "Preferences and post ledger preserved."
        )

    elif tool_name == "web_search":
        results = await res.search_web(tool_input["query"], tool_input.get("max_results", 5))
        lines = [
            f"[{r.get('url','')}] {r.get('title','')}: {r.get('snippet','')[:300]}"
            for r in results if "error" not in r
        ]
        return "\n".join(lines) if lines else "No results found."

    elif tool_name == "save_post":
        status = tool_input.get("status", "suggested")
        post_id = mem.add_post(
            org_id,
            {
                "platform": tool_input["platform"],
                "content": tool_input["content"],
                "status": status,
                "source_summary": tool_input.get("source_summary", ""),
            },
        )
        return f"Saved to ledger as {post_id} (status: {status})"

    elif tool_name == "update_post_status":
        ok = mem.update_post_status(org_id, tool_input["post_id"], tool_input["status"])
        if ok:
            return f"Updated {tool_input['post_id']} → {tool_input['status']}"
        return f"Post {tool_input['post_id']} not found in ledger"

    elif tool_name == "save_preference":
        mem.save_preference(org_id, tool_input["key"], tool_input["value"])
        return f"Preference saved: {tool_input['key']} = {tool_input['value']}"

    elif tool_name == "generate_calendar":
        return await _generate_calendar_async(org_id, tool_input["month_year"])

    return f"Unknown tool: {tool_name}"


async def _generate_calendar_async(org_id: str, month_year: str) -> str:
    """
    Generate a 4-week content calendar by running one LLM call per week in parallel.

    Four plan_week() coroutines run concurrently via asyncio.gather with
    return_exceptions=True so a single week failure does not abort the whole
    calendar. Failed weeks appear as error rows in the table; successful weeks
    are saved to the ledger with status 'planned'.

    Args:
        org_id:     The active org identifier.
        month_year: Target month string, e.g. "July 2026".

    Returns:
        A markdown table summarising the four-week plan, with a footer line
        showing how many weeks were saved (e.g. "3 of 4 weeks saved").
    """
    profile = mem.load_profile(org_id)
    org_name = profile.get("name", "this organization")
    model = resolve_model()

    research = profile.get("research") or {}
    research_context = (
        research.get("website_content", "")[:800]
        + " ".join(r.get("snippet", "") for r in research.get("news", [])[:3])
    ).strip() or "No research available yet."

    prefs = profile.get("preferences", {})
    prefs_context = (
        "\n".join(f"- {k}: {v}" for k, v in prefs.items()) if prefs else "No preferences recorded."
    )

    existing_snippets = [p["content"][:60] for p in profile.get("post_ledger", [])[-10:]]

    async def plan_week(week_label: str) -> dict:
        prompt = (
            f"You are a social media strategist for {org_name}.\n"
            f"Plan ONE post for {week_label} of {month_year}.\n\n"
            f"Research context:\n{research_context}\n\n"
            f"Brand preferences:\n{prefs_context}\n\n"
            "Respond ONLY with a JSON object (no markdown fences) with these exact keys:\n"
            '  "platform": one of "linkedin", "instagram", "facebook"\n'
            '  "angle": one sentence describing the post idea\n'
            '  "pillar": one of "Community Impact", "Programs & Services", '
            '"Awareness & Education", "Events & Campaigns", "Behind the Scenes"\n'
            '  "rationale": one sentence citing the source or reasoning\n\n'
            + (
                "Avoid repeating these existing ideas:\n"
                + "\n".join(f"- {s}" for s in existing_snippets)
                if existing_snippets
                else ""
            )
        )
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Model may have wrapped the JSON in markdown fences or added prose;
            # extract the first {...} block as a fallback.
            match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
            try:
                data = json.loads(match.group(0)) if match else {}
            except (json.JSONDecodeError, AttributeError):
                data = {}
        if not data.get("angle"):
            data = {
                "platform": "general",
                "angle": text[:120],
                "pillar": "Community Impact",
                "rationale": "Generated from org research",
            }
        return {"week": week_label, **data}

    weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
    results = await asyncio.gather(*[plan_week(w) for w in weeks], return_exceptions=True)

    lines = [
        f"## Content Calendar — {month_year}\n",
        "| Week | Platform | Angle | Pillar |",
        "|---|---|---|---|",
    ]
    saved = 0
    for item in results:
        if isinstance(item, Exception):
            lines.append("| — | — | Error generating plan | — |")
            continue
        content = (
            f"[{item['week']} of {month_year}] [{item.get('pillar', '')}] {item.get('angle', '')}"
        )
        mem.add_post(org_id, {
            "platform": item.get("platform", "general"),
            "content": content,
            "status": "planned",
            "source_summary": item.get("rationale", ""),
        })
        saved += 1
        lines.append(
            f"| {item['week']} | {item.get('platform', '').capitalize()} "
            f"| {item.get('angle', '')} | {item.get('pillar', '')} |"
        )

    lines.append(f"\n{saved} of 4 weeks saved to the ledger with status 'planned'.")
    return "\n".join(lines)


def _append_trace(
    org_id: str, tool_name: str, tool_input: dict[str, Any], result: str, latency_ms: int
) -> None:
    """Append one JSONL entry to data/{org_id}/trace.jsonl."""
    entry = {
        "ts": time.time(),
        "tool": tool_name,
        "input": tool_input,
        "result_preview": result[:300],
        "latency_ms": latency_ms,
    }
    path = mem.trace_path(org_id)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
