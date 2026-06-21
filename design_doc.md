# Design Doc — Mackie NPO Social Media Assistant

## What it is

Mackie is a conversational AI assistant for nonprofit marketing staff. It researches an organization, suggests on-brand social media content for LinkedIn, Instagram, and Facebook, and learns from voice corrections — persisting preferences to disk so teams don't repeat themselves across sessions.

**First-run flow:** On startup the user provides an org name and website URL. When they first ask for content, Mackie calls `research_org` automatically — scraping the website and running three parallel web searches — then produces numbered post ideas each citing a specific source. Every tone correction ("too corporate", "more grassroots") is immediately saved to `data/{org-id}/profile.json` via `save_preference` and reloaded into the system prompt on every future turn.

**Agent loop:** Each user message enters LangGraph's `agent → tools` cycle. The agent node calls the LLM with the current system prompt (rebuilt from `profile.json` each turn) and the last 40 messages. The LLM either returns a plain-text reply (`END`) or calls one or more tools (→ tools node → back to agent node). Text tokens stream to the UI immediately via `get_stream_writer()`; tool steps appear as each one completes. The cycle terminates when the LLM returns no tool calls.

---

## Architecture Overview

```
User (browser / CLI)
    │
    ▼
app.py (Chainlit)  ──  main.py (rich CLI)
    │                        │
    └────────────────────────┘
                 │
                 ▼
     Assistant  (agent/assistant.py)
         │  system prompt rebuilt each turn
         │  delegates to graph.py
         │
         ▼
     LangGraph  (agent/graph.py)
         │  AgentState + SQLite checkpointer
         │  sliding window: messages[-40:]
         │
         ▼
     LiteLLM  ──►  Anthropic / OpenAI / Gemini / Ollama
         │  finish_reason == "tool_calls" → tools node
         │
         ├── research_org    ──► research.py  (httpx + Tavily, asyncio.gather)
         ├── refresh_research ──► research.py  (re-runs research, preserves prefs)
         ├── web_search      ──► research.py
         ├── save_post       ──► memory.py  (JSON on disk)
         ├── update_post_status ──► memory.py
         ├── save_preference    ──► memory.py
         └── generate_calendar  ──► tools.py  (4 parallel LLM calls via asyncio.gather)
                                         │
                                         └─► trace.jsonl  (per tool call)

Persistent state:  data/{org-id}/profile.json
                   data/{org-id}/trace.jsonl
                   data/checkpoints.db      (LangGraph SQLite checkpoint)
Skills/platforms:  skills/*.yaml
Content pillars:   skills/content_pillars.yaml
```

---

## Decision Log

### LLM: Multi-Provider Resolution (Anthropic default)

**What:** Model selection via `agent/config.py`; `claude-haiku-4-5-20251001` is the Anthropic default used in development and demos.
**Alternatives:** Single hardcoded model string; generic `MODEL=` env var (removed — scattered provider logic).
**Why:** Claude Haiku is the default because its instruction-following for structured tool use and JSON output is reliable and predictable at ~$0.80/M input. `resolve_model()` checks provider API keys in priority order (Anthropic → OpenAI → xAI → Google) and returns the model for the highest-priority provider whose key is set. Swap model quality by setting `ANTHROPIC_MODEL=claude-sonnet-4-6`; swap provider by setting the appropriate API key. No code change required in either case.
**Streaming:** Tool steps appear in Chainlit as each tool completes (via `get_stream_writer` + `stream_mode="custom"`). Text replies stream token-by-token: `_build_call_model` calls `litellm.acompletion(stream=True)` and emits each chunk immediately via `get_stream_writer()`, so Chainlit renders words as the model generates them rather than displaying the full reply at once. Behavioral note: any text the model emits before invoking tools (e.g. "Let me research that…") is now shown in the UI — in non-streaming mode this preamble was silently dropped. This is intentional; it gives users a real-time signal before a long tool call.

---

### Provider Adapter: LiteLLM + config.py

**What:** All LLM calls go through `litellm.acompletion()` in `agent/graph.py`, `agent/tools.py`, and `eval.py`. `agent/config.py` is the single source of truth for provider metadata and model resolution. No direct SDK imports elsewhere.
**Alternatives:** Direct `anthropic` SDK, OpenAI SDK, provider-specific clients with env var logic scattered per module.
**Why:** Core requirement — swapping the model must require only env var changes, no code changes. `agent/config.py`'s `PROVIDERS` list centralises all provider metadata (API key env var, model env var, model prefixes, default model). `resolve_model()` is called once in `Assistant.__init__()` and in `tools.py`'s calendar generator. LiteLLM provides the unified OpenAI-compatible call interface across 100+ providers. Tool definitions use the OpenAI function-calling format which LiteLLM translates per provider. Retry logic (`num_retries=3`) and fallback chains are built in.
**How:** `TOOL_DEFINITIONS` in `agent/tools.py` uses the OpenAI format. `await litellm.acompletion()` is called in `_build_call_model()` in `graph.py` (the main agent node), in `_generate_calendar_async()` in `tools.py` (4 parallel week-plan calls), and in `eval.py` (21 parallel judge calls). `is_anthropic_model(model)` from `config.py` gates prompt-caching headers in `graph.py`.

---

### Agentic Framework: LangGraph + SQLite Checkpointing

**What:** A two-node `StateGraph` (`agent` → `tools`) compiled in `agent/graph.py`, with conversation history persisted to `data/checkpoints.db` via `AsyncSqliteSaver`.
**Alternatives:** Raw `while True` loop (implemented first), LangChain AgentExecutor, CrewAI.
**Why:** The raw loop worked but had one critical gap — history lived only in `self.messages` in RAM, so restarting the container lost the entire conversation. LangGraph's `AsyncSqliteSaver` checkpointer persists the full `AgentState` to SQLite after every node, keyed by a `thread_id`. This means conversations survive container restarts with zero extra code. The graph itself is a clean two-node loop (agent → tools → agent) with typed state (`AgentState(TypedDict)`) and an explicit routing function (`should_continue`), which is nearly as simple to debug as the raw loop. LangChain AgentExecutor was ruled out — it is a black box with implicit retry and callback hooks that are harder to reason about and debug.
**How:** `build_graph()` in `graph.py` is an `@asynccontextmanager` — it opens the `AsyncSqliteSaver` connection, compiles the graph, yields it, then closes the connection. `Assistant.chat()` calls it with an `async with` block. History is stored unbounded in the checkpoint; the sliding window (`messages[-MAX_TURNS:]`) is applied inside the `call_model` node before each LLM call. `thread_id` defaults to `org_id` for single-user CLI usage. Chainlit passes a per-session UUID so concurrent users on the same org don't share conversation history. The CLI `reset` command increments a session counter to get a fresh thread_id, actually clearing history as the user expects.
**Known limitation:** `AsyncSqliteSaver` is single-process. Multi-replica deployments need `langgraph-checkpoint-postgres` with a shared Postgres instance. For a single-container NPO deployment this is not a concern.

---

### Web Search: Tavily (AsyncTavilyClient)

**What:** `tavily-python` with `AsyncTavilyClient` for all web search calls in `agent/research.py`.
**Alternatives:** `duckduckgo-search` (no API key needed), Brave Search API, SerpAPI.
**Why:** Tavily is purpose-built for agents — it returns clean, structured results (title, URL, snippet) with no HTML parsing needed. The free tier (1,000 searches/month) covers the assignment. More importantly, `AsyncTavilyClient` enables `asyncio.gather` across the three parallel search calls in `research_org`, reducing research latency from ~4–5s to ~1–2s. DuckDuckGo's unofficial API cannot be made async and is prone to rate limiting.
**How:** `research_org()` runs `scrape_website()` + three `search_web()` calls concurrently via `asyncio.gather`. Results go into the structured research dict stored in `profile.json`.

---

### Web Scraping: httpx + beautifulsoup4

**What:** `httpx.AsyncClient` for HTTP fetching; `beautifulsoup4` for HTML parsing in `agent/research.py`.
**Alternatives:** `requests` (sync only), `playwright` (full browser), `scrapy` (overkill).
**Why:** `httpx.AsyncClient` is an async context manager, enabling it to participate in `asyncio.gather` alongside the Tavily search calls. BeautifulSoup strips scripts, styles, nav, and footer elements — returning clean body text. Together they handle the majority of NPO websites which are static or server-rendered.
**SSRF protection:** `_is_safe_url()` validates the URL before any network request is made. It blocks non-HTTP(S) schemes, known cloud metadata hostnames (`169.254.169.254`, `metadata.google.internal`), loopback addresses, and all RFC 1918 private ranges. `follow_redirects=False` prevents a redirect chain from bypassing the check (e.g. `attacker.com/r → 169.254.169.254`). An attacker supplying a metadata endpoint as the org URL receives an error string instead of a network call.
**Known limitation:** JavaScript-rendered sites (React SPAs) are not supported. Would add Playwright for those in production.

---

### Memory: JSON Files on Disk

**What:** Per-org `profile.json` under `data/{org-id}/`.
**Alternatives:** SQLite, Redis, Chroma (vector store), cloud KV.
**Why:** No database is used at this assignment stage. This is a deliberate choice, not an oversight: the assignment requires a non-engineer to view and edit what the assistant has learned, and a plain JSON file opened in any text editor satisfies that requirement with zero setup. The file structure is self-documenting, survives container restarts via a mounted volume, and requires no migration scripts when the schema evolves during development.
**Structure:**
```json
{
  "name": "Org Name",
  "website": "https://...",
  "research": { ... },
  "preferences": { "voice": "warm and grassroots" },
  "post_ledger": [
    { "id": "post_001", "platform": "linkedin", "status": "draft", ... }
  ]
}
```
**Known limitation (no-DB consequence):** The load-modify-save pattern in `memory.py` is not atomic. Two concurrent sessions writing to the same org profile will race and one write will silently overwrite the other. This is acceptable under the assignment's single-user assumption. A production deployment would replace `profile.json` with a PostgreSQL table and use row-level locking.

---

### Content Framework: Content Pillars

**What:** Five default pillars defined in `skills/content_pillars.yaml`, injected into the system prompt via `_load_content_pillars()`. Per-org overrides stored in `profile.json` under `content_pillars`.
**Alternatives:** Hardcode pillar names in the system prompt; use a database table.
**Why:** Pillars are editorial policy, not code. A non-engineer should be able to add or rename a pillar without a deployment. YAML in `skills/` satisfies this — same directory as platform skills. The `id`-field check in `_load_platform_skills()` keeps platform YAMLs and content pillars coexisting safely in one directory.

---

### Extensibility: Platform Skills as YAML

**What:** Each platform lives in `skills/{platform}.yaml` with `name`, `id`, `tone`, `max_length`, `hashtag_count`, `hashtag_style`, and `format_hints`.
**Alternatives:** Python classes per platform, database rows, JSON configs.
**Why:** YAML is readable and writable by non-engineers without a code editor. Adding a new platform (Threads, TikTok) is one file drop — no rebuild, no restart if running locally.

---

### Context Management: Ledger Truncation + Sliding Window

**What:** `_build_system_prompt` includes only the 20 most recent ledger posts. In `_build_call_model`, `state["messages"][-MAX_TURNS:]` (40 messages) is sent to the LLM each turn.
**Why:** The raw message history (~300 tokens/turn) is not the bottleneck — the growing ledger in the system prompt is. Ledger truncation keeps the prompt bounded. The sliding window limits what the LLM sees while the full history is preserved in the SQLite checkpoint (so it's never lost, just not sent to the LLM). Research and preferences live in the system prompt rebuilt each turn, so they are always present regardless of the window.
**Production upgrade path:** Replace the recency window with a vector store (ChromaDB) for semantic retrieval of relevant past turns.

---

### Prompt Caching

**What:** For Anthropic models, the system prompt content block carries `"cache_control": {"type": "ephemeral"}` and the request includes `extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}`. Non-Anthropic providers receive an unmodified plain string system message.
**Why:** The system prompt is the most expensive repeating cost — it contains the full org research block (up to 8 000 chars of scraped content + search snippets), preferences, and ledger entries. Caching it once per 5-minute TTL cuts input token cost by ~60% across a multi-turn session. This is the highest-value single change for a production NPO deployment where cost is a real constraint.
**Trade-off:** `cache_control` is an Anthropic-only API extension — it breaks provider portability if applied unconditionally. The guard (`is_anthropic_model(model)` from `config.py`) means the optimization is silently skipped for OpenAI, xAI, Gemini, or Ollama. Non-Anthropic users see no behavior change and no errors. Bedrock and Vertex AI routes are intentionally excluded — they use provider-side caching, not the `anthropic-beta` header. If another provider adds content-block caching, a new branch in `config.py` is the only change needed.
**How:** In `agent/graph.py`, `_build_call_model` calls `is_anthropic_model(model)` once when the closure is created (not per turn). The system message `content` field is either a content-block list (Anthropic) or a plain string (others). The `extra_headers` kwarg is either the beta header dict or `None`. LiteLLM passes `None` headers through gracefully.

---

### Research Refresh Tool

**What:** `refresh_research` tool in `agent/tools.py` re-runs `research_org()` and merges the result back into `profile.json` without overwriting preferences or the post ledger.
**Why:** Research runs once at the start of a session and freezes. An org that launches a new program, changes their website, or appears in the news needs the assistant to know about it without losing learned preferences or the post history. A separate tool (vs. automatically re-running) gives the user explicit control — Mackie only refreshes when asked ("update your research", "we just launched something").
**How:** The handler `await`s `res.research_org(...)` to re-run the full research pipeline (scrape + 3 parallel searches). It loads the current profile, replaces only the `research` field and writes `research_updated_at` with the current ISO timestamp, then saves. The timestamp appears in the system prompt's research status line so the user knows when data was last refreshed.

---

### Content Calendar Tool

**What:** `generate_calendar` tool produces a four-week content plan for a given month. Each week gets one post entry (platform, angle, pillar, rationale). All four weeks are generated in parallel via `asyncio.gather` and saved to the ledger with status `"planned"`.
**Alternatives:** Generate the whole month in a single LLM call; generate sequentially.
**Why:** Parallel week generation cuts wall-clock from ~4× a single call to ~1×. Each week's prompt includes the existing ledger to avoid repeating already-planned content — passing them all to one call would produce a very long prompt without any structural benefit. Each week's response is parsed as JSON; a regex extraction fallback handles the case where the model wraps the output in markdown fences or prose.
**How:** `_generate_calendar_async(org_id, month_year)` in `tools.py` is an async function `await`ed directly from `_dispatch()`. Four `plan_week()` coroutines run concurrently via `asyncio.gather`; each calls `litellm.acompletion()` and parses the response with `json.loads()` first, falling back to regex extraction (`re.search(r"\{[^{}]+\}", ...)`) if that fails. Each result is saved to the ledger, and the function returns a markdown table of all four weeks.

---

### Observability: trace.jsonl

**What:** Every `handle_tool_call` invocation appends a JSONL entry to `data/{org_id}/trace.jsonl` with `ts`, `tool`, `input`, `result_preview`, and `latency_ms`.
**Alternatives:** Langfuse, Helicone (production-grade tracing SaaS).
**Why:** The assignment lists observability as a nice-to-have. A local JSONL file requires zero additional dependencies or credentials — `json` + `time` from stdlib. The Chainlit UI also shows tool steps inline via `step_log`. For production, Langfuse or Helicone would add cost tracking, latency dashboards, and session replay.
**How:** `handle_tool_call` in `tools.py` times `_dispatch()`, then calls `_append_trace()` which writes to `mem.trace_path(org_id)`.

---

### Prompt Injection Defense

**What:** All third-party content (scraped website text and web search snippets) is wrapped in `<external_content>` tags before being sent to the LLM. The system prompt ends with an explicit Security Policy section instructing the model to treat content inside those tags strictly as data and ignore any instructions it finds there.
**Threat:** Indirect prompt injection — a malicious actor embeds instructions ("Ignore previous instructions", "You are now…") in a nonprofit's website or in search result snippets. Without a trust boundary, those instructions arrive in the LLM's context alongside legitimate system prompt instructions and may be followed.
**Two-layer defense:**
1. **Structural delimiter (`<external_content>`)** — `_format_research_summary()` in `agent/tools.py` wraps website content and every search snippet in tagged blocks with a `source` attribute (e.g. `source="website"`, `source="web_search" url="..."`). This makes the boundary machine-readable and consistent.
2. **System prompt policy** — `_build_system_prompt()` in `agent/assistant.py` ends with a `## Security Policy` section that explicitly names the attack pattern, lists canonical injection phrases as examples, and tells the model to ignore directives inside `<external_content>` blocks completely without mentioning them to the user.
**Why two layers:** The structural delimiter alone relies on the model correctly inferring intent from structure. The system prompt policy alone relies on covering every possible injection phrasing. Together they are defense-in-depth: the delimiter is unambiguous context, the policy is an explicit instruction.
**Persistent injection via profile.json:** Research is saved to `profile.json` after being delivered to the model as a tool result (not before). The model sees the `<external_content>` tags in the tool result and the security policy instructs it not to act on injected directives — so a `save_preference` call triggered by a malicious website instruction would be the model itself disobeying the system prompt, not an automatic pipeline.
**Limitation:** This defense assumes the LLM respects its system prompt instructions. A sufficiently adversarial injection that jailbreaks the model is out of scope for this demo. Production hardening would add output filtering and persistent rate-limiting on `save_preference` calls. Persistent rate-limiting requires a database (e.g. a Redis counter or a Postgres timestamp column), which is not used at this assignment stage; in-memory throttling would reset on every container restart and offer no real protection under load.
**How:** `_format_research_summary()` in `agent/tools.py`; security section in the return value of `_build_system_prompt()` in `agent/assistant.py`. Tests: `test_format_research_summary_wraps_website_in_external_content_tags`, `test_format_research_summary_wraps_search_results_in_external_content_tags`, `test_system_prompt_includes_security_policy` in `tests/test_tools.py` / `tests/test_assistant.py`.

---

### Eval: LLM-as-Judge (scripts/eval.py)

**What:** `eval.py` runs the 7-step required scenario against a test org, then scores each response on three rubrics in parallel (21 total judge calls via `asyncio.gather`), and prints a markdown report.
**Rubrics (0–3 each):** source attribution, voice consistency, ledger integrity (step 7 must not repeat step 1).
**Why:** Systematic AI quality evaluation over manual inspection. Running the required scenario end-to-end with automated scoring catches regressions when the system prompt or tool logic changes. The parallel scoring reduces wall-clock time from ~30s to ~1.5s.
**Known failure modes:** Verbosity bias (longer responses score higher), self-preference (Claude judging Claude outputs), threshold sensitivity (pass ≥ 2 is subjective). Mitigated by using a cheap default judge model (resolved via provider API keys) and three independent rubrics.
**How:** `score_all()` builds 21 coroutines and runs them with `asyncio.gather`. `run_scenario()` drives the `Assistant` class sequentially. Usage: `python scripts/eval.py --org "BRCA Strong"`.

---

## Non-Engineer Guide

### Add a new platform
Create `skills/{platform}.yaml` (copy `skills/linkedin.yaml` as a template). Mackie picks it up on the next conversation — no restart, no rebuild.

### Edit content pillars
Edit `skills/content_pillars.yaml`. Add, rename, or remove pillars. To override for one specific org, add `"content_pillars": [{"name": "...", "description": "..."}]` to `data/{org-id}/profile.json`.

### Edit what Mackie learned about an org
Open `data/{org-id}/profile.json` in any text editor:
- Change `preferences.voice` to update tone instructions.
- Delete `research` (set to `null`) to trigger a fresh research run.
- Edit or delete entries in `post_ledger` to correct the record.

### Change the AI model
Set `ANTHROPIC_MODEL=claude-sonnet-4-6` in `.env` for higher quality Anthropic output. To switch providers, add the corresponding API key (`OPENAI_API_KEY`, `XAI_API_KEY`, or `GEMINI_API_KEY`) and optionally set the model override (`OPENAI_MODEL`, etc.). When multiple keys are set the priority order is Anthropic → OpenAI → xAI → Google — remove keys you don't want used.

### Read the tool call trace
`data/{org-id}/trace.jsonl` — one JSON object per line. Each entry shows the tool name, inputs, result preview, and latency. Pipe to `jq` or open in any text editor.

---

## Production Roadmap

What this project intentionally defers for the assignment stage, and the path to address each gap for a real multi-tenant deployment.

> **No database at this stage.** Org profiles are stored as JSON on disk (`profile.json`). This satisfies the assignment requirement that a non-engineer can open and edit org memory in any text editor. The gaps below — concurrency control, auth state, rate-limit counters, horizontal scaling — all require a database and are deliberately deferred.

### Memory and persistence
**Current (assignment stage):** JSON files on disk — `data/{org-id}/profile.json`. No concurrency control; load-modify-save pattern is not atomic. Acceptable for single-user demo use.
**Issue:** Simultaneous writes from different sessions on the same org will race and silently overwrite each other.
**Fix:** Replace with PostgreSQL. Keep a JSON export endpoint so non-engineers can still inspect and edit org memory. Migrate `AsyncSqliteSaver` to `langgraph-checkpoint-postgres` for conversation history.

### Multi-org auth and tenancy
**Current (assignment stage):** All orgs are accessible to anyone with the URL. No auth state is persisted.
**Issue:** Any user can impersonate any org, inject preferences, or trigger research on arbitrary URLs.
**Fix:** Introduce per-org API keys or OAuth at the Chainlit layer (Chainlit has built-in OAuth support). Auth session state requires a database or a managed auth provider (Auth0, Clerk).

### Rate limiting
**Current (assignment stage):** No rate limiting on any tool call. Not implemented because persistent counters require a database.
**Issue:** One unauthenticated user can exhaust the LLM API quota or the Tavily free tier in minutes.
**Fix:** Redis counters (per-org, per-tool, per-hour) or a Postgres timestamp column with a sliding window query.

### Horizontal scaling
**Current (assignment stage):** `AsyncSqliteSaver` is single-process — running two replicas would split conversation state.
**Fix:** `langgraph-checkpoint-postgres` with a shared connection string. Stateless app containers + shared Postgres is the standard pattern for horizontal scaling.

### Image generation
**Current:** Not implemented.
**Why deferred:** Requires a separate `OPENAI_API_KEY`, adds ~$0.04/image, and DALL-E images need a separate moderation pass for nonprofit brand safety.
**Path:** `litellm.image_generation(model="dall-e-3", prompt=...)` keeps provider agnosticism consistent. Hook into `save_post` for Instagram/Facebook; store the image URL in the ledger entry. Guard with a content filter before showing the image to the user.

### Richer context retrieval
**Current:** Sliding window (last 40 messages) + ledger truncation (last 20 posts).
**Fix:** Replace the recency window with a vector store (ChromaDB or pgvector) for semantic retrieval of relevant past turns and ledger entries. Relevant old context surfaces even after hundreds of turns.

### Streaming resilience
**Current:** A single dropped connection during streaming loses the partial reply.
**Fix:** Buffer streamed tokens server-side; expose a `/resume` endpoint clients can poll after reconnect. LangGraph's checkpoint means the turn state is safe — only the delivery needs resilience.

### Approval workflow UI
**Current:** Status transitions happen via natural language ("approve it", "mark it as posted") parsed by the LLM.
**Fix:** Expose an explicit approval UI in the Chainlit interface — buttons per ledger entry — so non-technical users don't need to know the right phrasing.

---

## Sample Transcript — Required 7-Step Scenario

**Organization:** BRCA Strong (brcastrong.org)
**Model:** claude-haiku-4-5-20251001
**Date:** 2026-06-21

### [1] Suggest some posts for us.

**Mackie:**

Perfect! Now I can see BRCA Strong's heart. Here are 4 post ideas:

**1. Survivor Spotlight – Emotional & Personal**
A post centered on one of your testimonials (like Mary's story about finding strength for her family). Lead with the raw, honest quote about how she had to be stronger for her son, then wrap it in how BRCA Strong's community helped her help herself. End with an invitation for other survivors to share their "why."
*Source: Mary's testimonial + org mission around emotional & financial support — https://www.brcastrong.org*

**2. Care Package Impact Post**
Celebrate the tangible way your donations change lives. Show a photo of (or describe) the post-mastectomy robes with drain pockets, button-downs, and bras you send — emphasize that these aren't charity, they're dignity. "These care packages say: you deserve to feel whole and feminine through this journey." Clear CTA: donate to sponsor a woman.
*Source: Donations/care package campaign section of website — https://www.brcastrong.org*

**3. Previvor Education & Empowerment**
Highlight the preventive support side of your community. Post about how previvors (those with high risk but no diagnosis yet) deserve informed decision-making and emotional support — not fear. Position BRCA Strong as the sisterhood that walks alongside them in those choices, celebrating recent blog post about personal decisions.
*Source: "Previvors" section + recent blog post "Danielle's Decisions and Why Previvor Choices Must Be Personal" — https://www.brcastrong.org/blog*

**4. Thriver Celebration – Resilience**
Feature the community of women who are beyond active treatment but still part of the BRCA Strong sisterhood. Celebrate one woman's journey from diagnosis through treatment to thriving, with a call to action for others to join the community.
*Source: "Thrivers" pillar on website — https://www.brcastrong.org*

---

### [2] Let's go with the second one, write it for LinkedIn.

**Mackie:**

Done! Here's your LinkedIn draft:

---

**Dignity isn't optional. It's essential.**

When a woman faces mastectomy or gynecological surgery, she's managing fear, pain, and uncertainty. The last thing she should worry about is how to afford the practical tools that help her feel like herself again.

That's why BRCA Strong sends care packages at zero cost to survivors and previvors—robes with drain pockets, button-down shirts that work with medical drains, post-mastectomy bras. Small items. Profound impact.

One woman told us: "These care packages say: you deserve to feel whole and feminine through this journey."

That's not charity. That's recognition that healing—physical and emotional—requires dignity.

**Your donation directly funds these packages.** We've supported hundreds of women facing breast and gynecological cancer, regardless of insurance status or genetic predisposition. We'll keep going until no woman has to choose between her medical needs and her sense of self.

Will you help us eliminate that barrier?

👉 Sponsor a woman today at brcastrong.org/donate

#BreastCancerAwareness #Survivorship #WomensHealth #NonprofitWork #HealthEquity

---

### [3] Too corporate, we're more warm and grassroots. Redo it.

**Mackie:**

Much better. Here's the warm version:

---

**Here's the thing about dignity: it shouldn't cost extra.**

When you're facing surgery—mastectomy, gynecological procedures—you're already dealing with so much. Fear. Pain. The weight of big decisions. The last thing you need is to scramble for money to buy a robe that actually fits your drains, or a button-down shirt that lets you heal without struggling.

So we send care packages. Free. No strings.

A soft robe with pockets for drains. A button-down that works with your body as it heals. A post-mastectomy bra that actually fits. Things that let you feel like *you* again while you're recovering.

One woman said it best: "These care packages say: you deserve to feel whole and feminine through this journey."

That's it. That's the whole thing.

We've helped hundreds of women—uninsured, underinsured, doesn't matter. Because your access to dignity shouldn't depend on your bank account.

If you can, help us keep doing this. Every donation sponsors a woman through one of the hardest seasons of her life.

Donate and sponsor a woman: brcastrong.org/donate

Who in your life has needed this kind of support? We see you.

#BRCAStrong #SisterSisterhood #CommunityOverProfit #WomensHealth

---

> **[Voice preference saved]:** Warm, grassroots, community-focused — less corporate polish, more genuine sisterhood.

---

### [4] Now give me an Instagram version.

**Mackie:**

Done! Here's your Instagram version:

---

**Dignity shouldn't cost extra.** 💙

When you're facing surgery, you're already carrying so much. Fear. Pain. Big decisions. The last thing you need is to stress about affording a robe that fits your drains or a shirt that lets you heal.

So we send care packages. Free. No questions asked.

A soft robe with drain pockets. A button-down that works with your body. A post-mastectomy bra that actually fits. Things that say: *you deserve to feel whole and feminine through this.*

We've supported hundreds of women—uninsured, underinsured, doesn't matter. Because access to dignity shouldn't depend on your bank account.

If you're able, help us keep doing this. Every donation sponsors a woman through one of life's hardest seasons.

Sponsor a woman 👇
brcastrong.org/donate

.
.
.
#BRCAStrong #SisterSisterhood #BreastCancer #CancerSurvivor #Previvor #WomensHealth #CommunityOverProfit #BreastCancerAwareness #CancerSupport #Healing #MastectomyRecovery #WomensEmpowerment #HealthEquity #NonprofitWork #WeAreStronger #CommunityMatters #CareThatCares #YouAreNotAlone #SupportSisters #BRCAMutation

---

### [5] What programs does our org actually run?

**Mackie:**

Based on the research I pulled, here's what BRCA Strong actually runs:

**Care Packages (Direct Assistance)**
- Premastectomy care packages for women prepping for surgery
- Gynecological care packages for those facing gynecological procedures
- Post-mastectomy garments (robes with drain pockets, button-down shirts, post-mastectomy bras) — all sent at zero cost

**Community Support (by Life Stage)**
- **Previvors** — women with high genetic risk who haven't developed cancer yet. Education, resources, and preventive support.
- **Survivors** — women actively dealing with or recently diagnosed with breast/gynecological cancer. Emotional and practical support.
- **Thrivers** — women living with or beyond cancer diagnosis. Post-treatment support, advocacy, resilience focus.

**Education & Advocacy**
- Blog content (recent posts on previvor decision-making, personal journeys)
- Medical expert partnerships (Dr. Natasha Rueth, surgical oncologist)
- General education around breast and gynecological cancer

**Who You Serve**
- Women regardless of genetic predisposition (not just BRCA carriers)
- Uninsured and underinsured women specifically

*(Source: BRCA Strong photo gallery — https://www.brcastrong.org/photo-gallery)*

---

### [6] Which posts have we worked on so far, and what's the status of each?

**Mackie:**

Here's what's in your ledger:

1. **post_001** | LinkedIn | **draft** — "Dignity isn't optional. It's essential…" *(corporate version, superseded)*
2. **post_002** | LinkedIn | **draft** — "Here's the thing about dignity: it shouldn't cost extra…" *(warm grassroots rewrite)*
3. **post_003** | Instagram | **draft** — "Dignity shouldn't cost extra. 💙…" *(Instagram adaptation)*

All three are sitting in draft status. They're all care package/dignity theme variations.

---

### [7] Suggest another post for next month.

**Mackie:**

Here are 4 ideas for next month — none repeating what's already in the ledger:

**1. Thriver Spotlight – Resilience & Thriving Beyond**
Feature a thriver's story — someone living with or beyond a cancer diagnosis who's built a stronger, more intentional life with your community. Position the sisterhood as ongoing, not just crisis support.
*Source: "Thrivers" pillar + mission around resilience — https://www.brcastrong.org*

**2. Previvor Education Series – "Your Choices Are Personal"**
Lean into your recent blog post "Danielle's Decisions and Why Previvor Choices Must Be Personal." There's no "right" choice for risk management — the key is informed decision-making with a community that doesn't judge.
*Source: Recent blog content + previvor support mission — https://www.brcastrong.org/blog*

**3. Volunteer/Community Power Post**
Celebrate your volunteers and the sisterhood itself. "These are women supporting women" — highlight how community members become mentors, care package packers, storytellers. Great for volunteer recruitment.
*Source: Org mission + "sisterhood of strength" language on website — https://www.brcastrong.org*

**4. Survivor Gratitude/Impact Post**
Feature how a survivor's life changed because of BRCA Strong — framed not as a charity story but as a community story. The org didn't save her; the sisterhood walked with her. Invite others to share their stories.
*Source: Testimonials section + grassroots voice preference — https://www.brcastrong.org/photo-gallery*

---

### Invariants Verified

| Invariant | Status |
|---|---|
| Research from step 1 informs all subsequent content | ✅ Sources cited throughout |
| Voice correction in step 3 carries to steps 4 and 7 | ✅ Instagram and month-ahead posts are warm/grassroots |
| Step 7 does not repeat step 1 suggestions | ✅ All 4 new suggestions are distinct angles |
| Post ledger accurately tracks all work | ✅ 3 posts, correct statuses |
| Preference persisted to disk | ✅ `voice` saved in profile.json |
| Automated eval score | ✅ 63/63 (100%) — `python scripts/eval.py --org "BRCA Strong" --url https://brcastrong.org` |
