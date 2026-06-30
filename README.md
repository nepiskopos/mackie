# Mackie — NPO Social Media Assistant

A conversational AI assistant for nonprofit marketing teams. Mackie researches your organization, suggests on-brand social media content across LinkedIn, Instagram, and Facebook, learns from feedback, and tracks every post through a review workflow.

## Features

- **Org research before writing** — scrapes your website and runs parallel web searches; every content suggestion cites its source
- **Platform-specific drafts** — LinkedIn, Instagram, and Facebook each get the correct tone, length, and hashtag style, driven by YAML config files anyone can edit
- **Content pillars** — five editorial themes (Community Impact, Programs & Services, Awareness & Education, Events & Campaigns, Behind the Scenes) frame every suggestion; overridable per org without touching code
- **Cross-session memory** — voice corrections and brand preferences are written to disk immediately and apply automatically in every future session
- **Post ledger** — every suggestion and draft is tracked with a status (`suggested → draft → approved → posted`) queryable in plain language
- **Content calendar** — four-week calendar generated with four parallel LLM calls, each week saved to the ledger
- **Extensibility without code** — add a new platform by dropping a YAML file; edit org memory in any text editor; no restart required

## Demo

On startup, the app prompts for your organization's name and website URL. After entering them, this conversation works out of the box (try it with any real nonprofit):

```
"Suggest some posts for us."
"Let's go with the second one, write it for LinkedIn."
"Too corporate, we're more warm and grassroots. Redo it."
"Now give me an Instagram version."
"What programs does our org actually run?"
"Which posts have we worked on so far, and what's the status of each?"
"Suggest another post for next month."
```

The voice correction in step 3 ("warm and grassroots") carries automatically through all subsequent posts — including the fresh suggestion in step 7, which avoids repeating any idea from step 1.

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| LLM provider adapter | [LiteLLM](https://github.com/BerriAI/litellm) | Provider selected by API key priority (Anthropic → OpenAI → xAI → Google); model override via `{PROVIDER}_MODEL` env var |
| Agent framework | [LangGraph](https://github.com/langchain-ai/langgraph) | Typed state, `AsyncSqliteSaver` checkpoint, clean agent→tools loop |
| Conversation persistence | SQLite (`AsyncSqliteSaver`) | Survives container restarts; sessions isolated by `thread_id` |
| Org memory | JSON on disk | Human-readable; editable in any text editor without migrations |
| Web search | [Tavily](https://tavily.com) | Async-native, structured results, built for agents |
| Web scraping | httpx + BeautifulSoup4 | Async HTTP; strips nav/scripts/styles for clean text |
| Web UI | [Chainlit](https://github.com/Chainlit/chainlit) | Native async; built-in tool step display; streaming |
| Observability | `trace.jsonl` | Zero deps; one JSONL entry per tool call |
| Eval | LLM-as-judge (`scripts/eval.py`) | 21 parallel rubric calls via `asyncio.gather` |
| CI | GitHub Actions | Runs tests on every push and pull request |

## Architecture

**How it works:** Each user message enters a `agent → tools` loop. The LLM receives the full system prompt (rebuilt from `profile.json` every turn) and the last 40 messages; it either replies in plain text or calls one of the tools listed below. Text streams token-by-token to the UI; tool steps appear as each one completes. The loop exits when the LLM returns no tool calls.

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
         │  system prompt rebuilt each turn from profile.json
         │
         ▼
     LangGraph  (agent/graph.py)
         │  AgentState + SQLite checkpoint  ·  sliding window: messages[-40]
         │
         ▼
     LiteLLM  ──►  Anthropic / OpenAI / Gemini / Ollama
         │
         ├── research_org    ──► httpx scrape + 3× Tavily search (asyncio.gather)
         ├── refresh_research ──► re-runs research, preserves preferences + ledger
         ├── web_search      ──► Tavily one-shot
         ├── save_post       ──► data/{org-id}/profile.json
         ├── update_post_status
         ├── save_preference ──► persists to disk immediately
         └── generate_calendar ──► 4 parallel LLM calls (asyncio.gather)

Persistent state:  data/{org-id}/profile.json   (org memory, preferences, ledger)
                   data/{org-id}/trace.jsonl     (per tool call: ts, tool, latency)
                   data/checkpoints.db           (LangGraph conversation history)
Extensible config: skills/*.yaml                (platforms + content pillars)
```

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env   # fill in TAVILY_API_KEY and at least one provider API key
docker compose up --build
```

Opens at `http://localhost:8000`.

The welcome screen asks for your org name and website URL — enter them to begin. The first time you ask for content suggestions, Mackie researches your org automatically (website scrape + 3 parallel web searches, ~5–15 seconds). Results are cached in `data/{org-id}/profile.json` so subsequent sessions skip the research step.

### Local (no Docker)

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in real values
chainlit run app.py    # web UI → http://localhost:8000
python main.py         # CLI
```

See [design_doc.md](design_doc.md) for the full architecture rationale.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TAVILY_API_KEY` | required | Tavily search API key — free tier at [tavily.com](https://tavily.com) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (set at least one provider key) |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `XAI_API_KEY` | — | xAI API key |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Anthropic model override |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model override |
| `XAI_MODEL` | `xai/grok-3-mini` | xAI model override |
| `GEMINI_MODEL` | `gemini/gemini-2.0-flash` | Google model override |

When multiple provider keys are set the priority order is: Anthropic → OpenAI → xAI → Google.

## Tests

```bash
pytest tests/ -v   # 110 tests, all offline (no API calls), < 10 seconds
```

## Evaluator

Runs the 7-step scenario end-to-end and scores each response on three rubrics (source attribution, voice consistency, ledger integrity) using 21 parallel LLM judge calls:

```bash
python scripts/eval.py --org "BRCA Strong" --url https://brcastrong.org
```

The judge model is resolved automatically from your provider API keys (same priority order as the main app).

## Project Structure

```
.
├── app.py                        # Web UI entry point (Chainlit)
├── main.py                       # CLI entry point
├── chainlit.md                   # Web UI welcome screen
├── agent/
│   ├── __init__.py               # re-exports PROVIDERS, resolve_model
│   ├── config.py                 # provider selection + model resolution
│   ├── assistant.py              # System prompt builder + chat entry point
│   ├── graph.py                  # LangGraph agent→tools loop + SQLite checkpoint
│   ├── memory.py                 # Org profile, post ledger, preferences (JSON)
│   ├── research.py               # Website scraping + Tavily web search
│   └── tools.py                  # Tool definitions + dispatch
├── skills/                       # Platform configs — editable without code changes
│   ├── linkedin.yaml
│   ├── instagram.yaml
│   ├── facebook.yaml
│   └── content_pillars.yaml
├── tests/                        # 110 unit tests (pytest, all offline)
├── scripts/
│   └── eval.py                   # LLM-as-judge evaluator
├── .github/workflows/
│   └── deploy.yml                # CI: runs tests on every push and pull request
└── data/                         # Per-org storage (auto-created, gitignored)
    └── {org-id}/
        ├── profile.json          # Org memory — open in any text editor
        └── trace.jsonl           # Tool call trace (ts, tool, latency)
```

## Adding a New Platform

Create a new file in `skills/`, e.g. `skills/threads.yaml`:

```yaml
name: Threads
id: threads
tone: casual, conversational, text-forward
max_length: 500
hashtag_count: "0-3"
hashtag_style: minimal, only if natural
format_hints: |
  - Short and punchy — like a tweet with more room
  - First-person voice works well here
  - No link previews — put the CTA in text
```

Mackie picks it up on the next conversation turn — no restart, no rebuild.

## Editing Org Memory

Each org's learned data is stored in `data/{org-id}/profile.json`. Open it in any text editor to view or change the post ledger, learned preferences, or org profile. Set `"research": null` to trigger a fresh research run next session.

## Production Roadmap

This is a complete working demo. No database is used at this assignment stage — org memory is JSON on disk, which lets a non-engineer edit it in any text editor. The gaps below are deliberate deferrals; all except image generation and the UI button require a database to fix properly.

| Gap | Why deferred | Fix |
|---|---|---|
| No concurrency control on writes | No DB at this stage — load-modify-save is not atomic | PostgreSQL for org memory + `langgraph-checkpoint-postgres` for conversation history |
| No auth | Auth session state requires a DB or managed auth provider | Per-org API keys or OAuth at the Chainlit layer |
| No rate limiting | Persistent counters require a DB or Redis | Redis per-org/per-tool counters or a Postgres sliding-window query |
| Single-process SQLite | No shared DB for multi-replica state | `langgraph-checkpoint-postgres` for horizontal scaling |
| No image generation | Extra API key + cost + moderation needed | `litellm.image_generation(model="dall-e-3")` on Instagram/Facebook saves |
| Context window grows unbounded | Vector store requires a DB | Replace recency window with pgvector for semantic retrieval |
| Status changes via natural language only | UI work, not DB-related | Explicit approve/reject buttons in the Chainlit UI |
| No social media publishing | Requires platform API credentials and OAuth per platform; adds compliance complexity | Add a `publish_post` tool calling Facebook Graph API / LinkedIn API; store OAuth tokens per org in the database; gate on `approved` status |
| No content scheduling | Ledger has no `scheduled_for` field; calendar assigns week-level buckets, not specific dates | Add `scheduled_for` timestamp to the ledger schema; derive specific dates in `generate_calendar`; trigger publishing via a job queue (Celery + Redis or AWS EventBridge) |
| No export to scheduling tools | Out of scope for a demo with no database | CSV export of approved/planned posts; or direct integration with Buffer / Hootsuite publishing APIs |
| Single-org per session | Chainlit session is bound to one org at startup; switching orgs requires a page reload | Store `org_id` in `cl.user_session` and add an org-switcher command; update `thread_id` on switch to preserve per-org history isolation |
| Research never expires | `refresh_research` requires explicit user trigger; no staleness detection | Add `research_expires_at` to `profile.json`; surface a proactive "Research is X days old — refresh?" prompt at session start |
| `web_search` results lack prompt-injection defence | `_dispatch` web_search branch formats results as plain text; `<external_content>` wrapping is only applied in `_format_research_summary` for `research_org` | Wrap each snippet in `<external_content source="web_search" url="...">` tags in the `web_search` dispatch branch, matching the defence applied to research results |
| Silent scraping failures on redirecting sites | `follow_redirects=False` is required to prevent redirect-chain SSRF bypass; most sites redirect HTTP → HTTPS or non-www → www | After a 3xx response, validate the `Location` header through `_is_safe_url` and retry to the canonical destination |
| Content calendar generates plans, not drafts | Each calendar entry is a one-sentence angle; `plan_week` is not asked to write post copy | After week planning, pass each entry back through the draft pipeline with full platform guidelines; save as `"draft"` instead of `"planned"` |

Full rationale in [design_doc.md — Production Roadmap](design_doc.md).
