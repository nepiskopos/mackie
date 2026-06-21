"""
LLM-as-judge evaluator for the end-to-end 7-step scenario.

Owns: driving the 7-step scenario against a test org, scoring each
response on three rubrics in parallel (21 total judge calls), and printing a
markdown report. Does not own the assistant logic — it imports and drives it.

Usage (run from project root):
    python scripts/eval.py
    python scripts/eval.py --org "Myra's Kids" --url https://myraskids.org

Requires at least one provider API key (ANTHROPIC_API_KEY, OPENAI_API_KEY,
XAI_API_KEY, or GEMINI_API_KEY) and TAVILY_API_KEY for the research step.
The judge model is resolved by resolve_model() using the same provider-selection
logic as the main app (priority: Anthropic → OpenAI → xAI → Google).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from agent import resolve_model  # noqa: E402 — must follow load_dotenv

JUDGE_MODEL: str = resolve_model()

SCENARIO: list[str] = [
    "Suggest some posts for us.",
    "Let's go with the second one, write it for LinkedIn.",
    "Too corporate, we're more warm and grassroots. Redo it.",
    "Now give me an Instagram version.",
    "What programs does our org actually run?",
    "Which posts have we worked on so far, and what's the status of each?",
    "Suggest another post for next month.",
]

RUBRICS: dict[str, str] = {
    "source_attribution": (
        "Does this response cite at least one specific source that informed its content? "
        "MANDATORY EXEMPTIONS — output score=3 immediately with no evaluation for: "
        "step 2 (writing a platform draft of an already-sourced suggestion), "
        "step 3 (voice rewrite of an existing draft), "
        "step 4 (platform adaptation of an existing draft), "
        "step 6 (ledger status query — answered from conversation history, not external research). "
        "For steps 1, 5, and 7 only: "
        "0 = no source mentioned; "
        "1 = vague reference only ('my research', 'your website') with no URL; "
        "2 = source named but URL missing, or URL present but no descriptive label; "
        "3 = at least one complete citation with BOTH a descriptive label AND a URL."
    ),
    "voice_consistency": (
        "In a 7-step scenario, the user corrected the tone to 'warm and grassroots' in step 3. "
        "MANDATORY RULE: if this is step 1 or step 2, you MUST output score=3 with no evaluation — "
        "the tone correction has not happened yet. "
        "For steps 3-7 only: 0 = formal or corporate; 1 = neutral; 2 = somewhat warm; 3 = clearly warm and grassroots."
    ),
    "ledger_integrity": (
        "For step 7 only: does this response avoid repeating the specific post ideas from step 1? "
        "0 = same ideas recycled; 1 = mostly same; 2 = mostly different; 3 = all new ideas. "
        "For every step except step 7: score 3 (not applicable)."
    ),
}

# A rubric score of MIN_PASS_SCORE or above counts as passing for invariant checks.
MIN_PASS_SCORE: int = 2


async def _score_one(
    step: int,
    user_msg: str,
    reply: str,
    rubric_name: str,
    rubric_desc: str,
) -> dict:
    """Call the judge LLM for one (step, rubric) pair and return {rubric, score, reason}."""
    prompt = (
        f"You are a strict evaluator. Score the assistant response on the rubric below.\n\n"
        f"RUBRIC ({rubric_name}): {rubric_desc}\n\n"
        f"STEP {step} — User said: \"{user_msg}\"\n"
        f"Assistant replied: \"{reply[:1500]}\"\n\n"
        "Reply with ONLY valid JSON — no markdown fences, no extra text: "
        "{\"score\": <integer 0-3>, \"reason\": \"<one sentence>\"}"
    )
    response = await litellm.acompletion(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0,
    )
    text = response.choices[0].message.content.strip()
    # Strip markdown fences if the model wrapped its JSON (e.g. ```json...```)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        data = json.loads(text)
        return {"rubric": rubric_name, "score": int(data["score"]), "reason": data["reason"]}
    except Exception:
        # Last-resort: extract the first {...} block
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return {"rubric": rubric_name, "score": int(data["score"]), "reason": data["reason"]}
            except Exception:
                pass
        return {"rubric": rubric_name, "score": -1, "reason": f"parse error: {text[:80]}"}


async def score_all(responses: list[dict]) -> list[list[dict]]:
    """
    Run all 21 judge calls (7 steps × 3 rubrics) concurrently via asyncio.gather.

    Args:
        responses: List of dicts with keys step, user, reply.

    Returns:
        List of 7 items, each a list of 3 rubric-score dicts.
    """
    tasks = [
        _score_one(item["step"], item["user"], item["reply"], name, desc)
        for item in responses
        for name, desc in RUBRICS.items()
    ]
    flat = await asyncio.gather(*tasks)
    n = len(RUBRICS)
    return [list(flat[i * n: (i + 1) * n]) for i in range(len(responses))]


def run_scenario(org_name: str, org_url: str) -> list[dict]:
    """
    Drive the Assistant through the 7-step required scenario.

    Initialises the org profile with the name and website before the first
    turn so the system prompt shows the correct org and the assistant can
    call research_org immediately when asked for suggestions. This mirrors
    what app.py does via the AskUserMessage onboarding prompts.

    Args:
        org_name: Test org name (used as org_id after slugification).
        org_url:  Optional website URL. Stored in the profile so the
                  assistant can scrape it during the research step.

    Returns:
        List of 7 dicts with keys: step, user, reply.
    """
    from agent.assistant import Assistant
    from agent.memory import slugify, load_profile, save_profile

    org_id = slugify(org_name)

    # Pre-populate the profile so the system prompt shows the correct org
    # name and website — the same information the user supplies via the
    # onboarding prompts in app.py.
    profile = load_profile(org_id)
    if not profile["name"]:
        profile["name"] = org_name
    if not profile["website"] and org_url:
        profile["website"] = org_url
    save_profile(org_id, profile)

    async def _run() -> list[dict]:
        # Use a unique thread_id so each eval run starts with a clean
        # conversation history — no contamination from prior runs stored
        # in data/checkpoints.db under the same org_id thread.
        thread_id = f"{org_id}-eval-{int(time.time())}"
        assistant = Assistant(org_id, thread_id=thread_id)
        responses: list[dict] = []
        for i, msg in enumerate(SCENARIO, start=1):
            print(f"  [{i}/7] {msg}")
            reply = await assistant.chat(msg)
            print(f"        → {reply[:100].replace(chr(10), ' ')}...\n")
            responses.append({"step": i, "user": msg, "reply": reply})
        return responses

    return asyncio.run(_run())


def _check_invariants(scores: list[list[dict]]) -> dict[str, bool]:
    """
    Derive the three required invariant results from the rubric scores.

    Returns a dict mapping invariant name → pass/fail bool.
    """
    # range(3, 7) = 0-based indices for steps 4-7 (voice correction happens in step 3)
    return {
        "source_attribution_step1": scores[0][0]["score"] >= MIN_PASS_SCORE,
        "voice_consistency_steps4_7": all(
            scores[i][1]["score"] >= MIN_PASS_SCORE for i in range(3, 7)
        ),
        "ledger_integrity_step7": scores[6][2]["score"] >= MIN_PASS_SCORE,
    }


def print_report(responses: list[dict], scores: list[list[dict]]) -> None:
    """Print the full evaluation report as markdown to stdout."""
    lines: list[str] = ["# Eval Report\n"]

    total, max_total = 0, 0
    for item, step_scores in zip(responses, scores):
        step_sum = sum(s["score"] for s in step_scores if s["score"] >= 0)
        total += step_sum
        max_total += len(step_scores) * 3
        lines.append(f"## Step {item['step']}: {item['user']}")
        lines.append(f"**Score: {step_sum}/{len(step_scores) * 3}**\n")
        for s in step_scores:
            mark = "pass" if s["score"] >= MIN_PASS_SCORE else "FAIL"
            lines.append(f"- [{mark}] **{s['rubric']}**: {s['score']}/3 — {s['reason']}")
        lines.append("")

    pct = round(total / max_total * 100) if max_total else 0
    verdict = "PASS" if pct >= 70 else "NEEDS WORK"
    lines.append(f"## Overall: {total}/{max_total} ({pct}%) — {verdict}\n")

    lines.append("## Invariant Checks\n")
    invariants = _check_invariants(scores)
    labels = {
        "source_attribution_step1": "Source attribution present in step 1",
        "voice_consistency_steps4_7": "Warm/grassroots voice maintained in steps 4-7",
        "ledger_integrity_step7": "Step 7 introduces new content (not step 1 repeats)",
    }
    for key, label in labels.items():
        result = "PASS" if invariants[key] else "FAIL"
        lines.append(f"- **{result}** — {label}")

    print("\n".join(lines))


def main() -> None:
    """Parse CLI args, run the scenario, score, and print the report."""
    parser = argparse.ArgumentParser(
        description="Evaluate Mackie on the 7-step required scenario."
    )
    parser.add_argument("--org", default="BRCA Strong", help="Test org name (default: BRCA Strong)")
    parser.add_argument("--url", default="", help="Org website URL (optional)")
    args = parser.parse_args()

    print(f"Running 7-step scenario for: {args.org}\n")
    responses = run_scenario(args.org, args.url)

    print(f"Scoring {len(responses) * len(RUBRICS)} rubric calls in parallel...")
    scores = asyncio.run(score_all(responses))

    print_report(responses, scores)


if __name__ == "__main__":
    main()
