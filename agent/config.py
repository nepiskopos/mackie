"""
Provider selection and model resolution for Mackie.

Owns: mapping environment variables (provider API keys and model names) to the
active LiteLLM model string. Does not own any LLM call logic — it only reads
os.environ and returns a model string.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class _Provider:
    """Static descriptor for one LLM provider."""

    id: str
    key_env: str
    model_env: str
    model_prefixes: tuple[str, ...]
    default_model: str


# Priority order: Anthropic → OpenAI → xAI → Google.
# This list is the single source of truth for provider metadata — do not duplicate
# default_model values elsewhere; import from here instead.
PROVIDERS: list[_Provider] = [
    _Provider(
        id="anthropic",
        key_env="ANTHROPIC_API_KEY",
        model_env="ANTHROPIC_MODEL",
        model_prefixes=("claude", "anthropic/"),
        default_model="claude-haiku-4-5-20251001",
    ),
    _Provider(
        id="openai",
        key_env="OPENAI_API_KEY",
        model_env="OPENAI_MODEL",
        model_prefixes=("gpt-", "o1-", "o3-", "openai/"),
        default_model="gpt-4o-mini",
    ),
    _Provider(
        id="xai",
        key_env="XAI_API_KEY",
        model_env="XAI_MODEL",
        model_prefixes=("grok-", "xai/"),
        default_model="xai/grok-3-mini",
    ),
    _Provider(
        id="google",
        key_env="GEMINI_API_KEY",
        model_env="GEMINI_MODEL",
        model_prefixes=("gemini/",),
        default_model="gemini/gemini-2.0-flash",
    ),
]


def model_matches_provider(model: str, provider: _Provider) -> bool:
    """Return True if the model string is compatible with the given provider."""
    return any(model.startswith(prefix) for prefix in provider.model_prefixes)


def is_anthropic_model(model: str) -> bool:
    """
    Return True if model targets the native Anthropic API.

    Used by graph.py to decide whether to attach prompt-caching headers.
    Bedrock ("bedrock/anthropic.claude-*", "us.anthropic.claude-*") and
    Vertex AI routes are intentionally excluded — they use provider-side
    caching, not the anthropic-beta header.
    """
    anthropic = next(p for p in PROVIDERS if p.id == "anthropic")
    return model_matches_provider(model, anthropic)


def resolve_model() -> str:
    """
    Resolve the LLM model to use from environment variables.

    Checks provider API keys in priority order (Anthropic → OpenAI → xAI → Google)
    and returns the model string for the highest-priority provider whose key is set.
    Prints an error to stderr if the configured model doesn't match its provider
    (e.g. ANTHROPIC_API_KEY set but ANTHROPIC_MODEL=gpt-4o-mini).

    When no API key is set (e.g. during tests), falls back to the Anthropic default.

    Returns:
        A LiteLLM-compatible model string ready to pass to litellm.acompletion().
    """
    available = [p for p in PROVIDERS if os.getenv(p.key_env)]
    if not available:
        return PROVIDERS[0].default_model  # Anthropic default; covers offline/test use

    selected = available[0]
    model = os.getenv(selected.model_env) or selected.default_model

    if not model_matches_provider(model, selected):
        print(
            f"ERROR: {selected.model_env}={model!r} does not match provider "
            f"'{selected.id}' (expected prefix in {selected.model_prefixes}). "
            f"Check your .env configuration.",
            file=sys.stderr,
        )

    return model
