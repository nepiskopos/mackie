"""Unit tests for agent/config.py — provider selection and model resolution."""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from agent.config import resolve_model


def _env(**kwargs: str) -> dict[str, str]:
    """Build a full os.environ patch: unset all relevant vars, then apply overrides.

    Empty string is falsy in os.getenv(), so unset vars are treated as absent
    by resolve_model(). This isolates tests from whatever the caller's shell has set.
    """
    keys = {
        "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
        "OPENAI_API_KEY", "OPENAI_MODEL",
        "XAI_API_KEY", "XAI_MODEL",
        "GEMINI_API_KEY", "GEMINI_MODEL",
    }
    base: dict[str, str] = {k: "" for k in keys}
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# No keys set — offline / test fallback
# ---------------------------------------------------------------------------

def test_no_keys_returns_anthropic_default():
    with patch.dict("os.environ", _env(), clear=False):
        assert resolve_model() == "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Single provider selected
# ---------------------------------------------------------------------------

def test_anthropic_key_returns_default_anthropic_model():
    with patch.dict("os.environ", _env(ANTHROPIC_API_KEY="sk-ant-abc"), clear=False):
        assert resolve_model() == "claude-haiku-4-5-20251001"


def test_anthropic_key_with_custom_model():
    with patch.dict("os.environ", _env(ANTHROPIC_API_KEY="sk-ant-abc", ANTHROPIC_MODEL="claude-opus-4-8"), clear=False):
        assert resolve_model() == "claude-opus-4-8"


def test_openai_key_returns_default_openai_model():
    with patch.dict("os.environ", _env(OPENAI_API_KEY="sk-abc"), clear=False):
        assert resolve_model() == "gpt-4o-mini"


def test_openai_key_with_custom_model():
    with patch.dict("os.environ", _env(OPENAI_API_KEY="sk-abc", OPENAI_MODEL="gpt-4o"), clear=False):
        assert resolve_model() == "gpt-4o"


def test_xai_key_returns_default_xai_model():
    with patch.dict("os.environ", _env(XAI_API_KEY="xai-abc"), clear=False):
        assert resolve_model() == "xai/grok-3-mini"


def test_gemini_key_returns_default_gemini_model():
    with patch.dict("os.environ", _env(GEMINI_API_KEY="AIzaabc"), clear=False):
        assert resolve_model() == "gemini/gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Priority: multiple keys set → Anthropic wins
# ---------------------------------------------------------------------------

def test_anthropic_beats_openai_when_both_keys_set():
    with patch.dict("os.environ", _env(ANTHROPIC_API_KEY="sk-ant-abc", OPENAI_API_KEY="sk-abc"), clear=False):
        assert resolve_model().startswith("claude")


def test_anthropic_beats_all_providers():
    with patch.dict(
        "os.environ",
        _env(ANTHROPIC_API_KEY="sk-ant-abc", OPENAI_API_KEY="sk-abc",
             XAI_API_KEY="xai-abc", GEMINI_API_KEY="AIzaabc"),
        clear=False,
    ):
        assert resolve_model().startswith("claude")


def test_openai_beats_xai_and_google():
    with patch.dict("os.environ", _env(OPENAI_API_KEY="sk-abc", XAI_API_KEY="xai-abc", GEMINI_API_KEY="AIzaabc"), clear=False):
        assert resolve_model().startswith("gpt-")


def test_xai_beats_google():
    with patch.dict("os.environ", _env(XAI_API_KEY="xai-abc", GEMINI_API_KEY="AIzaabc"), clear=False):
        assert resolve_model().startswith("xai/")


# ---------------------------------------------------------------------------
# Mismatch error
# ---------------------------------------------------------------------------

def test_mismatch_prints_error(capsys):
    with patch.dict("os.environ", _env(ANTHROPIC_API_KEY="sk-ant-abc", ANTHROPIC_MODEL="gpt-4o-mini"), clear=False):
        resolve_model()
    err = capsys.readouterr().err
    assert "ERROR" in err
    assert "ANTHROPIC_MODEL" in err
    assert "gpt-4o-mini" in err


def test_mismatch_still_returns_the_configured_model(capsys):
    with patch.dict("os.environ", _env(ANTHROPIC_API_KEY="sk-ant-abc", ANTHROPIC_MODEL="gpt-4o-mini"), clear=False):
        model = resolve_model()
    assert model == "gpt-4o-mini"


def test_correct_model_no_error(capsys):
    with patch.dict("os.environ", _env(ANTHROPIC_API_KEY="sk-ant-abc", ANTHROPIC_MODEL="claude-opus-4-8"), clear=False):
        resolve_model()
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# is_anthropic_model / model_matches_provider
# ---------------------------------------------------------------------------

from agent.config import is_anthropic_model, model_matches_provider, PROVIDERS


def test_is_anthropic_model_native_claude():
    assert is_anthropic_model("claude-haiku-4-5-20251001") is True


def test_is_anthropic_model_with_anthropic_prefix():
    assert is_anthropic_model("anthropic/claude-haiku-4-5-20251001") is True


def test_is_anthropic_model_rejects_openai():
    assert is_anthropic_model("gpt-4o-mini") is False


def test_is_anthropic_model_rejects_bedrock():
    # Bedrock cross-region and bedrock/ prefix must NOT match — they use AWS-side caching.
    assert is_anthropic_model("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0") is False
    assert is_anthropic_model("us.anthropic.claude-3-5-sonnet-20241022-v2:0") is False


def test_model_matches_provider_openai():
    openai = next(p for p in PROVIDERS if p.id == "openai")
    assert model_matches_provider("gpt-4o-mini", openai) is True
    assert model_matches_provider("claude-haiku-4-5-20251001", openai) is False


def test_model_matches_provider_google():
    google = next(p for p in PROVIDERS if p.id == "google")
    assert model_matches_provider("gemini/gemini-2.0-flash", google) is True
    assert model_matches_provider("gpt-4o-mini", google) is False
