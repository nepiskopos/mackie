"""Unit tests for agent/memory.py — org profile, post ledger, preferences."""


def test_slugify_converts_name_to_id():
    from agent.memory import slugify
    assert slugify("BRCA Strong") == "brca-strong"
    assert slugify("Myra's Kids") == "myra-s-kids"


def test_slugify_collapses_special_chars():
    from agent.memory import slugify
    assert slugify("Vets & Drones") == "vets-drones"
    assert slugify("--Leading Dashes--") == "leading-dashes"


def test_slugify_handles_unicode():
    from agent.memory import slugify
    result = slugify("Café Ñoño")
    # Non-ASCII chars are replaced by hyphens; result must be ASCII and non-empty.
    assert result.isascii()
    assert result


def test_load_profile_returns_defaults_for_new_org(tmp_data_dir):
    from agent.memory import load_profile
    profile = load_profile("new-org")
    assert profile["org_id"] == "new-org"
    assert profile["name"] == ""
    assert profile["post_ledger"] == []
    assert profile["preferences"] == {}
    assert profile["research"] is None


def test_save_and_load_roundtrip(tmp_data_dir):
    from agent.memory import load_profile, save_profile
    profile = load_profile("test-org")
    profile["name"] = "Test Org"
    profile["website"] = "https://testorg.org"
    save_profile("test-org", profile)
    loaded = load_profile("test-org")
    assert loaded["name"] == "Test Org"
    assert loaded["website"] == "https://testorg.org"


def test_save_profile_creates_directory(tmp_data_dir):
    from agent.memory import save_profile, load_profile
    profile = load_profile("brand-new-org")
    save_profile("brand-new-org", profile)
    assert (tmp_data_dir / "brand-new-org" / "profile.json").exists()


def test_add_post_assigns_sequential_ids(tmp_data_dir):
    from agent.memory import add_post
    id1 = add_post("org", {"platform": "linkedin", "content": "Post 1"})
    id2 = add_post("org", {"platform": "instagram", "content": "Post 2"})
    id3 = add_post("org", {"platform": "facebook", "content": "Post 3"})
    assert id1 == "post_001"
    assert id2 == "post_002"
    assert id3 == "post_003"


def test_add_post_sets_default_status(tmp_data_dir):
    from agent.memory import add_post, get_ledger
    add_post("org", {"platform": "linkedin", "content": "Hello"})
    assert get_ledger("org")[0]["status"] == "suggested"


def test_add_post_preserves_explicit_status(tmp_data_dir):
    from agent.memory import add_post, get_ledger
    add_post("org", {"platform": "linkedin", "content": "Hello", "status": "draft"})
    assert get_ledger("org")[0]["status"] == "draft"


def test_add_post_sets_created_at(tmp_data_dir):
    from agent.memory import add_post, get_ledger
    add_post("org", {"platform": "linkedin", "content": "Hello"})
    assert "created_at" in get_ledger("org")[0]


def test_update_post_status_success(tmp_data_dir):
    from agent.memory import add_post, update_post_status, get_ledger
    post_id = add_post("org", {"platform": "linkedin", "content": "Hello"})
    result = update_post_status("org", post_id, "approved")
    assert result is True
    assert get_ledger("org")[0]["status"] == "approved"


def test_update_post_status_missing_post_returns_false(tmp_data_dir):
    from agent.memory import update_post_status
    result = update_post_status("org", "post_999", "approved")
    assert result is False


def test_save_preference_persists(tmp_data_dir):
    from agent.memory import save_preference, load_profile
    save_preference("org", "voice", "warm and grassroots")
    assert load_profile("org")["preferences"]["voice"] == "warm and grassroots"


def test_save_preference_overwrites_existing(tmp_data_dir):
    from agent.memory import save_preference, load_profile
    save_preference("org", "voice", "corporate")
    save_preference("org", "voice", "grassroots")
    assert load_profile("org")["preferences"]["voice"] == "grassroots"


def test_get_ledger_empty_for_new_org(tmp_data_dir):
    from agent.memory import get_ledger
    assert get_ledger("new-org") == []


def test_trace_path_returns_correct_path(tmp_data_dir):
    from agent.memory import trace_path
    path = trace_path("my-org")
    assert path.parent == tmp_data_dir / "my-org"
    assert path.name == "trace.jsonl"
