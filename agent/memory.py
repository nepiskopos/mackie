"""
Persistent storage for org profiles, post ledgers, and preferences.

No database is used at this assignment stage. All org state is stored as
JSON on disk at data/{org_id}/profile.json. This is intentional: the
assignment requires a non-engineer to read and edit what the assistant has
learned, and a plain JSON file in any text editor satisfies that requirement
without a database setup or migration scripts.

Consequence: the load-modify-save pattern used by save_preference, add_post,
and update_post_status is not atomic. Two concurrent sessions writing to the
same org profile will race and one write will silently overwrite the other.
This is acceptable under the assignment's single-user assumption. A
production deployment would replace profile.json with a PostgreSQL table and
use row-level locking.

Owns: reading and writing profile.json for each org, assigning post IDs,
timestamping entries, and returning file paths for trace output.
Does not own LLM interaction, tool dispatch, or research.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_DIR: Path = Path(__file__).parent.parent / "data"


def slugify(text: str) -> str:
    """Convert a display name to a filesystem-safe org ID."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _org_dir(org_id: str) -> Path:
    """Return (and create) the per-org data directory."""
    d = DATA_DIR / org_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _profile_path(org_id: str) -> Path:
    """Return the path to the org's profile.json file."""
    return _org_dir(org_id) / "profile.json"


def load_profile(org_id: str) -> dict[str, Any]:
    """
    Load the org profile from disk, returning a default skeleton if none exists.

    Args:
        org_id: Filesystem-safe org identifier.

    Returns:
        Dict with keys: org_id, name, website, research, preferences, post_ledger.
    """
    path = _profile_path(org_id)
    if path.exists():
        return json.loads(path.read_text())
    return {
        "org_id": org_id,
        "name": "",
        "website": "",
        "research": None,
        "preferences": {},
        "post_ledger": [],
    }


def save_profile(org_id: str, profile: dict[str, Any]) -> None:
    """Write the org profile to disk, overwriting any existing file.

    No concurrency control: this is a plain file write with no locking.
    Concurrent calls for the same org_id will race. Acceptable at this
    assignment stage (single-user assumption); a production deployment
    would use a database with row-level locking.

    Args:
        org_id: Filesystem-safe org identifier.
        profile: Full profile dict as returned by load_profile.
    """
    _profile_path(org_id).write_text(json.dumps(profile, indent=2))


def save_preference(org_id: str, key: str, value: Any) -> None:
    """Persist a single brand preference key/value pair for the org.

    Uses a load-modify-save pattern. Not atomic — see module docstring.

    Args:
        org_id: Filesystem-safe org identifier.
        key:    Preference name (e.g. "voice", "banned_topics").
        value:  Preference value — any JSON-serialisable type.
    """
    profile = load_profile(org_id)
    profile["preferences"][key] = value
    save_profile(org_id, profile)


def add_post(org_id: str, post: dict[str, Any]) -> str:
    """
    Append a post to the org's ledger, assigning an ID and timestamp.

    Uses a load-modify-save pattern. Not atomic — see module docstring.

    Args:
        org_id: Filesystem-safe org identifier.
        post:   Post dict with at minimum a 'platform' and 'content' key.
                'status' defaults to "suggested" if not provided.

    Returns:
        The assigned post ID string (e.g. "post_001").
    """
    profile = load_profile(org_id)
    post_id = f"post_{len(profile['post_ledger']) + 1:03d}"
    post["id"] = post_id
    post["created_at"] = datetime.now().isoformat()
    post.setdefault("status", "suggested")
    profile["post_ledger"].append(post)
    save_profile(org_id, profile)
    return post_id


def update_post_status(org_id: str, post_id: str, status: str) -> bool:
    """
    Update the status of an existing ledger entry.

    Args:
        org_id:  Filesystem-safe org identifier.
        post_id: ID of the post to update (e.g. "post_001").
        status:  New status string ("suggested", "draft", "approved", "posted", "planned").

    Returns:
        True if the post was found and updated; False if post_id does not exist.
    """
    profile = load_profile(org_id)
    for post in profile["post_ledger"]:
        if post["id"] == post_id:
            post["status"] = status
            save_profile(org_id, profile)
            return True
    return False


def get_ledger(org_id: str) -> list[dict[str, Any]]:
    """Return the full post ledger list for an org.

    Args:
        org_id: Filesystem-safe org identifier.

    Returns:
        List of post dicts; empty list if no posts have been saved.
    """
    return load_profile(org_id)["post_ledger"]


def trace_path(org_id: str) -> Path:
    """Return the path to the JSONL tool-call trace file for an org."""
    return _org_dir(org_id) / "trace.jsonl"
