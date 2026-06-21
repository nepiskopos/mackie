"""
Shared pytest fixtures for the Mackie test suite.

Owns: the tmp_data_dir fixture that redirects disk I/O to a temp directory
so tests never touch real org data or the production SQLite checkpoint database.
Does not own test logic — that lives in the individual test_*.py files.
"""
import pytest
from unittest.mock import patch


@pytest.fixture
def tmp_data_dir(tmp_path):
    """
    Redirect DATA_DIR and CHECKPOINTS_DB to temp paths so tests never touch
    real org data or the real SQLite checkpoint database.
    """
    with patch("agent.memory.DATA_DIR", tmp_path), \
         patch("agent.graph.CHECKPOINTS_DB", tmp_path / "checkpoints.db"):
        yield tmp_path
