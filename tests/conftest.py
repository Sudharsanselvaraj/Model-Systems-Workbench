"""Pytest fixtures for msu-llm-lab test suite."""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_kb_dir(tmp_path: Path) -> Path:
    """Provides a temporary directory for isolated KB test fixtures."""
    kb_dir = tmp_path / "kbs"
    kb_dir.mkdir(parents=True, exist_ok=True)
    return kb_dir
