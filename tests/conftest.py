"""Pytest fixtures for msu-llm-lab test suite."""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_kb_dir(tmp_path: Path) -> Path:
    """Provides a temporary directory for isolated KB test fixtures."""
    kb_dir = tmp_path / "kbs"
    kb_dir.mkdir(parents=True, exist_ok=True)
    return kb_dir


@pytest.fixture(autouse=True)
def _isolate_local_state(tmp_path_factory, monkeypatch):
    """Tests never read or write the real ledger/database under data/: point the settings at a private temp location."""
    from msu_llm_lab.core.config import settings
    base = tmp_path_factory.mktemp("isolated_state")
    monkeypatch.setattr(settings, "data_dir", base / "data")
    monkeypatch.setattr(settings, "reports_dir", base / "reports")
