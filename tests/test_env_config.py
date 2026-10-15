"""Tests for secure .env configuration and OpenRouter API key loading."""

import os
import stat
import urllib.request
from pathlib import Path
import pytest
from typer.testing import CliRunner

from msu_llm_lab.cli import app
from msu_llm_lab.core.config import (
    ensure_secure_dotenv,
    get_api_key_source,
    load_project_env,
)


def test_load_project_env_from_file(tmp_path: Path, monkeypatch):
    """A key can be loaded from a designated .env file."""
    fake_key = "sk-or-v1-TESTKEYFROMDOTENV1234567890"
    env_file = tmp_path / ".env"
    env_file.write_text(f"OPENROUTER_API_KEY={fake_key}\n")
    
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    loaded, target = load_project_env(dotenv_path=env_file)
    
    assert loaded is True
    assert target == env_file
    assert os.environ.get("OPENROUTER_API_KEY") == fake_key
    assert ".env file" in get_api_key_source()


def test_shell_env_precedence_over_dotenv(tmp_path: Path, monkeypatch):
    """An explicitly exported environment variable takes precedence over the .env value."""
    shell_key = "sk-or-v1-SHELLKEYPRECEDENCE9999999"
    dotenv_key = "sk-or-v1-DOTENVKEYSHOULDNOTOVERWRITE"
    env_file = tmp_path / ".env"
    env_file.write_text(f"OPENROUTER_API_KEY={dotenv_key}\n")
    
    monkeypatch.setenv("OPENROUTER_API_KEY", shell_key)
    loaded, target = load_project_env(dotenv_path=env_file, override=False)
    
    # Value must remain the shell key
    assert os.environ.get("OPENROUTER_API_KEY") == shell_key
    source = get_api_key_source()
    assert "shell export" in source


def test_empty_or_missing_key_reported_safely(tmp_path: Path, monkeypatch):
    """Missing or empty keys are handled cleanly without exceptions or crashes."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=\n")
    
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    loaded, target = load_project_env(dotenv_path=env_file)
    
    assert os.environ.get("OPENROUTER_API_KEY") in (None, "")
    assert "none" in get_api_key_source()


def test_ensure_secure_dotenv_permissions(tmp_path: Path):
    """Ensure .env file permissions are set to 0600 where supported."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=test\n")
    env_file.chmod(0o644)
    
    ensure_secure_dotenv(env_file)
    mode = stat.S_IMODE(env_file.stat().st_mode)
    assert mode == 0o600


def test_api_preflight_loads_dotenv_and_never_leaks_secret(tmp_path: Path, monkeypatch):
    """API preflight detects key from .env without printing or leaking the secret."""
    sentinel_secret = "sk-or-v1-SECRETTESTVALUE9876543210"
    env_file = tmp_path / ".env"
    env_file.write_text(f"OPENROUTER_API_KEY={sentinel_secret}\n")
    
    # Block any network calls
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network touched"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    
    # Load from the tmp env
    load_project_env(dotenv_path=env_file)
    
    runner = CliRunner(env={"COLUMNS": "200"})
    result = runner.invoke(app, ["api-preflight"])
    
    assert result.exit_code == 0
    assert "yes" in result.output
    # Secret itself must NEVER appear in output
    assert sentinel_secret not in result.output
    assert "SECRETTESTVALUE" not in result.output
    assert "sk-or" not in result.output


def test_missing_credentials_never_touch_network(monkeypatch):
    """When credentials are missing, CLI commands fail closed without touching the network."""
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network touched"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    
    runner = CliRunner(env={"COLUMNS": "200"})
    result = runner.invoke(app, ["api-preflight"])
    
    assert result.exit_code == 1
    assert "no" in result.output
