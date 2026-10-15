"""Application configuration and budget policy definitions."""

import hashlib
import os
from decimal import Decimal
from pathlib import Path
from typing import Optional, Tuple
from pydantic import BaseModel, Field


class BudgetPolicy(BaseModel):
    """Local budget tracking policies.
    
    NOTE: These are local experiment governance ceilings tracked within this
    workbench, NOT provider-enforced account balance limits.
    """
    total_budget_usd: Decimal = Field(default=Decimal("10.00"), description="Total lifetime experiment ceiling")
    pilot_budget_usd: Decimal = Field(default=Decimal("2.00"), description="Ceiling for initial pilot evaluation")
    per_run_ceiling_usd: Decimal = Field(default=Decimal("0.50"), description="Max spend allowed per single evaluation run")
    per_case_ceiling_usd: Decimal = Field(default=Decimal("0.02"), description="Max spend allowed per individual evaluation case")
    retry_buffer_pct: Decimal = Field(default=Decimal("0.05"), description="Buffer for network/rate-limit retries (5%)")

    @property
    def total_budget_micros(self) -> int:
        return int(self.total_budget_usd * 1_000_000)

    @property
    def pilot_budget_micros(self) -> int:
        return int(self.pilot_budget_usd * 1_000_000)

    @property
    def per_run_ceiling_micros(self) -> int:
        return int(self.per_run_ceiling_usd * 1_000_000)

    @property
    def per_case_ceiling_micros(self) -> int:
        return int(self.per_case_ceiling_usd * 1_000_000)


class Settings(BaseModel):
    """Global workbench environment and filesystem settings."""
    app_name: str = "msu-llm-lab"
    workspace_root: Path = Field(default_factory=lambda: Path(os.getcwd()))
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    db_filename: str = "workbench.db"
    reports_dir: Path = Field(default_factory=lambda: Path("reports"))
    
    budget: BudgetPolicy = Field(default_factory=BudgetPolicy)

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    def ensure_directories(self) -> None:
        """Ensures local storage directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_DOTENV_LOADED_PATH: Optional[Path] = None
_DOTENV_KEY_HASH: Optional[str] = None
_KEY_ORIGINALLY_IN_SHELL: bool = False


def ensure_secure_dotenv(dotenv_path: Optional[Path] = None) -> None:
    """Ensure .env has restrictive permissions (0600) on POSIX systems."""
    path = dotenv_path or (PROJECT_ROOT / ".env")
    if path.is_file() and hasattr(os, "chmod"):
        try:
            path.chmod(0o600)
        except OSError:
            pass


DEFAULT_DOTENV_PATH: Optional[Path] = None


def load_project_env(
    dotenv_path: Optional[Path] = None,
    override: bool = False,
) -> Tuple[bool, Optional[Path]]:
    """Load environment variables from project .env file if it exists.
    
    Predictably resolves to PROJECT_ROOT / '.env' (or Path.cwd() / '.env')
    unless a specific path is given. Does NOT search arbitrary parent directories.
    Preserves explicitly exported shell variables (override=False by default).
    Honors MSU_LLM_SKIP_DOTENV=1 for offline test isolation.
    
    Returns:
        (loaded: bool, resolved_path: Optional[Path])
    """
    global _DOTENV_LOADED_PATH, _DOTENV_KEY_HASH, _KEY_ORIGINALLY_IN_SHELL

    if os.environ.get("MSU_LLM_SKIP_DOTENV") == "1" and dotenv_path is None:
        return False, None

    target = dotenv_path or DEFAULT_DOTENV_PATH
    if target is None:
        env_override = os.environ.get("DOTENV_PATH")
        if env_override:
            target = Path(env_override)
        elif (PROJECT_ROOT / ".env").is_file():
            target = PROJECT_ROOT / ".env"
        elif (Path.cwd() / ".env").is_file():
            target = Path.cwd() / ".env"
        else:
            target = PROJECT_ROOT / ".env"

    shell_key = os.environ.get("OPENROUTER_API_KEY")
    _KEY_ORIGINALLY_IN_SHELL = bool(shell_key and shell_key.strip())

    if target.is_file():
        ensure_secure_dotenv(target)
        from dotenv import load_dotenv
        loaded = load_dotenv(dotenv_path=target, override=override)
        _DOTENV_LOADED_PATH = target
        current_key = os.environ.get("OPENROUTER_API_KEY")
        if current_key and current_key.strip() and not _KEY_ORIGINALLY_IN_SHELL:
            _DOTENV_KEY_HASH = hashlib.sha256(current_key.strip().encode()).hexdigest()
        return loaded, target

    return False, None


def get_api_key_source() -> str:
    """Report the provenance of OPENROUTER_API_KEY safely without exposing any secret."""
    raw = os.environ.get("OPENROUTER_API_KEY")
    if not (raw and raw.strip()):
        return "none (checked shell environment and .env)"

    curr_hash = hashlib.sha256(raw.strip().encode()).hexdigest()
    if _DOTENV_KEY_HASH and curr_hash == _DOTENV_KEY_HASH and not _KEY_ORIGINALLY_IN_SHELL:
        source_name = _DOTENV_LOADED_PATH.name if _DOTENV_LOADED_PATH else ".env"
        return f".env file ({source_name})"

    if _KEY_ORIGINALLY_IN_SHELL:
        return "environment variable (shell export; takes precedence over .env)"
    return "environment variable (shell export)"


# Shared global configuration instance
settings = Settings()

# Automatically load project .env on configuration initialization
load_project_env()
