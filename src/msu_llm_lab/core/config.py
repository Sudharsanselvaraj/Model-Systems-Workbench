"""Application configuration and budget policy definitions."""

import os
from decimal import Decimal
from pathlib import Path
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


# Shared global configuration instance
settings = Settings()
