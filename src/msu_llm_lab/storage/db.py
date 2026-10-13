"""SQLite database initialization, migration, and transactional persistence."""

import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Dict, Any

from msu_llm_lab.core.config import settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS budget_ledger (
    transaction_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    run_id TEXT,
    model_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    amount_micros INTEGER NOT NULL,
    amount_usd TEXT NOT NULL,
    is_pilot INTEGER NOT NULL DEFAULT 1,
    is_preflight_estimate INTEGER NOT NULL DEFAULT 0,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_portfolio (
    model_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    provider TEXT NOT NULL,
    deployment_mode TEXT NOT NULL,
    license TEXT NOT NULL,
    context_length INTEGER,
    input_price_per_1m_usd TEXT NOT NULL,
    output_price_per_1m_usd TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    experimental_purpose TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class DatabaseManager:
    """Manages SQLite connections, idempotent schemas, and budget ledger records."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path is not None else settings.db_path

    def get_connection(self) -> sqlite3.Connection:
        """Opens a connection with WAL mode and row factory."""
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        if str(self.db_path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Idempotently initializes tables without modifying existing data."""
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    def record_transaction(
        self,
        transaction_id: str,
        model_id: str,
        provider: str,
        amount_usd: Decimal,
        description: str,
        run_id: Optional[str] = None,
        is_pilot: bool = True,
        is_preflight_estimate: bool = False,
        timestamp: Optional[str] = None,
    ) -> None:
        """Records a new entry in the budget ledger using exact integer micros and Decimal string."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        
        amount_micros = int(amount_usd * 1_000_000)
        formatted_usd = f"{amount_usd:.6f}"

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO budget_ledger (
                    transaction_id, timestamp, run_id, model_id, provider,
                    amount_micros, amount_usd, is_pilot, is_preflight_estimate, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction_id,
                    timestamp,
                    run_id,
                    model_id,
                    provider,
                    amount_micros,
                    formatted_usd,
                    1 if is_pilot else 0,
                    1 if is_preflight_estimate else 0,
                    description,
                ),
            )
            conn.commit()

    def get_cumulative_spend(self, include_estimates: bool = False) -> tuple[int, Decimal]:
        """Returns total historical spend as (micros, Decimal)."""
        with self.get_connection() as conn:
            where_clause = "" if include_estimates else "WHERE is_preflight_estimate = 0"
            cursor = conn.execute(f"SELECT COALESCE(SUM(amount_micros), 0) FROM budget_ledger {where_clause}")
            row = cursor.fetchone()
            micros = row[0] if row else 0
            return micros, Decimal(micros) / Decimal("1000000")

    def get_pilot_spend(self, include_estimates: bool = False) -> tuple[int, Decimal]:
        """Returns spend attributed to pilot evaluation as (micros, Decimal)."""
        with self.get_connection() as conn:
            where_clause = "WHERE is_pilot = 1"
            if not include_estimates:
                where_clause += " AND is_preflight_estimate = 0"
            cursor = conn.execute(f"SELECT COALESCE(SUM(amount_micros), 0) FROM budget_ledger {where_clause}")
            row = cursor.fetchone()
            micros = row[0] if row else 0
            return micros, Decimal(micros) / Decimal("1000000")

    def get_recent_transactions(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Retrieves most recent ledger transactions."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT transaction_id, timestamp, run_id, model_id, provider,
                       amount_usd, amount_micros, is_pilot, is_preflight_estimate, description
                FROM budget_ledger
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def upsert_model(
        self,
        model_id: str,
        display_name: str,
        provider: str,
        deployment_mode: str,
        license_type: str,
        context_length: Optional[int],
        input_price_per_1m_usd: str,
        output_price_per_1m_usd: str,
        verification_status: str,
        experimental_purpose: str,
    ) -> None:
        """Inserts or updates a model in the local portfolio registry."""
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO model_portfolio (
                    model_id, display_name, provider, deployment_mode, license,
                    context_length, input_price_per_1m_usd, output_price_per_1m_usd,
                    verification_status, experimental_purpose, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    provider = excluded.provider,
                    deployment_mode = excluded.deployment_mode,
                    license = excluded.license,
                    context_length = excluded.context_length,
                    input_price_per_1m_usd = excluded.input_price_per_1m_usd,
                    output_price_per_1m_usd = excluded.output_price_per_1m_usd,
                    verification_status = excluded.verification_status,
                    experimental_purpose = excluded.experimental_purpose
                """,
                (
                    model_id,
                    display_name,
                    provider,
                    deployment_mode,
                    license_type,
                    context_length,
                    input_price_per_1m_usd,
                    output_price_per_1m_usd,
                    verification_status,
                    experimental_purpose,
                    now,
                ),
            )
            conn.commit()

    def get_all_models(self) -> List[Dict[str, Any]]:
        """Returns all registered models from the portfolio."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT model_id, display_name, provider, deployment_mode, license,
                       context_length, input_price_per_1m_usd, output_price_per_1m_usd,
                       verification_status, experimental_purpose
                FROM model_portfolio
                ORDER BY deployment_mode, model_id
                """
            )
            return [dict(row) for row in cursor.fetchall()]
