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

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    total_cases INTEGER NOT NULL,
    completed_cases INTEGER NOT NULL,
    failed_cases INTEGER NOT NULL,
    total_cost_usd TEXT NOT NULL,
    mean_latency_ms REAL NOT NULL,
    mean_accuracy REAL NOT NULL,
    mean_groundedness REAL NOT NULL,
    mean_overall_score REAL NOT NULL,
    is_dry_run INTEGER NOT NULL DEFAULT 1,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    result_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    expected_answer TEXT,
    generated_output TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd TEXT NOT NULL,
    accuracy REAL NOT NULL,
    groundedness REAL NOT NULL,
    format_adherence REAL NOT NULL,
    overall_score REAL NOT NULL,
    evaluator_notes TEXT,
    FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id)
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

    def save_evaluation_run(self, run: Dict[str, Any]) -> None:
        """Saves a completed or aborted evaluation run header."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO evaluation_runs (
                    run_id, timestamp, model_id, dataset_path, total_cases,
                    completed_cases, failed_cases, total_cost_usd, mean_latency_ms,
                    mean_accuracy, mean_groundedness, mean_overall_score, is_dry_run, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    completed_cases = excluded.completed_cases,
                    failed_cases = excluded.failed_cases,
                    total_cost_usd = excluded.total_cost_usd,
                    mean_latency_ms = excluded.mean_latency_ms,
                    mean_accuracy = excluded.mean_accuracy,
                    mean_groundedness = excluded.mean_groundedness,
                    mean_overall_score = excluded.mean_overall_score,
                    notes = excluded.notes
                """,
                (
                    run["run_id"],
                    run["timestamp"],
                    run["model_id"],
                    run["dataset_path"],
                    run["total_cases"],
                    run["completed_cases"],
                    run["failed_cases"],
                    str(run["total_cost_usd"]),
                    float(run["mean_latency_ms"]),
                    float(run["mean_accuracy"]),
                    float(run["mean_groundedness"]),
                    float(run["mean_overall_score"]),
                    1 if run.get("is_dry_run", True) else 0,
                    run.get("notes", ""),
                ),
            )
            conn.commit()

    def save_evaluation_results(self, results: List[Dict[str, Any]]) -> None:
        """Batch inserts individual evaluation results."""
        with self.get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO evaluation_results (
                    result_id, run_id, case_id, prompt, expected_answer,
                    generated_output, latency_ms, prompt_tokens, completion_tokens,
                    cost_usd, accuracy, groundedness, format_adherence, overall_score,
                    evaluator_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(result_id) DO NOTHING
                """,
                [
                    (
                        r["result_id"],
                        r["run_id"],
                        r["case_id"],
                        r["prompt"],
                        r.get("expected_answer", ""),
                        r["generated_output"],
                        float(r["latency_ms"]),
                        int(r["prompt_tokens"]),
                        int(r["completion_tokens"]),
                        str(r["cost_usd"]),
                        float(r["accuracy"]),
                        float(r["groundedness"]),
                        float(r["format_adherence"]),
                        float(r["overall_score"]),
                        r.get("evaluator_notes", ""),
                    )
                    for r in results
                ],
            )
            conn.commit()

    def get_evaluation_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Fetches a specific evaluation run by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM evaluation_runs WHERE run_id = ?",
                (run_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_evaluation_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent evaluation runs sorted newest first."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM evaluation_runs ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_evaluation_results_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Returns all individual results for a run."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM evaluation_results WHERE run_id = ? ORDER BY result_id",
                (run_id,),
            )
            return [dict(row) for row in cursor.fetchall()]
