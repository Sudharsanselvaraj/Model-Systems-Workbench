"""Model portfolio registry and candidate model metadata."""

from decimal import Decimal
from typing import List, Optional
from pydantic import BaseModel, Field

from msu_llm_lab.storage.db import DatabaseManager


class ModelCandidate(BaseModel):
    """Metadata specification for an evaluation model candidate."""
    model_id: str = Field(description="Unique provider model identifier (e.g. 'qwen/qwen-2.5-7b-instruct')")
    display_name: str = Field(description="Human-readable model name")
    provider: str = Field(description="Upstream model vendor or aggregator")
    deployment_mode: str = Field(description="HOSTED_OPENROUTER, HOSTED_DIRECT, or SELF_HOSTED_VLLM")
    license: str = Field(description="License type (e.g. Apache-2.0, Meta-Llama, Proprietary)")
    context_length: Optional[int] = Field(default=None, description="Max token context window")
    input_price_per_1m_usd: Decimal = Field(description="Estimated cost per 1M input tokens")
    output_price_per_1m_usd: Decimal = Field(description="Estimated cost per 1M output tokens")
    verification_status: str = Field(
        default="UNVERIFIED_CATALOG_ENTRY",
        description="Verification state against live provider API catalog",
    )
    experimental_purpose: str = Field(description="Role and rationale for inclusion in research workbench")


DEFAULT_PORTFOLIO: List[ModelCandidate] = [
    ModelCandidate(
        model_id="google/gemini-3.5-flash-lite",
        display_name="Gemini 3.5 Flash Lite",
        provider="google",
        deployment_mode="HOSTED_DIRECT",
        license="Proprietary Commercial API",
        context_length=1000000,
        input_price_per_1m_usd=Decimal("0.0750"),
        output_price_per_1m_usd=Decimal("0.3000"),
        verification_status="UNVERIFIED_CATALOG_ENTRY",
        experimental_purpose="High-throughput multimodal baseline for student counsellor triage.",
    ),
    ModelCandidate(
        model_id="anthropic/claude-haiku-5.5",
        display_name="Claude Haiku 5.5 (Proposed)",
        provider="anthropic",
        deployment_mode="HOSTED_OPENROUTER",
        license="Proprietary Commercial API",
        context_length=200000,
        input_price_per_1m_usd=Decimal("0.8000"),
        output_price_per_1m_usd=Decimal("4.0000"),
        verification_status="UNVERIFIED_CATALOG_ENTRY",
        experimental_purpose="Hosted fast candidate (requires validation against live OpenRouter ID).",
    ),
    ModelCandidate(
        model_id="openai/gpt-6-luna",
        display_name="GPT-6 Luna (Proposed)",
        provider="openai",
        deployment_mode="HOSTED_OPENROUTER",
        license="Proprietary Commercial API",
        context_length=128000,
        input_price_per_1m_usd=Decimal("0.1500"),
        output_price_per_1m_usd=Decimal("0.6000"),
        verification_status="UNVERIFIED_CATALOG_ENTRY",
        experimental_purpose="Hosted quality ceiling reference (requires validation against live OpenRouter ID).",
    ),
    ModelCandidate(
        model_id="qwen/qwen3-8b",
        display_name="Qwen3-8B (Proposed)",
        provider="qwen",
        deployment_mode="HOSTED_OPENROUTER",
        license="Open-Weight (License TBD)",
        context_length=32768,
        input_price_per_1m_usd=Decimal("0.2000"),
        output_price_per_1m_usd=Decimal("0.2000"),
        verification_status="UNVERIFIED_CATALOG_ENTRY",
        experimental_purpose="Primary open-weight candidate for eventual self-hosted vLLM serving.",
    ),
    ModelCandidate(
        model_id="qwen/qwen3-14b",
        display_name="Qwen3-14B (Proposed)",
        provider="qwen",
        deployment_mode="HOSTED_OPENROUTER",
        license="Open-Weight (License TBD)",
        context_length=32768,
        input_price_per_1m_usd=Decimal("0.3500"),
        output_price_per_1m_usd=Decimal("0.4000"),
        verification_status="UNVERIFIED_CATALOG_ENTRY",
        experimental_purpose="Secondary larger open-weight candidate for capacity comparison.",
    ),
]


def seed_default_portfolio(db: Optional[DatabaseManager] = None) -> None:
    """Seeds default initial model portfolio into SQLite registry idempotently."""
    db_mgr = db or DatabaseManager()
    db_mgr.init_db()
    for candidate in DEFAULT_PORTFOLIO:
        db_mgr.upsert_model(
            model_id=candidate.model_id,
            display_name=candidate.display_name,
            provider=candidate.provider,
            deployment_mode=candidate.deployment_mode,
            license_type=candidate.license,
            context_length=candidate.context_length,
            input_price_per_1m_usd=f"{candidate.input_price_per_1m_usd:.4f}",
            output_price_per_1m_usd=f"{candidate.output_price_per_1m_usd:.4f}",
            verification_status=candidate.verification_status,
            experimental_purpose=candidate.experimental_purpose,
        )


def get_portfolio(db: Optional[DatabaseManager] = None) -> List[ModelCandidate]:
    """Retrieves all registered models from the local database."""
    db_mgr = db or DatabaseManager()
    db_mgr.init_db()
    rows = db_mgr.get_all_models()
    if not rows:
        seed_default_portfolio(db_mgr)
        rows = db_mgr.get_all_models()
    else:
        existing_ids = [r["model_id"] for r in rows]
        if "google/gemini-flash-1.5" in existing_ids and "google/gemini-3.5-flash-lite" not in existing_ids:
            with db_mgr.get_connection() as conn:
                conn.execute("DELETE FROM model_portfolio WHERE model_id = 'google/gemini-flash-1.5'")
            seed_default_portfolio(db_mgr)
            rows = db_mgr.get_all_models()

    candidates = []
    for r in rows:
        candidates.append(
            ModelCandidate(
                model_id=r["model_id"],
                display_name=r["display_name"],
                provider=r["provider"],
                deployment_mode=r["deployment_mode"],
                license=r["license"],
                context_length=r["context_length"],
                input_price_per_1m_usd=Decimal(r["input_price_per_1m_usd"]),
                output_price_per_1m_usd=Decimal(r["output_price_per_1m_usd"]),
                verification_status=r["verification_status"],
                experimental_purpose=r["experimental_purpose"],
            )
        )
    return candidates
