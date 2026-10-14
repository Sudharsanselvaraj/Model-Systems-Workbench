"""Model execution schemas for inference tracking."""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ModelExecutionRecord(BaseModel):
    """Execution telemetry captured from candidate model inference."""
    execution_id: str = Field(description="Unique run ID (e.g. exec_20261014_001)")
    case_id: str = Field(description="Referenced EvaluationCase ID")
    model_id: str = Field(description="Candidate model identifier")
    prompt_version: str = Field(default="v1.0", description="Evaluation prompt template version")
    output_text: str = Field(description="Raw generated model response")
    latency_ms: float = Field(ge=0.0, description="End-to-end response latency in milliseconds")
    prompt_tokens: int = Field(ge=0, description="Token count for input prompt")
    completion_tokens: int = Field(ge=0, description="Token count for generated response")
    cost_usd: Decimal = Field(default=Decimal("0.0000"), ge=Decimal("0.0000"), description="Calculated inference cost")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deployment_mode: str = Field(default="HOSTED_DIRECT", description="Deployment environment (e.g. HOSTED_DIRECT, vLLM)")
    metadata: Dict[str, Any] = Field(default_factory=dict)
