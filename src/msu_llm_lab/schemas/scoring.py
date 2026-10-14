"""Scoring and evaluation metric schemas."""

from typing import Optional
from pydantic import BaseModel, Field, model_validator


class EvaluationScore(BaseModel):
    """Rubric-based evaluation scores for model output against ground truth."""
    score_id: str = Field(description="Unique evaluation score ID")
    execution_id: str = Field(description="Associated ModelExecutionRecord ID")
    accuracy: float = Field(ge=0.0, le=1.0, description="Factual correctness against verified reference")
    groundedness: float = Field(ge=0.0, le=1.0, description="Absence of hallucinated or external claims")
    format_adherence: float = Field(ge=0.0, le=1.0, description="Compliance with structured output requirements")
    overall_score: float = Field(ge=0.0, le=1.0, description="Weighted composite score")
    evaluator_type: str = Field(default="RUBRIC_HEURISTIC", description="Evaluation method (e.g. LLM_JUDGE, HUMAN_EXPERT, RUBRIC_HEURISTIC)")
    notes: Optional[str] = Field(default=None, description="Diagnostic notes or justification")

    @model_validator(mode="before")
    @classmethod
    def calculate_default_overall(cls, data: dict) -> dict:
        if isinstance(data, dict) and "overall_score" not in data:
            acc = float(data.get("accuracy", 0.0))
            grd = float(data.get("groundedness", 0.0))
            fmt = float(data.get("format_adherence", 0.0))
            # Default weighting: 50% accuracy, 35% groundedness, 15% format
            data["overall_score"] = round(0.50 * acc + 0.35 * grd + 0.15 * fmt, 4)
        return data
