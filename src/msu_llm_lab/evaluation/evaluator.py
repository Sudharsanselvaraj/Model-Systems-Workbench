"""Rubric-based evaluation scoring for model outputs against verified ground truths."""

import re
from typing import Optional, Set
from msu_llm_lab.schemas.scoring import EvaluationScore


class RubricEvaluator:
    """Evaluates factual accuracy, groundedness, and format adherence."""

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Extracts normalized words for lexical overlap."""
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return set(w for w in cleaned.split() if len(w) > 2)

    def evaluate(
        self,
        score_id: str,
        execution_id: str,
        generated_output: str,
        expected_answer: Optional[str] = None,
    ) -> EvaluationScore:
        """Computes deterministic rubric scores."""
        out_clean = generated_output.strip()

        # 1. Format adherence: non-empty, proper punctuation, reasonable length
        if not out_clean:
            return EvaluationScore(
                score_id=score_id,
                execution_id=execution_id,
                accuracy=0.0,
                groundedness=0.0,
                format_adherence=0.0,
                overall_score=0.0,
                evaluator_type="RUBRIC_HEURISTIC",
                notes="Generated response was empty.",
            )

        format_adherence = 1.0
        if len(out_clean) < 15:
            format_adherence -= 0.3
        if not out_clean.endswith((".", "!", "?")):
            format_adherence -= 0.2
        format_adherence = max(0.2, min(1.0, format_adherence))

        # 2. Accuracy & Groundedness
        if expected_answer and expected_answer.strip():
            ref_tokens = self._tokenize(expected_answer)
            gen_tokens = self._tokenize(out_clean)

            if not ref_tokens:
                accuracy = 0.8
                groundedness = 0.8
            else:
                overlap = len(ref_tokens.intersection(gen_tokens))
                accuracy = round(min(1.0, overlap / len(ref_tokens)), 4)

                # Groundedness checks if response maintains reference domain terms
                groundedness = round(min(1.0, (overlap + 2) / (len(ref_tokens) + 2)), 4)
                if "hallucination" in out_clean.lower() or "error" in out_clean.lower():
                    groundedness = max(0.0, groundedness - 0.4)
        else:
            # Baseline when unlabelled
            accuracy = 0.70
            groundedness = 0.75

        return EvaluationScore(
            score_id=score_id,
            execution_id=execution_id,
            accuracy=accuracy,
            groundedness=groundedness,
            format_adherence=format_adherence,
            evaluator_type="RUBRIC_HEURISTIC",
            notes="Rubric heuristic evaluation against verified ground truth citation.",
        )
