"""Evaluation dataset manager for JSONL reading, writing, partitioning, and validation."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from msu_llm_lab.dataset.leakage import LeakageDetector, LeakageReport
from msu_llm_lab.schemas.evaluation import (
    DataClassification,
    DatasetSplit,
    EvaluationCase,
    VerificationStatus,
)


class DatasetValidationSummary(BaseModel):
    """Statistical summary of dataset validation."""
    source_path: str
    total_cases: int
    train_count: int
    dev_count: int
    test_count: int
    unassigned_count: int
    classification_breakdown: Dict[str, int]
    verification_breakdown: Dict[str, int]
    external_eligible_count: int
    blocked_count: int
    leakage_report: Optional[LeakageReport] = None
    is_valid: bool = True
    validation_errors: List[str] = Field(default_factory=list)


class DatasetManager:
    """Manages loading, serialization, train/dev/test partitioning, and validation of datasets."""

    @staticmethod
    def load_jsonl(path: Path) -> List[EvaluationCase]:
        """Loads and validates evaluation cases from a JSONL file."""
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        cases: List[EvaluationCase] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("#"):
                    continue
                try:
                    payload = json.loads(clean_line)
                    cases.append(EvaluationCase.model_validate(payload))
                except Exception as e:
                    raise ValueError(f"Line {line_idx} failed schema validation: {e}") from e

        return cases

    @staticmethod
    def save_jsonl(cases: List[EvaluationCase], path: Path) -> None:
        """Serializes evaluation cases to a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for case in cases:
                line = case.model_dump_json()
                f.write(f"{line}\n")

    @staticmethod
    def partition(
        cases: List[EvaluationCase],
        train_ratio: float = 0.70,
        dev_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[DatasetSplit, List[EvaluationCase]]:
        """Stratifies or randomly partitions cases into TRAIN, DEV, and TEST splits."""
        if round(train_ratio + dev_ratio + test_ratio, 2) != 1.00:
            raise ValueError("Partition ratios must sum to 1.00.")

        rng = random.Random(seed)
        shuffled = list(cases)
        rng.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        dev_end = train_end + int(n * dev_ratio)

        train_cases = shuffled[:train_end]
        dev_cases = shuffled[train_end:dev_end]
        test_cases = shuffled[dev_end:]

        for c in train_cases:
            c.split = DatasetSplit.TRAIN
        for c in dev_cases:
            c.split = DatasetSplit.DEV
        for c in test_cases:
            c.split = DatasetSplit.TEST

        return {
            DatasetSplit.TRAIN: train_cases,
            DatasetSplit.DEV: dev_cases,
            DatasetSplit.TEST: test_cases,
        }

    def validate_dataset(
        self, cases: List[EvaluationCase], source_path: str = "", check_leakage: bool = True
    ) -> DatasetValidationSummary:
        """Validates schema invariants, confidentiality boundaries, and cross-split leakage."""
        errors: List[str] = []

        total = len(cases)
        train_cnt = sum(1 for c in cases if c.split == DatasetSplit.TRAIN)
        dev_cnt = sum(1 for c in cases if c.split == DatasetSplit.DEV)
        test_cnt = sum(1 for c in cases if c.split == DatasetSplit.TEST)
        unassigned_cnt = sum(1 for c in cases if c.split == DatasetSplit.UNASSIGNED)

        classifications: Dict[str, int] = {}
        for c in cases:
            k = c.classification.value
            classifications[k] = classifications.get(k, 0) + 1

        verifications: Dict[str, int] = {}
        for c in cases:
            k = c.verification.status.value
            verifications[k] = verifications.get(k, 0) + 1

        eligible_cnt = sum(1 for c in cases if c.is_eligible_for_external_dispatch)
        blocked_cnt = total - eligible_cnt

        # Check unique IDs
        ids = [c.id for c in cases]
        if len(ids) != len(set(ids)):
            duplicates = [x for x in ids if ids.count(x) > 1]
            errors.append(f"Duplicate case IDs found: {set(duplicates)}")

        # Cross-split leakage check
        leak_report = None
        if check_leakage and (train_cnt > 0 or dev_cnt > 0 or test_cnt > 0):
            detector = LeakageDetector()
            leak_report = detector.analyze(cases)
            if leak_report.has_leakage:
                errors.append(
                    f"Cross-split leakage detected: {len(leak_report.findings)} contaminating case pairs."
                )

        is_valid = len(errors) == 0

        return DatasetValidationSummary(
            source_path=source_path,
            total_cases=total,
            train_count=train_cnt,
            dev_count=dev_cnt,
            test_count=test_cnt,
            unassigned_count=unassigned_cnt,
            classification_breakdown=classifications,
            verification_breakdown=verifications,
            external_eligible_count=eligible_cnt,
            blocked_count=blocked_cnt,
            leakage_report=leak_report,
            is_valid=is_valid,
            validation_errors=errors,
        )
