"""Privacy safeguards for classification datasets. The PUBLIC/SYNTHETIC gate is reused, not weakened."""

import hashlib
import re

from msu_llm_lab.providers.base import ConfidentialityViolationError
from msu_llm_lab.schemas.evaluation import DataClassification

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?6?0?1\d[\s-]?\d{3,4}[\s-]?\d{4}|\+?\d{9,13})(?!\d)")


def assert_dispatch_eligible(classification: DataClassification) -> None:
    """Fail closed: only PUBLIC and SYNTHETIC may ever leave the machine."""
    if classification not in (DataClassification.PUBLIC, DataClassification.SYNTHETIC):
        raise ConfidentialityViolationError(
            f"Classification '{classification.value}' is barred from outbound dispatch; only PUBLIC and SYNTHETIC are eligible."
        )


def assert_synthetic_fixture(classification: DataClassification) -> None:
    """Pipeline-design fixtures must be explicitly SYNTHETIC."""
    if classification != DataClassification.SYNTHETIC:
        raise ConfidentialityViolationError("Only SYNTHETIC cases may be used as pipeline fixtures")


def contains_contact_details(text: str) -> bool:
    return bool(_EMAIL.search(text) or _PHONE.search(text))


def redact_for_log(text: str) -> str:
    """Log-safe stand-in for student text: length plus a short hash, never the content."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"<redacted len={len(text)} sha={digest}>"


def assert_student_data_allowed(model_key: str) -> None:
    """Unknown or unverified data policy or route means no student-derived data may be sent to this model."""
    from msu_llm_lab.models.registry import get_record

    blockers = get_record(model_key).student_data_blockers()
    if blockers:
        raise ConfidentialityViolationError(
            f"Student-derived data may not be sent to '{model_key}': unresolved {', '.join(blockers)}."
        )
