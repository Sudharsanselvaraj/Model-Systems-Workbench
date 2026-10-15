"""Offline classification-benchmark foundations (input_contract_v1). No network, no model adapters.

PRODUCTION PARITY IS NOT ESTABLISHED for anything in this package.
"""

from msu_llm_lab.classification.contracts import (
    CONTRACT_VERSION,
    PARITY_STATUS,
    AmbiguityRecord,
    ExclusionReason,
    ExtractedAction,
    HistoryStatus,
    IntentCase,
    IntentLabel,
    LeadCase,
    ReviewStatus,
    Role,
    Turn,
)
from msu_llm_lab.classification.validators import validate_intent_output, validate_lead_output

__all__ = [
    "CONTRACT_VERSION", "PARITY_STATUS", "AmbiguityRecord", "ExclusionReason", "ExtractedAction", "HistoryStatus",
    "IntentCase", "IntentLabel", "LeadCase", "ReviewStatus", "Role", "Turn",
    "validate_intent_output", "validate_lead_output",
]
