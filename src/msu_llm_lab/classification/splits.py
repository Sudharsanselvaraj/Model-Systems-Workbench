"""Deterministic, group-level (leakage-safe) split assignment. `user4` is never accepted as a key."""

import hashlib
from typing import Dict, Iterable, List, Mapping, Sequence, Set

from msu_llm_lab.schemas.evaluation import DatasetSplit

FORBIDDEN_KEYS = {"user4"}


class SplitError(ValueError):
    pass


def assert_no_user4(record_keys: Iterable[str]) -> None:
    """Reject any structure carrying the phone-prefix column."""
    bad = FORBIDDEN_KEYS.intersection(k.lower() for k in record_keys)
    if bad:
        raise SplitError(f"Forbidden sensitive column(s) present: {sorted(bad)}")


def _rank(group: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{group}".encode("utf-8")).hexdigest()


def assign_group_splits(
    group_ids: Sequence[str],
    ratios: Mapping[DatasetSplit, float],
    seed: str,
) -> Dict[str, DatasetSplit]:
    """Map each distinct group to one split. Same inputs and seed always give the same result."""
    if not seed:
        raise SplitError("A seed is required and must be stored with the dataset manifest")
    if any(not g for g in group_ids):
        raise SplitError("Every case needs a non-empty group_id")
    if not ratios or abs(sum(ratios.values()) - 1.0) > 1e-9 or any(r < 0 for r in ratios.values()):
        raise SplitError("ratios must be non-negative and sum to 1.0")
    if DatasetSplit.UNASSIGNED in ratios:
        raise SplitError("UNASSIGNED is not an assignable split")
    groups = sorted(set(group_ids), key=lambda g: (_rank(g, seed), g))
    n = len(groups)
    out: Dict[str, DatasetSplit] = {}
    items = list(ratios.items())
    cumulative = 0.0
    start = 0
    for i, (split, r) in enumerate(items):
        cumulative += r
        end = n if i == len(items) - 1 else round(cumulative * n)
        for g in groups[start:end]:
            out[g] = split
        start = end
    return out


def find_group_leaks(case_groups: Mapping[str, str], case_splits: Mapping[str, DatasetSplit]) -> List[str]:
    """Return group ids that appear in more than one split (empty list means no leakage)."""
    seen: Dict[str, Set[DatasetSplit]] = {}
    for case_id, group in case_groups.items():
        seen.setdefault(group, set()).add(case_splits[case_id])
    return sorted(g for g, s in seen.items() if len(s) > 1)


def assert_views_disjoint(natural_groups: Iterable[str], challenge_groups: Iterable[str]) -> None:
    """The natural-distribution and rare-class challenge sets must not share groups."""
    overlap = set(natural_groups) & set(challenge_groups)
    if overlap:
        raise SplitError(f"{len(overlap)} group(s) appear in both evaluation views")
