"""Deterministic, maximally balanced experiment-item selection."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .constants import DATASET_CATEGORIES
from .hashing import sha256_json, sha256_text


def _item_id(item: Mapping[str, Any]) -> str:
    return str(item.get("question_id") or item.get("id") or "")


def _stratum(item: Mapping[str, Any]) -> str:
    category = str(item.get("category") or "").upper()
    subcategory = str(item.get("subcategory") or "UNSPECIFIED")
    return f"{category}::{subcategory}"


def select_items(
    items: Iterable[Mapping[str, Any]],
    *,
    categories: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    seed: int = 42,
) -> List[Mapping[str, Any]]:
    """Select a deterministic round-robin sample across available strata.

    Hash ranking removes dependence on source-file order. Round-robin allocation
    keeps category/subcategory representation as even as the requested limit
    permits.
    """

    requested = (
        tuple(dict.fromkeys(str(value).upper() for value in categories))
        if categories
        else DATASET_CATEGORIES
    )
    unknown = sorted(set(requested) - set(DATASET_CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown EHQ categories: {', '.join(unknown)}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    filtered = [
        item
        for item in items
        if str(item.get("category") or "").upper() in requested
    ]
    if limit is None or limit >= len(filtered):
        return sorted(filtered, key=lambda item: (_stratum(item), _item_id(item)))

    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in filtered:
        groups[_stratum(item)].append(item)
    for stratum, members in groups.items():
        members.sort(
            key=lambda item: sha256_text(f"{seed}\0{stratum}\0{_item_id(item)}")
        )
    strata = sorted(groups, key=lambda value: sha256_text(f"{seed}\0{value}"))

    selected: List[Mapping[str, Any]] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for stratum in strata:
            members = groups[stratum]
            if offset < len(members):
                selected.append(members[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return sorted(selected, key=lambda item: (_stratum(item), _item_id(item)))


def selection_manifest(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    categories = Counter(str(item.get("category") or "").upper() for item in items)
    strata = Counter(_stratum(item) for item in items)
    ids = [_item_id(item) for item in items]
    return {
        "n_selected": len(items),
        "category_counts": dict(sorted(categories.items())),
        "stratum_counts": dict(sorted(strata.items())),
        "question_ids_sha256": sha256_json(ids),
        "question_ids": ids,
    }
