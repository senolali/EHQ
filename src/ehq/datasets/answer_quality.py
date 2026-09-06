"""Conservative machine gates for unusable PCQ gold answers."""

from __future__ import annotations

import re
from typing import Tuple

from ..evaluation.correctness import contains_normalized_phrase, normalize_text


_SELF_REFERENTIAL_DURATION = re.compile(
    r"^(?:since|from)\s+(?:its|their|the|that|this)\s+"
    r"(?:founding|beginning|creation|establishment|inception|start)$",
    flags=re.IGNORECASE,
)
_NONANSWER = re.compile(
    r"^(?:unknown|not\s+(?:specified|provided|stated)|n\s*/?\s*a|"
    r"various|multiple|several|the\s+same|as\s+above)$",
    flags=re.IGNORECASE,
)


def pcq_answer_quality_issues(question: str, gold_answer: str) -> Tuple[str, ...]:
    """Return high-precision defects; semantic review remains mandatory."""

    question = " ".join(str(question or "").split())
    gold = " ".join(str(gold_answer or "").split())
    issues = []
    if not normalize_text(gold):
        issues.append("missing_gold_answer")
        return tuple(issues)
    if _SELF_REFERENTIAL_DURATION.fullmatch(gold):
        issues.append("self_referential_duration_answer")
    if _NONANSWER.fullmatch(gold):
        issues.append("nonanswer_gold")
    if contains_normalized_phrase(question, gold):
        issues.append("gold_answer_leaked_in_question")
    return tuple(issues)
