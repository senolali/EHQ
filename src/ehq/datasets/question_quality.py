"""Deterministic question-form quality gates used by PCQ production."""

from __future__ import annotations

import re
from typing import Tuple


_QUESTION_START = re.compile(
    r"^\s*(?:who|what|which|when|where|why|how|is|are|was|were|did|does|do|"
    r"has|have|can|could|would|will)\b",
    flags=re.IGNORECASE,
)
_QUESTION_CLAUSE = re.compile(
    r"\b(?:who|what|which|when|where|why|how|is|are|was|were|did|does|do|"
    r"has|have|can|could|would|will)\b",
    flags=re.IGNORECASE,
)
_IMPERATIVE_REQUEST = re.compile(
    r"(?:^|,\s*)(?:report|name|identify|state|give|provide|list)\b",
    flags=re.IGNORECASE,
)

_NESTED_TEMPLATE_TARGET = re.compile(
    r"(?:\bwhat\s+was|"
    r"\bwhat\s+did\b.{0,100}?\breport\s+as|"
    r"\bwhat\s+does\b.{0,100}?\bidentify\s+as|"
    r"\bwhat\s+does\b.{0,100}?\bgive\s+as|"
    r"\bwhat\s+(?:value\s+or\s+name|answer)\b.{0,100}?\b(?:give|provide)\s+for|"
    r"\bhow\s+did\b.{0,100}?\bdocument)"
    r"\s+(?:who|what|which|when|where|why|how)\b",
    flags=re.IGNORECASE,
)


def looks_like_complete_question(value: str) -> bool:
    return bool(_QUESTION_START.search(value or ""))


def pcq_question_quality_issues(question: str) -> Tuple[str, ...]:
    """Return machine-verifiable question-form defects without semantic guessing."""

    text = " ".join(str(question or "").split())
    issues = []
    imperative = bool(_IMPERATIVE_REQUEST.search(text))
    if not _QUESTION_CLAUSE.search(text) and not imperative:
        issues.append("not_interrogative")
    punctuation_valid = (
        text.endswith("?") and text.count("?") == 1
    ) or (imperative and text.endswith(".") and text.count("?") == 0)
    if not punctuation_valid:
        issues.append("invalid_question_mark_count")
    if _NESTED_TEMPLATE_TARGET.search(text):
        issues.append("nested_interrogative_template")
    return tuple(issues)
