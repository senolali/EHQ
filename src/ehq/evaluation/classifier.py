"""Traceable rule-based response classification."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

from ..types import Classification
from .correctness import matches_gold

ABSTENTION_PATTERNS = (
    r"\bi (?:do not|don't) know\b",
    r"\bi (?:do not|don't) have "
    r"(?:(?:any|enough|reliable|specific|verified)\s+|reliable,\s*verified\s+)?"
    r"(?:information|details|data|knowledge|records|evidence)\b",
    r"\bi (?:cannot|can't) (?:answer|determine|tell|verify|provide)\b",
    r"\bi lack (?:enough )?(?:information|context|evidence)\b",
    r"\bnot enough (?:information|context)\b",
    r"\binsufficient (?:information|context|evidence)\b",
    r"\bthe (?:information|answer) (?:is|was) (?:not provided|redacted|unavailable)\b",
    r"\bno reliable (?:information|basis|way)\b",
    r"\bcannot be (?:determined|inferred|verified)\b",
    r"\bthere is no way to (?:know|determine|infer|verify)\b",
    r"\b(?:the|this) (?:document|text|passage) (?:does not|doesn't) "
    r"(?:explicitly )?"
    r"(?:state|specify|mention|provide|include|contain|indicate)\b",
    r"\b(?:not|isn't|wasn't) (?:stated|specified|mentioned|provided|included|contained|indicated) "
    r"in (?:the|this) (?:document|text|passage)\b",
    r"\b(?:is|was) not disclosed\b",
    r"\b(?:answer|information|value|amount|price|level|limit|percentage|date|name|figure|detail) "
    r"(?:is|was) not available\b",
    r"\b(?:information|value|name|date|figure|detail) (?:has been|is|was) redacted\b",
    r"\[(?:redacted|omitted|withheld)\]",
    r"\b(?:answer|information|value|amount|price|level|limit|percentage|date|name|figure|detail) "
    r"(?:is|was) not (?:mentioned|provided|stated|specified|disclosed)\b",
    r"\bpurchase price (?:is|was) not (?:mentioned|provided|stated|specified|disclosed)\b",
    r"\bunknown\b",
)

HEDGE_PATTERNS = (
    r"\bi(?:'m| am) not sure\b",
    r"\bi am uncertain\b",
    r"\bperhaps\b",
    r"\bpossibly\b",
    r"\bprobably\b",
    r"\blikely\b",
    r"\bpresumably\b",
    r"\bmay(?:be)?\b",
    r"\bmight\b",
    r"\bi (?:think|believe|guess|suspect)\b",
    r"\bit (?:could|may|might) be\b",
    r"\bappears to\b",
    r"\bseems to\b",
    r"\bas far as i know\b",
    r"\bto (?:the best of )?my knowledge\b",
)

_ANSWER_CUE = re.compile(
    r"(?:\bthe answer is\b|\bit is\b|\b\d[\d,.:/%-]*\b)",
    flags=re.IGNORECASE,
)
_PROPER_NAME_CUE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


def _matches(patterns: Sequence[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _substantive_answer_detected(text: str) -> bool:
    """Conservative cue for factual content following an abstention statement."""
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text)]
    non_abstention = [
        sentence
        for sentence in sentences
        if sentence
        and not _matches(ABSTENTION_PATTERNS, sentence)
        and len(sentence.split()) >= 3
    ]
    return any(
        _ANSWER_CUE.search(sentence) or _PROPER_NAME_CUE.search(sentence)
        for sentence in non_abstention
    )


def classify_response(
    response: str,
    item: Mapping[str, Any],
    *,
    adjudicated_label: Optional[str] = None,
    numeric_tolerance: float = 0.0,
) -> Classification:
    if not response or not response.strip():
        raise ValueError("Empty provider responses are technical failures, not abstentions")

    abstention = _matches(ABSTENTION_PATTERNS, response)
    hedge = _matches(HEDGE_PATTERNS, response)
    substantive = _substantive_answer_detected(response)
    correct = matches_gold(response, item, numeric_tolerance=numeric_tolerance)
    reasons = []

    if abstention and not substantive:
        automated = "ABSTAIN"
        reasons.append("explicit_abstention_without_substantive_answer")
    elif hedge or (abstention and substantive):
        automated = "HEDGE"
        reasons.append(
            "uncertainty_marker_with_substantive_answer"
            if substantive
            else "uncertainty_marker"
        )
    elif correct:
        automated = "CONFIDENT_CORRECT"
        reasons.append("unhedged_gold_match")
    else:
        automated = "CONFIDENT_WRONG"
        reasons.append("unhedged_non_gold_answer")

    label = adjudicated_label or automated
    if adjudicated_label:
        reasons.append("human_adjudication_override")
    return Classification(
        label=label,
        is_correct=correct,
        abstention_detected=abstention,
        hedge_detected=hedge,
        substantive_answer_detected=substantive,
        reasons=reasons,
        automated_label=automated,
        adjudicated_label=adjudicated_label,
    )
