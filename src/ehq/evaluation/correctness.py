"""Deterministic gold-answer matching with explicit aliases and numeric support."""

from __future__ import annotations

import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from ..constants import REDACTION_TOKEN

_TOKEN = re.compile(r"\w+", re.UNICODE)
_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    value = "".join(
        character
        for character in value
        if unicodedata.category(character) != "Mn"
    )
    value = " ".join(_TOKEN.findall(value))
    return value.strip()


def contains_normalized_phrase(text: str, phrase: str) -> bool:
    """Return whether a normalized phrase occurs on complete token boundaries."""
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    return bool(
        re.search(
            rf"(?:^|\s){re.escape(normalized_phrase)}(?:$|\s)",
            normalized_text,
        )
    )


def gold_answers(item: Mapping[str, Any]) -> List[str]:
    answers: List[str] = []
    value = item.get("correct_answer")
    if isinstance(value, str) and value.strip():
        answers.append(value.strip())
    aliases = item.get("acceptable_answers") or item.get("answer_aliases") or []
    if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)):
        answers.extend(str(alias).strip() for alias in aliases if str(alias).strip())
    return list(dict.fromkeys(answers))


def _numbers(value: str) -> List[Decimal]:
    result: List[Decimal] = []
    for match in _NUMBER.findall(value):
        try:
            result.append(Decimal(match.replace(",", ".")))
        except InvalidOperation:
            continue
    return result


def _numeric_match(response: str, answer: str, tolerance: float) -> bool:
    response_numbers = _numbers(response)
    answer_numbers = _numbers(answer)
    if len(answer_numbers) != 1 or not response_numbers:
        return False
    expected = answer_numbers[0]
    for observed in response_numbers:
        denominator = max(abs(float(expected)), 1.0)
        if abs(float(observed - expected)) / denominator <= tolerance:
            return True
    return False


def matches_gold(
    response: str,
    item: Mapping[str, Any],
    *,
    numeric_tolerance: float = 0.0,
) -> bool:
    """Match response against explicit gold evidence without semantic guessing."""
    category = str(item.get("category", "")).upper()
    answers = gold_answers(item)
    if category in {"FEQ", "CCQ"}:
        # These categories intentionally do not expose a factual answer.
        return False
    answers = [answer for answer in answers if answer != REDACTION_TOKEN]
    if not response or not answers:
        return False

    normalized_response = normalize_text(response)
    for answer in answers:
        normalized_answer = normalize_text(answer)
        if not normalized_answer:
            continue
        if normalized_response == normalized_answer:
            return True
        # A complete gold phrase may appear inside a longer explanatory answer.
        if re.search(
            rf"(?:^|\s){re.escape(normalized_answer)}(?:$|\s)",
            normalized_response,
        ):
            return True
        if numeric_tolerance >= 0 and _numeric_match(
            response, answer, numeric_tolerance
        ):
            return True
    return False
