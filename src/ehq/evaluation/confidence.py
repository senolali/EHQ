"""Auditable confidence parsing for the protocol's 0-100 integer response."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_INTEGER_ONLY = re.compile(r"^\s*(0|[1-9]\d?|100)\s*[%]?\s*$")
_ANY_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_NUMBER_TOKEN = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?(?![\w.])")


@dataclass(frozen=True)
class ConfidenceParseResult:
    value: Optional[float]
    strategy: Optional[str]
    reason: Optional[str]


def parse_confidence_detailed(text: Optional[str]) -> ConfidenceParseResult:
    """Parse one unambiguous integer and retain the parsing decision.

    Exact integer-only output is preferred. A prose wrapper is accepted only
    when the entire response contains exactly one numeric token, that token is
    an integer in [0, 100], and no sign/decimal interpretation is required.
    Multiple values remain invalid rather than being guessed.
    """

    if not text or not text.strip():
        return ConfidenceParseResult(None, None, "empty")
    exact = _INTEGER_ONLY.fullmatch(text)
    if exact:
        strategy = "exact_percent" if "%" in text else "exact_integer"
        return ConfidenceParseResult(int(exact.group(1)) / 100.0, strategy, None)

    all_tokens = _ANY_NUMBER.findall(text)
    if len(all_tokens) != 1:
        return ConfidenceParseResult(
            None,
            None,
            "multiple_numeric_tokens" if all_tokens else "no_numeric_token",
        )
    tokens = _NUMBER_TOKEN.findall(text)
    if len(tokens) != 1:
        return ConfidenceParseResult(None, None, "non_standalone_numeric_token")
    token = tokens[0]
    if token.startswith(("+", "-")) or "." in token or "," in token:
        return ConfidenceParseResult(None, None, "non_integer_numeric_token")
    value = int(token)
    if not 0 <= value <= 100:
        return ConfidenceParseResult(None, None, "out_of_range")
    return ConfidenceParseResult(value / 100.0, "single_integer_in_text", None)


def parse_confidence(text: Optional[str]) -> Optional[float]:
    """Return normalized confidence or None; never guess or impute a value."""

    return parse_confidence_detailed(text).value
