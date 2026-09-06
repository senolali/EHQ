"""Versioned prompt construction."""

from __future__ import annotations

from typing import Any, Mapping

ANSWER_PROMPT_VERSION = "answer-v1"
CONFIDENCE_PROMPT_VERSION = "confidence-v2"

CONFIDENCE_SYSTEM_PROMPT = (
    "You are a confidence-scoring endpoint. Your entire response must be "
    "exactly one ASCII integer from 0 to 100. Do not explain or add any "
    "other text."
)

CONFIDENCE_TEMPLATE = """You previously answered a benchmark question.

On a scale from 0 to 100, how confident are you that the substantive factual
content of your answer is correct? If you did not provide a substantive factual
answer, report 0.

QUESTION:
{question}

YOUR PREVIOUS ANSWER:
{answer}

Return exactly one ASCII integer from 0 to 100 and nothing else. Do not explain.

FINAL OUTPUT (ONE INTEGER ONLY):"""


def build_answer_prompt(item: Mapping[str, Any]) -> str:
    category = str(item.get("category", "")).upper()
    question = str(item["question"]).strip()
    if category == "CCQ":
        document = str(item.get("document") or "").strip()
        return (
            "Read the following document and answer the question based only on "
            "the information explicitly present in the document.\n\n"
            f"DOCUMENT:\n{document}\n\nQUESTION:\n{question}"
        )
    return question


def build_confidence_prompt(question_prompt: str, answer: str) -> str:
    return CONFIDENCE_TEMPLATE.format(
        question=question_prompt.strip(),
        answer=answer.strip(),
    )
