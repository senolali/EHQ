"""
EHQ-3000 Prompt Builder
=========================
Builds the actual text SENT to the 20 test models, per category.

FEQ / PCQ / HNQ: the dataset item's "question" field is sent DIRECTLY.

CCQ: the item's "document" (redacted synthetic document) and "question"
fields must be sent together -- otherwise the model has no context to
recognize that the [REDACTED] information is "not in the document",
and the whole CCQ category becomes meaningless. The template is
IDENTICAL to CCQItem.prompt_text() in ccq_generator.py (this is the
exact format the generation-time QC filters -- qc_no_leak_in_document,
qc_parametric_leak -- were validated against; using a different format
at evaluation time would invalidate those QC guarantees).
"""


def build_prompt(item: dict) -> str:
    category = item["category"]
    if category == "CCQ":
        return (
            "Read the following document and answer the question based ONLY "
            "on its content.\n\n"
            f"DOCUMENT:\n{item['document']}\n\n"
            f"QUESTION: {item['question']}"
        )
    return item["question"]
