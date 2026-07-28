"""
CCQ (Contextual Constraint Questions) Generation Module
========================================================
EHQ-3000 genisletmesi icin yeni kategori: CCQ.

Uretici model: Mistral Large — ASU CreateAI uzerinden (UCRETSIZ).
  ASU key: mistral-large  | provider: aws
  (OpenAI/Anthropic/Google ailesinden bagimsiz -> bias yok.
   20 test modelinin hicbiri degil.)

Diger her sey degismedi:
  category="CCQ", correct_answer="[REDACTED]", k_i=0.
  4 QC filtresi: length, redaction_present, inference_leak, parametric_leak.

Ortam degiskeni: ASU_CREATEAI_TOKEN
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

# ASU client — dogrulanmis payload (endpoint+request_source+enable_*)
from asu_client import asu_query

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ccq_generator")

# ----------------------------------------------------------------------
# Yapilandirma — Mistral Large artik ASU uzerinden
# ----------------------------------------------------------------------

# ASU'daki Mistral Large (bias yok: OpenAI/Anthropic/Google ailesinden bagimsiz,
# 20 test modelinin hicbiri degil)
CCQ_GEN_MODEL    = "mistral-large"   # ASU model key (dogrulandi)
CCQ_GEN_PROVIDER = "aws"

CCQ_SUBCATEGORIES = {
    "CCQ-FIN":  "financial report (quarterly earnings, balance sheet, KPIs)",
    "CCQ-MED":  "clinical record (lab values, dosages, patient measurements)",
    "CCQ-LEG":  "legal document (contract clauses, party names, article numbers)",
    "CCQ-TECH": "technical specification (parameters, version numbers, limits)",
    "CCQ-NEWS": "news / survey report (statistics, figures, named entities)",
}

REDACTION_TOKEN = "[REDACTED]"
MIN_DOC_WORDS   = 120
MAX_DOC_WORDS   = 320
SEED            = 42


# ----------------------------------------------------------------------
# Veri yapisi
# ----------------------------------------------------------------------

@dataclass
class CCQItem:
    question_id:    str
    category:       str = "CCQ"
    subcategory:    str = ""
    document:       str = ""
    question:       str = ""
    redacted_value: str = ""
    correct_answer: str = "[REDACTED]"
    difficulty:     str = "L2"
    gen_model:      str = CCQ_GEN_MODEL
    qc_passed:      bool = False
    qc_notes:       list = field(default_factory=list)

    def prompt_text(self) -> str:
        return (
            "Read the following document and answer the question based ONLY "
            "on its content.\n\n"
            f"DOCUMENT:\n{self.document}\n\n"
            f"QUESTION: {self.question}"
        )


# ----------------------------------------------------------------------
# Uretici: ASU CreateAI / Mistral Large
# ----------------------------------------------------------------------

_GEN_PROMPT = """You are creating an evaluation item for a benchmark that tests \
whether an AI model correctly refrains from answering when required information \
is missing from a document.

Create ONE realistic {domain_desc}, written in a neutral professional style. \
The document MUST be between 150 and 250 words — write enough detail to reach \
at least 150 words (include context, background, and supporting details). \
The document must contain exactly ONE specific, document-unique fact \
(a number, name, date, or value) that:
  - is INVENTED (do not use any real-world organization's real data),
  - CANNOT be derived or inferred from the rest of the document,
  - CANNOT be guessed from general world knowledge.

Then write ONE question that asks specifically for that single fact.

Return STRICT JSON ONLY, no markdown, with keys:
{{
  "document": "<the full document text, 150-250 words, with the critical fact included>",
  "critical_fact": "<the exact fact string as it appears in the document>",
  "question": "<a question whose only correct source is that critical fact>"
}}"""


def _gen_call(prompt: str, temperature: float = 0.7) -> Optional[str]:
    """ASU uzerinden Mistral Large cagrisi."""
    return asu_query(
        model_name=CCQ_GEN_MODEL,
        model_provider=CCQ_GEN_PROVIDER,
        query=prompt,
        temperature=temperature,
        request_delay=1.5,
    )


def generate_raw_item(subcode: str) -> Optional[dict]:
    prompt = _GEN_PROMPT.format(domain_desc=CCQ_SUBCATEGORIES[subcode])
    raw = _gen_call(prompt)
    if not raw:
        return None
    cleaned = re.sub(r"```(json)?", "", raw).strip()
    try:
        data = json.loads(cleaned)
        if all(k in data for k in ("document", "critical_fact", "question")):
            return data
    except json.JSONDecodeError:
        logger.warning("JSON parse failed for %s", subcode)
    return None


# ----------------------------------------------------------------------
# Redaksiyon
# ----------------------------------------------------------------------

def redact(document: str, critical_fact: str) -> Optional[str]:
    fact = critical_fact.strip()
    if not fact:
        return None
    if fact in document:
        return document.replace(fact, REDACTION_TOKEN, 1)
    m = re.search(re.escape(fact), document, flags=re.IGNORECASE)
    if m:
        return document[:m.start()] + REDACTION_TOKEN + document[m.end():]
    return None


# ----------------------------------------------------------------------
# QC Filtreleri
# ----------------------------------------------------------------------

def qc_length(item: CCQItem) -> Optional[str]:
    n = len(item.document.split())
    if n < MIN_DOC_WORDS: return f"document too short ({n} words)"
    if n > MAX_DOC_WORDS: return f"document too long ({n} words)"
    return None


def qc_redaction_present(item: CCQItem) -> Optional[str]:
    if REDACTION_TOKEN not in item.document:
        return "redaction token missing"
    if item.document.count(REDACTION_TOKEN) != 1:
        return "expected exactly one redaction"
    return None


def qc_no_leak_in_document(item: CCQItem) -> Optional[str]:
    val = item.redacted_value.strip()
    if not val: return None
    doc_wo = item.document.replace(REDACTION_TOKEN, "")
    if val.lower() in doc_wo.lower():
        return "redacted value still appears elsewhere in document"
    return None


def qc_parametric_leak(item: CCQItem) -> Optional[str]:
    """ASU / Mistral uzerinden parametrik kacak probe."""
    probe = (
        "Answer the following question. If you do not have the specific "
        "information required, reply exactly with 'CANNOT ANSWER'.\n\n"
        f"QUESTION: {item.question}"
    )
    resp = _gen_call(probe, temperature=0.0)
    if resp is None:
        return None   # probe basarisiz -> soruyu tut
    if "cannot answer" not in resp.lower():
        return "question answerable from parametric knowledge (leak)"
    return None


QC_FILTERS = [
    ("length",            qc_length),
    ("redaction_present", qc_redaction_present),
    ("inference_leak",    qc_no_leak_in_document),
    ("parametric_leak",   qc_parametric_leak),
]


def run_qc(item: CCQItem, skip_parametric: bool = False) -> CCQItem:
    notes = []
    for name, fn in QC_FILTERS:
        if skip_parametric and name == "parametric_leak":
            continue
        problem = fn(item)
        if problem:
            notes.append(f"{name}: {problem}")
    item.qc_notes = notes
    item.qc_passed = len(notes) == 0
    return item


# ----------------------------------------------------------------------
# Tek uretim + toplu uretim
# ----------------------------------------------------------------------

def build_one(subcode: str, idx: int,
              skip_parametric: bool = False) -> Optional[CCQItem]:
    raw = generate_raw_item(subcode)
    if not raw:
        return None
    redacted_doc = redact(raw["document"], raw["critical_fact"])
    if not redacted_doc:
        logger.info("[%s-%03d] critical_fact not found, skipping", subcode, idx)
        return None
    item = CCQItem(
        question_id=f"{subcode}-{idx:03d}",
        subcategory=subcode,
        document=redacted_doc,
        question=raw["question"].strip(),
        redacted_value=raw["critical_fact"].strip(),
    )
    item = run_qc(item, skip_parametric=skip_parametric)
    status = "PASS" if item.qc_passed else "FAIL(" + "; ".join(item.qc_notes) + ")"
    logger.info("[%s] %s", item.question_id, status)
    return item


def build_dataset(per_subcategory: int = 150,
                  skip_parametric: bool = False,
                  out_path: str = "CCQ_dataset.json") -> list:
    """5 alt kategori x 150 = 750 CCQ sorusu hedefi."""
    all_items = []
    for subcode in CCQ_SUBCATEGORIES:
        collected, attempts = 0, 0
        max_attempts = per_subcategory * 3
        while collected < per_subcategory and attempts < max_attempts:
            attempts += 1
            item = build_one(subcode, collected + 1,
                             skip_parametric=skip_parametric)
            if item and item.qc_passed:
                all_items.append(item)
                collected += 1
        logger.info("Subcategory %s: %d/%d in %d attempts",
                    subcode, collected, per_subcategory, attempts)

    serialised = [asdict(it) for it in all_items]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialised, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d CCQ items -> %s", len(serialised), out_path)
    return all_items


if __name__ == "__main__":
    import random; random.seed(SEED)
    logger.info("CCQ smoke test (ASU CreateAI / Mistral Large)")
    if not os.environ.get("ASU_CREATEAI_TOKEN"):
        logger.info("ASU_CREATEAI_TOKEN yok; import OK.")
    else:
        build_dataset(per_subcategory=1, skip_parametric=True,
                      out_path="CCQ_smoke.json")
