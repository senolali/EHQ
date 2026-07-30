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
import random
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
# Cesitlilik ekseni (pcq_generator.py / feq_generator.py'de ogrenilen ders):
# _GEN_PROMPT her cagrida SUBCODE basina TAMAMEN AYNI (domain_desc sabit)
# gonderiliyordu -- bu hem PCQ'nun "whole-pool" hatasinin hem de FEQ'nun
# "Dr. Elara Vesper" tekrar hatasinin kok nedeniyle ayni: modele ayni
# statik prompt'u tekrar tekrar gonderip "farkli bir sey uret" demek
# guvenilir calismiyor. Cesitliligi modelin taktirine birakmak yerine
# YAPISAL olarak zorluyoruz: her cagriya somut bir alt-baglam (sektor/
# klinik birim/sozlesme turu/standart turu/anket konusu) enjekte ediyoruz.
_FLAVORS = {
    "CCQ-FIN":  ["a mid-cap semiconductor manufacturer", "a regional airline",
                "an agricultural cooperative", "a boutique investment bank",
                "a renewable-energy utility", "a consumer electronics retailer",
                "a logistics and freight company", "a biotechnology startup",
                "a commercial real estate firm", "a telecommunications carrier"],
    "CCQ-MED":  ["a cardiology outpatient clinic", "a pediatric oncology ward",
                "an orthopedic surgery unit", "a diabetes management program",
                "an emergency department triage log", "a prenatal care clinic",
                "a psychiatric evaluation record",
                "a physical therapy rehabilitation log",
                "an infectious disease case report",
                "a geriatric care assessment"],
    "CCQ-LEG":  ["a commercial lease agreement", "a software licensing contract",
                "a merger and acquisition term sheet",
                "an employment severance agreement",
                "a construction subcontractor agreement",
                "a non-disclosure agreement",
                "an intellectual property licensing deal",
                "a supply chain vendor contract", "a shareholder agreement",
                "a franchise agreement"],
    "CCQ-TECH": ["a wireless communication protocol specification",
                "a structural engineering load standard",
                "a software API specification",
                "an automotive safety standard",
                "a data encryption protocol",
                "a manufacturing quality control standard",
                "a renewable energy grid interconnection standard",
                "a medical device technical specification",
                "an aviation maintenance standard",
                "a materials science testing standard"],
    "CCQ-NEWS": ["a regional employment survey", "a consumer spending report",
                "a public health survey", "an education outcomes study",
                "a housing market report", "a transportation usage survey",
                "an environmental impact assessment",
                "a crime statistics report", "a workplace satisfaction survey",
                "an agricultural yield report"],
}


def random_flavor(subcode: str) -> str:
    """Sabit domain_desc yerine, her cagrida somut bir alt-baglam secerek
    modelin ayni belgeyi/faktu tekrar tekrar uretmesini engeller."""
    base = CCQ_SUBCATEGORIES[subcode]
    flavor = random.choice(_FLAVORS[subcode])
    return f"{base}, specifically for {flavor}"


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
    """ASU uzerinden Mistral Large cagrisi.
    use_cache=False: PCQ'da tespit edilen hata -- asu_client'in disk
    cache'i temperature'i anahtara dahil etmiyor; domain_desc sabit
    kaldigi surece (asagidaki random_flavor duzeltmesinden ONCEKI halde
    oldugu gibi) ayni prompt+model cache'den ayni yaniti donduruyor,
    yani GERCEK API cagrisi olmadan sonsuz ayni belge/soru donguye
    giriyordu. Cache'i kapatmak + prompt'u cesitlendirmek birlikte
    calisiyor (bkz. random_flavor)."""
    return asu_query(
        model_name=CCQ_GEN_MODEL,
        model_provider=CCQ_GEN_PROVIDER,
        query=prompt,
        temperature=temperature,
        request_delay=1.5,
        use_cache=False,
    )


def generate_raw_item(subcode: str) -> Optional[dict]:
    prompt = _GEN_PROMPT.format(domain_desc=random_flavor(subcode))
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


def qc_duplicate(item: CCQItem, seen_facts: set, seen_questions: set) -> Optional[str]:
    # feq_generator.py'de tespit edilen ayni sinif hata: cesitlendirme
    # (random_flavor) tek basina cakismayi sifira indirmez, ek bir
    # guvenlik agi olarak duplicate kontrolu gerekiyor.
    if item.redacted_value.strip().lower() in seen_facts:
        return "duplicate critical fact (already generated in this run)"
    if item.question.strip().lower() in seen_questions:
        return "duplicate question"
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
    ("length",            lambda item, seen_facts, seen_questions: qc_length(item)),
    ("redaction_present", lambda item, seen_facts, seen_questions: qc_redaction_present(item)),
    ("inference_leak",    lambda item, seen_facts, seen_questions: qc_no_leak_in_document(item)),
    ("duplicate",         qc_duplicate),
    ("parametric_leak",   lambda item, seen_facts, seen_questions: qc_parametric_leak(item)),
]


def run_qc(item: CCQItem, seen_facts: set, seen_questions: set,
          skip_parametric: bool = False) -> CCQItem:
    notes = []
    for name, fn in QC_FILTERS:
        if skip_parametric and name == "parametric_leak":
            continue
        problem = fn(item, seen_facts, seen_questions)
        if problem:
            notes.append(f"{name}: {problem}")
            break   # ilk basarisizlikta dur -- pahali parametric_leak
                    # cagrisini gereksiz yere tetikleme
    item.qc_notes = notes
    item.qc_passed = len(notes) == 0
    return item


# ----------------------------------------------------------------------
# Tek uretim + toplu uretim
# ----------------------------------------------------------------------

def build_one(subcode: str, idx: int, seen_facts: set, seen_questions: set,
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
    item = run_qc(item, seen_facts, seen_questions, skip_parametric=skip_parametric)
    status = "PASS" if item.qc_passed else "FAIL(" + "; ".join(item.qc_notes) + ")"
    logger.info("[%s] %s", item.question_id, status)
    return item


def build_dataset(per_subcategory: int = 150,
                  skip_parametric: bool = False,
                  out_path: str = "CCQ_dataset.json") -> list:
    """5 alt kategori x 150 = 750 CCQ sorusu hedefi."""
    all_items = []
    # seen_facts/seen_questions TUM kategoriler arasinda paylasilir
    # (bkz. feq_generator.py'deki ayni tasarim gerekcesi).
    seen_facts: set = set()
    seen_questions: set = set()
    for subcode in CCQ_SUBCATEGORIES:
        collected, attempts = 0, 0
        max_attempts = per_subcategory * 4
        while collected < per_subcategory and attempts < max_attempts:
            attempts += 1
            item = build_one(subcode, collected + 1, seen_facts, seen_questions,
                             skip_parametric=skip_parametric)
            if item and item.qc_passed:
                seen_facts.add(item.redacted_value.strip().lower())
                seen_questions.add(item.question.strip().lower())
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
    import argparse
    random.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Tam uretim: 5 alt kategori x 150 = 750 hedef "
                             "(CCQ_dataset.json). Verilmezse smoke test calisir.")
    args = parser.parse_args()

    logger.info("CCQ | Uretici: Mistral-Large(ASU) | correct_answer=%s", REDACTION_TOKEN)
    if not os.environ.get("ASU_CREATEAI_TOKEN"):
        logger.info("ASU_CREATEAI_TOKEN yok; import OK.")
    elif args.full:
        logger.info("CCQ TAM URETIM | 5 alt kategori x 150 = 750 hedef")
        build_dataset(per_subcategory=150, out_path="CCQ_dataset.json")
    else:
        build_dataset(per_subcategory=1, skip_parametric=True,
                      out_path="CCQ_smoke.json")
