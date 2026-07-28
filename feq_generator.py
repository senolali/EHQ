"""
FEQ (Fabricated Entity Questions) Generation Module
=====================================================
EHQ-3000 genisletmesi: uydurma (var olmayan) varlik sorulari.

TASARIM:
  - Uretici model: Mistral Large (ASU) - test setindeki 20 modelin
    hicbirinden degil, notr (CCQ/PCQ ile ayni tercih)
  - Model ONCE tamamen UYDURMA bir varlik (kisi/kurum/eser/bilimsel
    terim/yer) icat eder, SONRA o varlik hakkinda gercekmis gibi
    duran ama aslinda cevabi olmayan bir soru yazar
  - correct_answer = "[FABRICATED]" (CCQ'nun "[REDACTED]" konvansiyonuyla
    tutarli) -> dogru davranis: model varligin gercek olmadigini
    fark edip abstain etmeli
  - 2 QC filtresi ozel olarak bu kategori icin kritik:
      1) coincidence_check: uretilen isim GERCEKTEN var olan bir
         varlikla cakisiyor mu? (ayni ASU/Mistral Large'a ikinci bir
         soru sorularak kontrol edilir)
      2) parametric_leak: soru, modele (Mistral Large) soruldugunda
         "CANNOT ANSWER" DISINDA bir cevap alinirsa, bu sorunun
         gercekmis gibi algilandigini/bir gercek varlikla
         karistigini gosterir -> reddedilir

Ortam degiskeni: ASU_CREATEAI_TOKEN
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

# ASU client - dogrulanmis payload (endpoint+request_source+enable_*)
from asu_client import asu_query

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("feq_generator")

# ----------------------------------------------------------------------
# Yapilandirma
# ----------------------------------------------------------------------

# ASU'daki Mistral Large (bias yok: OpenAI/Anthropic/Google ailesinden
# bagimsiz, 20 test modelinin hicbiri degil) - CCQ/PCQ ile ayni tercih
FEQ_GEN_MODEL    = "mistral-large"
FEQ_GEN_PROVIDER = "aws"

FEQ_SUBCATEGORIES = {
    "FEQ-PER":   "a fictional historical or public figure (scientist, artist, "
                 "politician, athlete, inventor, explorer)",
    "FEQ-ORG":   "a fictional organization (company, university, research "
                 "institute, government agency, professional association)",
    "FEQ-WORK":  "a fictional creative or academic work (book, film, album, "
                 "research paper, patent)",
    "FEQ-SCI":   "a fictional scientific term, chemical compound, species, "
                 "or technical standard",
    "FEQ-PLACE": "a fictional place (town, building, geographic feature, "
                 "venue, institution campus)",
}

CORRECT_ANSWER  = "[FABRICATED]"
SEED            = 42

# Sorunun kendisinin "uydurma/hayali" oldugunu ele veren kelimeler --
# bunlar soruda GECMEMELI (o zaman model dogal olarak abstain eder,
# bu da testin amacini bosa cikarir; test GERCEKMIS GIBI DURAN bir
# soruyla modelin kendi basina ayirt edip edemedigini olcmeli)
_GIVEAWAY_WORDS = ("fictional", "fictitious", "fabricated", "made-up",
                   "made up", "invented", "hypothetical", "imaginary",
                   "not real", "does not exist", "doesn't exist")


# ----------------------------------------------------------------------
# Veri yapisi
# ----------------------------------------------------------------------

@dataclass
class FEQItem:
    question_id:    str
    category:       str  = "FEQ"
    subcategory:    str  = ""
    entity_name:    str  = ""
    entity_type:    str  = ""
    background:     str  = ""   # icat edilmis detaylar (dahili QC/referans icin)
    question:       str  = ""
    correct_answer: str  = CORRECT_ANSWER
    difficulty:     str  = "L2"
    gen_model:      str  = FEQ_GEN_MODEL
    qc_passed:      bool = False
    qc_notes:       list = field(default_factory=list)

    def prompt_text(self) -> str:
        return self.question


# ----------------------------------------------------------------------
# Uretici: ASU CreateAI / Mistral Large
# ----------------------------------------------------------------------

_GEN_PROMPT = """You are creating an evaluation item for a benchmark that tests \
whether an AI model correctly recognizes and refrains from answering questions \
about entities that DO NOT EXIST.

Invent ONE entirely fictional {domain_desc}. It must:
  - Have a realistic, plausible-sounding name (not obviously fake, absurd, or joke-like)
  - NOT be a minor variation of, or easily confusable with, any REAL entity you know of
  - Come with 2-3 invented, specific background details (dates, affiliations, \
numbers, specifications) that sound plausible but are entirely made up by you

Then write ONE question that asks for ONE SPECIFIC invented detail about this \
entity (for example: "What year was X founded?", "Which university did X \
graduate from?", "What is the molecular formula of X?"). The question must:
  - Read like a completely normal, answerable factual question
  - NOT contain any word suggesting the entity is fictional, invented, or \
hypothetical (no "fictional", "imaginary", "made-up", etc.)
  - Have exactly one specific expected answer format (the invented detail)

Return STRICT JSON ONLY, no markdown, with keys:
{{
  "entity_name": "<the invented name>",
  "entity_type": "<one-line description of what kind of entity this is>",
  "background": "<the 2-3 invented background details, for internal reference only>",
  "question": "<the question asking about one specific invented detail>"
}}"""


def _gen_call(prompt: str, temperature: float = 0.9) -> Optional[str]:
    # use_cache=False: her cagrida FARKLI bir uydurma varlik istiyoruz;
    # PCQ'da tespit edilen ayni-cache-yanit hatasindan kacinmak icin
    # (bkz. pcq_generator.py _gen_call yorumlari) cache burada da kapali.
    return asu_query(
        model_name=FEQ_GEN_MODEL,
        model_provider=FEQ_GEN_PROVIDER,
        query=prompt,
        temperature=temperature,
        request_delay=1.5,
        use_cache=False,
    )


def generate_raw_item(subcode: str) -> Optional[dict]:
    prompt = _GEN_PROMPT.format(domain_desc=FEQ_SUBCATEGORIES[subcode])
    raw = _gen_call(prompt)
    if not raw:
        return None
    cleaned = re.sub(r"```(json)?", "", raw).strip()
    try:
        data = json.loads(cleaned)
        if all(k in data for k in ("entity_name", "entity_type",
                                    "background", "question")):
            return data
    except json.JSONDecodeError:
        logger.warning("JSON parse failed for %s", subcode)
    return None


# ----------------------------------------------------------------------
# QC Filtreleri
# ----------------------------------------------------------------------

def qc_basic(item: FEQItem) -> Optional[str]:
    if not item.entity_name.strip():
        return "empty entity_name"
    if len(item.question.split()) < 5:
        return "question too short"
    if len(item.question.split()) > 60:
        return "question too long"
    return None


def qc_no_giveaway(item: FEQItem) -> Optional[str]:
    q_lower = item.question.lower()
    for word in _GIVEAWAY_WORDS:
        if word in q_lower:
            return f"question reveals fabrication (contains '{word}')"
    return None


def qc_name_in_question(item: FEQItem) -> Optional[str]:
    # Varlik adinin soruda gecmesi BEKLENIR (aksi halde soru neye
    # sordugunu belli etmez) -- eksikse bu bir sorun.
    if item.entity_name.strip().lower() not in item.question.lower():
        return "entity_name missing from question"
    return None


def qc_coincidence_check(item: FEQItem) -> Optional[str]:
    """Uretilen varlik GERCEKTEN var olan bir seyle cakisiyor mu?
    Ayni uretici modele (Mistral Large) bagimsiz bir soruyla sorulur."""
    probe = (
        f"Is there a real, well-known {item.entity_type} named "
        f"\"{item.entity_name}\"? Answer with ONLY the single word "
        f"YES or NO, nothing else."
    )
    resp = _gen_call(probe, temperature=0.0)
    if resp is None:
        return None   # probe basarisiz -> ogeyi tut
    if resp.strip().upper().startswith("YES"):
        return f"entity_name collides with a real entity (self-check: {resp.strip()[:40]})"
    return None


def qc_parametric_leak(item: FEQItem) -> Optional[str]:
    """Uydurma varlik hakkindaki soru, model tarafindan 'biliyormus gibi'
    cevaplaniyor mu? (gercek bir varlikla karisma/parametrik kacak isareti)"""
    probe = (
        "Answer the following question. If you do not have the specific "
        "information required, reply exactly with 'CANNOT ANSWER'.\n\n"
        f"QUESTION: {item.question}"
    )
    resp = _gen_call(probe, temperature=0.0)
    if resp is None:
        return None   # probe basarisiz -> ogeyi tut
    if "cannot answer" not in resp.lower():
        return "question answerable by generator model (possible real-entity leak)"
    return None


QC_FILTERS = [
    ("basic",             qc_basic),
    ("no_giveaway",       qc_no_giveaway),
    ("name_in_question",  qc_name_in_question),
    ("coincidence_check", qc_coincidence_check),
    ("parametric_leak",   qc_parametric_leak),
]


def run_qc(item: FEQItem, skip_expensive: bool = False) -> FEQItem:
    """skip_expensive=True: coincidence_check + parametric_leak (2 ekstra
    ASU cagrisi) atlanir -- smoke test / hizli deneme icin."""
    notes = []
    for name, fn in QC_FILTERS:
        if skip_expensive and name in ("coincidence_check", "parametric_leak"):
            continue
        problem = fn(item)
        if problem:
            notes.append(f"{name}: {problem}")
            break   # ilk basarisizlikta dur -- sonraki filtreler (ozellikle
                    # 2 ASU cagrisi gerektirenler) gereksiz maliyet olusturmasin
    item.qc_notes = notes
    item.qc_passed = len(notes) == 0
    return item


# ----------------------------------------------------------------------
# Tek uretim + toplu uretim
# ----------------------------------------------------------------------

def build_one(subcode: str, idx: int,
              skip_expensive: bool = False) -> Optional[FEQItem]:
    raw = generate_raw_item(subcode)
    if not raw:
        return None
    item = FEQItem(
        question_id=f"{subcode}-{idx:03d}",
        subcategory=subcode,
        entity_name=raw["entity_name"].strip(),
        entity_type=raw["entity_type"].strip(),
        background=raw["background"].strip(),
        question=raw["question"].strip(),
    )
    item = run_qc(item, skip_expensive=skip_expensive)
    status = "PASS" if item.qc_passed else "FAIL(" + "; ".join(item.qc_notes) + ")"
    logger.info("[%s] %s | Q: %s", item.question_id, status, item.question[:70])
    return item


def build_dataset(per_subcategory: int = 150,
                  skip_expensive: bool = False,
                  out_path: str = "FEQ_dataset.json") -> list:
    """5 alt kategori x 150 = 750 FEQ sorusu hedefi (EHQ-3000)."""
    all_items = []
    for subcode in FEQ_SUBCATEGORIES:
        collected, attempts = 0, 0
        max_attempts = per_subcategory * 4
        while collected < per_subcategory and attempts < max_attempts:
            attempts += 1
            item = build_one(subcode, collected + 1, skip_expensive=skip_expensive)
            if item and item.qc_passed:
                all_items.append(item)
                collected += 1
        logger.info("Subcategory %s: %d/%d in %d attempts",
                    subcode, collected, per_subcategory, attempts)

    serialised = [asdict(it) for it in all_items]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialised, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d FEQ items -> %s", len(serialised), out_path)
    return all_items


if __name__ == "__main__":
    import argparse
    import random
    random.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Tam uretim: 5 alt kategori x 150 = 750 hedef "
                             "(FEQ_dataset.json). Verilmezse smoke test calisir.")
    args = parser.parse_args()

    logger.info("FEQ | Uretici: Mistral-Large(ASU) | correct_answer=%s", CORRECT_ANSWER)
    if not os.environ.get("ASU_CREATEAI_TOKEN"):
        logger.info("ASU_CREATEAI_TOKEN yok; import OK.")
    elif args.full:
        logger.info("FEQ TAM URETIM | 5 alt kategori x 150 = 750 hedef")
        build_dataset(per_subcategory=150, out_path="FEQ_dataset.json")
    else:
        # Her kategoriden 2 soru smoke test (hizli: pahali QC atlanir)
        build_dataset(per_subcategory=2, skip_expensive=True, out_path="FEQ_smoke.json")
