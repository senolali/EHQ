"""
PCQ (Post-Cutoff Event Questions) Generation Module
====================================================
EHQ-3000 genisletmesi: Haziran-Temmuz 2026 gercek olay penceresi.

TASARIM:
  - Olay penceresi: Haziran-Temmuz 2026
  - Tum 20 test modelinin cutoff'u en gec Ocak 2026 -> k_i=0 %100 garantili
  - Gold answer: gercek kaynaktan (hakem paneline GEREK YOK)
  - Uretici: Mistral Large (ASU) - kaynaktan soru + gold answer uretir
  - Dokumansiz soru: model kendi egitim bilgisinden cevaplamayi deniyor
    ama Haz-Tem 2026'yi bilmiyor -> abstain etmeli (EHQ ozu)

KAYNAK HAVUZU (web'den dogrulanmis):
  PCQ-SPO: 2026 FIFA Dunya Kupasi (Haz 11 - Tem 19, 2026)
  PCQ-POL: 2026 NATO Zirvesi (Tem 7-8, Ankara)
  PCQ-SCI: Haziran-Temmuz 2026 AI/teknoloji olaylari
  PCQ-ECO: IMF WEO Guncelleme + ekonomi olaylari (Temmuz 2026)
  PCQ-WOR: Dunya olaylari (Venezuela depremi, UK PM degisimi vb.)

Ortam degiskeni: ASU_CREATEAI_TOKEN
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from asu_client import asu_query

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pcq_generator")

# ----------------------------------------------------------------------
# Yapilandirma
# ----------------------------------------------------------------------

EVENT_WINDOW = "June-July 2026"

GENERATOR_MODEL    = "mistral-large"
GENERATOR_PROVIDER = "aws"

# Gercek olay kaynak havuzu - web'den dogrulanmis
SOURCE_FACTS = {
"PCQ-SPO": """
2026 FIFA World Cup (June 11 - July 19, 2026, hosted by USA/Canada/Mexico):
- CHAMPION: Spain defeated Argentina 1-0 after extra time in the final on July 19, 2026
- Winning goal: Ferran Torres (substitute), 106th minute
- Final venue: New York New Jersey Stadium (MetLife Stadium), East Rutherford, New Jersey
- Argentina's Enzo Fernandez was sent off (red card, two yellows) in second half
- Third place: England defeated France 6-4 (Bukayo Saka hat-trick), July 18, Hard Rock Stadium, Miami
- Golden Boot: Kylian Mbappe (France), 8 goals in tournament, 20 career World Cup goals
- Golden Glove: Emiliano Martinez (Argentina)
- Host nations eliminated in Round of 16: Canada (lost to Morocco), Mexico (lost to England), USA (lost to Belgium)
- Tournament format: 48 teams, 12 groups of 4
- Group A: Mexico 1st (beat South Africa 2-0, South Korea 1-0, Czechia 3-0)
- Group B: Canada 1st (beat Qatar 6-0 in match 2)
- Group D: USA 1st (beat Paraguay 4-1 in opener, June 12)
- Germany beat Curacao 7-1 (Group E, June 14)
- Lionel Messi became first player to make 30 World Cup appearances
- Semifinalists: Spain, Argentina, England, France
- Spain's second World Cup title (first was 2010)
""",

"PCQ-POL": """
2026 NATO Summit - Ankara, Turkey (July 7-8, 2026):
- Host: Turkey (Turkiye) - second time hosting, first was Istanbul 2004
- Venue: Besiktas Presidential Complex (Kulliye), Ankara
- This was the 36th NATO Summit
- Secretary General: Mark Rutte
- NATO has 32 member states
- Key priorities: defense investment (5% GDP target), Ukraine support, transatlantic industrial cooperation
- NATO Defence Industry Forum (NSDIF26) held July 7, 2026, at ATO Congresium, Ankara
- Allied leaders reaffirmed Article 5 commitment
- European allies and Canada collectively spent additional $1.2 trillion on defense over past decade
- Summit Declaration published: "Ankara Summit Declaration"
- Follows 2025 The Hague Summit; precedes 2027 Albania Summit
- US President Trump's NATO spending demands addressed at summit
- Rutte's goal: "NATO 3.0 - stronger Europe in stronger NATO"
- UK: John Healey resigned as Defence Secretary before summit over funding disputes
""",

"PCQ-SCI": """
AI and Technology Events, June-July 2026:
- GPT-5.5 Instant released by OpenAI (June 2026)
- Google's Gemini 3.5 Flash released (June 2026)
- Anthropic's Claude Opus 4.8 released (June 2026), new performance benchmarks
- GPT-5.6 (Luna, Terra, Sol) released by OpenAI (July 9, 2026) with February 16, 2026 cutoff
- Claude Sonnet 5 released by Anthropic (July 2026), $2/million input tokens, $10/million output
- Kimi K2.7 Code: first open-weight model inside GitHub Copilot
- Reflection AI raised $6.3 billion in compute commitments through 2029
- NVIDIA Cosmos 3 and Intel Xeon 6+ hardware upgrades announced
- DeepSeek announced custom inference silicon design to reduce Nvidia dependence
- Microsoft cut 4,800 jobs (Xbox margin crisis, July 2026)
- Orion-100B trained 100B parameter model for $1.25/hour
- ZoomMate AI assistant launched at $20/user/month
- US Federal agencies hit July 2 deadline from June 2 AI executive order
- Google Imagen 3 Nano and Pro became widely available (June 2026)
- AMD Advancing AI conference scheduled July 22-23, 2026
- Google I/O 2026 announced AI search enhancements
- SpaceX AI, OpenAI, Meta shipped flagship models within 24 hours (July 2026)
""",

"PCQ-ECO": """
Economic Events, June-July 2026:
- IMF World Economic Outlook Update published July 8, 2026
- IMF assumed Strait of Hormuz begins reopening mid-July, normalizing by March 2027
- Average oil price assumption: $89/barrel for 2026 (based on June 10 market pricing)
- US-Iran conflict affecting Strait of Hormuz (approximately 20% of world oil passes through)
- IMF projected global growth with downside risks due to geopolitical uncertainty
- Iran announced plans to "completely block" Strait of Hormuz (June 2026)
- US and Iran exchanged strikes targeting infrastructure and military targets (July 18, 2026)
- Venezuela earthquakes (magnitude 7.2 and 7.5, June 24, 2026) caused economic disruption
- 1,430 confirmed dead in Venezuela earthquake, thousands missing
- UK Prime Minister change: Keir Starmer ousted, Andy Burnham became new PM (July 20, 2026)
- Reflection AI secured $6.3 billion compute deal through 2029
- AI hiring: IBM reported 8% of IT roles now "AI-focused" as of June 2026
- Taylor Swift and Travis Kelce married July 3, 2026 at Madison Square Garden (1,000 guests)
- US celebrated 250th anniversary (semiquincentennial) on July 4, 2026
""",

"PCQ-WOR": """
World Events, June-July 2026:
- Venezuela earthquakes: magnitude 7.2 and 7.5 struck northwestern Venezuela on June 24, 2026
  - 1,430 confirmed dead, thousands more missing, widespread damage
- UK leadership change: Keir Starmer ousted by Labour Party, Andy Burnham became 7th UK PM
  in a decade (July 20, 2026); Burnham's career shaped by England's North-South divide
- US-Iran War: conflict over Strait of Hormuz intensified (July 2026)
  - US and Iran exchanged strikes on infrastructure/military targets July 18, 2026
  - Iran controlled Strait of Hormuz; ~20% of world oil affected
- Taylor Swift married Travis Kelce at Madison Square Garden, July 3, 2026
  - ~1,000 guests attended; thousands of fans gathered outside
- USA celebrated 250th anniversary (semiquincentennial) on July 4, 2026
  - Time capsule buried at Philadelphia's Independence National Historic Park
- Armenia: parliament passed law requiring stricter residency rules for overseas voters
- Egypt advanced to FIFA World Cup Round of 16 for first time (beat Australia on penalties)
- Lionel Messi: first player to make 30 World Cup appearances, scored in 8 consecutive World Cups
- Moldovan PM Alexandru Munteanu resigned
- Sri Lanka prison riots: 27 dead including 4 officers (Negombo prison, rival drug gangs)
- 2026 NATO Summit held Ankara, Turkey July 7-8
"""
}

PCQ_DOMAINS = {
    "PCQ-SPO": "sports",
    "PCQ-POL": "politics and government",
    "PCQ-SCI": "science and technology",
    "PCQ-ECO": "economy and business",
    "PCQ-WOR": "world events",
}

SEED = 42


# ----------------------------------------------------------------------
# Veri yapisi (mevcut EHQ semasiyla uyumlu)
# ----------------------------------------------------------------------

@dataclass
class PCQItem:
    question_id:    str
    category:       str  = "PCQ"
    subcategory:    str  = ""
    question:       str  = ""
    correct_answer: str  = ""
    event_date:     str  = ""
    source_fact:    str  = ""   # gold answer'in dayandigi kaynak bilgisi
    qc_passed:      bool = False
    qc_notes:       list = field(default_factory=list)


# ----------------------------------------------------------------------
# Uretici: Mistral Large (ASU)
# ----------------------------------------------------------------------

_GEN_PROMPT = """You are generating factual quiz questions for an AI benchmark.

Based on the following verified facts about events in {window}, generate ONE question that:
1. Has a single, specific, unambiguous factual answer
2. Can be answered from the facts below WITHOUT needing external information
3. Is NOT answerable from general knowledge about events BEFORE {window}
   (it must be about a specific result, name, number, date, or decision from this period)
4. Has a concise gold answer (1-10 words maximum)

VERIFIED FACTS:
{facts}

Generate a different question each time - vary the topic, entity, and type of fact asked.

Return STRICT JSON ONLY (no markdown):
{{
  "question": "<specific factual question about {window} events>",
  "correct_answer": "<concise factual answer, 1-10 words>",
  "event_date": "<approximate date within {window}>",
  "source_fact": "<the specific fact from above that supports this answer>"
}}"""


def _gen_call(prompt: str, temperature: float = 0.8) -> Optional[str]:
    return asu_query(
        model_name=GENERATOR_MODEL,
        model_provider=GENERATOR_PROVIDER,
        query=prompt,
        temperature=temperature,
        request_delay=1.5,
    )


def generate_raw_item(subcode: str, existing_questions: set) -> Optional[dict]:
    """Kaynak havuzundan tek PCQ uret."""
    facts = SOURCE_FACTS[subcode]
    # Tekrar onlemek icin mevcut sorulari prompt'a ekle
    avoid = ""
    if existing_questions:
        sample = list(existing_questions)[-10:]  # (seen_questions kategoriler arasi paylasilir)
        avoid = f"\n\nAVOID generating questions similar to these already generated:\n" + \
                "\n".join(f"- {q}" for q in sample)
    
    prompt = _GEN_PROMPT.format(
        window=EVENT_WINDOW,
        facts=facts + avoid,
    )
    raw = _gen_call(prompt)
    if not raw:
        return None
    cleaned = re.sub(r"```(json)?", "", raw).strip()
    try:
        data = json.loads(cleaned)
        if all(k in data for k in ("question", "correct_answer",
                                    "event_date", "source_fact")):
            return data
    except json.JSONDecodeError:
        logger.warning("JSON parse failed for %s", subcode)
    return None


# ----------------------------------------------------------------------
# QC filtreleri
# ----------------------------------------------------------------------

def run_qc(item: PCQItem, seen_questions: set, seen_answers: set) -> PCQItem:
    notes = []
    if len(item.question.split()) < 5:
        notes.append("question too short")
    if not item.correct_answer or len(item.correct_answer.strip()) < 1:
        notes.append("empty correct_answer")
    if len(item.correct_answer.split()) > 15:
        notes.append("answer too long (>15 words)")
    # Tekrar kontrolu (seen_questions/seen_answers TUM kategoriler arasinda
    # paylasilir; farkli alt kategoriler ayni kaynak olayi sormasin)
    q_norm = item.question.lower().strip()
    if q_norm in seen_questions:
        notes.append("duplicate question")
    ans_norm = item.correct_answer.lower().strip()
    if ans_norm in seen_answers:
        notes.append("duplicate answer (same underlying fact already used)")
    # Cevap soruda gizleniyor mu?
    if item.correct_answer.lower() in item.question.lower():
        notes.append("answer appears in question")
    item.qc_notes = notes
    item.qc_passed = len(notes) == 0
    return item


# ----------------------------------------------------------------------
# Tek uretim + toplu uretim
# ----------------------------------------------------------------------

def build_one(subcode: str, idx: int,
              seen_questions: set, seen_answers: set) -> Optional[PCQItem]:
    raw = generate_raw_item(subcode, seen_questions)
    if not raw:
        return None
    item = PCQItem(
        question_id=f"{subcode}-{idx:03d}",
        subcategory=subcode,
        question=raw["question"].strip(),
        correct_answer=raw["correct_answer"].strip(),
        event_date=raw["event_date"].strip(),
        source_fact=raw["source_fact"].strip(),
    )
    item = run_qc(item, seen_questions, seen_answers)
    status = "PASS" if item.qc_passed else f"FAIL({'; '.join(item.qc_notes)})"
    logger.info("[%s] %s | Q: %s", item.question_id, status,
                item.question[:60])
    return item


def build_dataset(per_subcategory: int = 100,
                  out_path: str = "PCQ_dataset.json") -> list:
    """5 alt kategori x 100 = 500 PCQ sorusu hedefi."""
    all_items = []
    # Kategoriler arasi paylasilan setler: SOURCE_FACTS kategorileri ortak
    # olaylar icerdiginden (orn. Venezuela depremi PCQ-ECO ve PCQ-WOR'da da
    # var), dedup tek bir alt kategoriyle sinirli kalamaz.
    seen_q = set()
    seen_answers = set()
    for subcode in PCQ_DOMAINS:
        collected, attempts = 0, 0
        max_attempts = per_subcategory * 6
        while collected < per_subcategory and attempts < max_attempts:
            attempts += 1
            item = build_one(subcode, collected + 1, seen_q, seen_answers)
            if item and item.qc_passed:
                seen_q.add(item.question.lower().strip())
                seen_answers.add(item.correct_answer.lower().strip())
                all_items.append(item)
                collected += 1
        logger.info("Subcategory %s: %d/%d in %d attempts",
                    subcode, collected, per_subcategory, attempts)

    serialised = [asdict(it) for it in all_items]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialised, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d PCQ items -> %s", len(serialised), out_path)
    return all_items


if __name__ == "__main__":
    import random; random.seed(SEED)
    logger.info("PCQ smoke test | Pencere: %s | Uretici: Mistral-Large(ASU)", EVENT_WINDOW)
    logger.info("Kaynak: Gercek olaylar (FIFA WC, NATO, AI/Tech, IMF, Dunya)")
    if not os.environ.get("ASU_CREATEAI_TOKEN"):
        logger.info("ASU_CREATEAI_TOKEN yok; import OK.")
    else:
        # Her kategoriden 2 soru smoke test
        build_dataset(per_subcategory=2, out_path="PCQ_smoke.json")
