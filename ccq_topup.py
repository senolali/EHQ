"""
CCQ Top-up Module
==================
Bir onceki --full calistirmasi CCQ-LEG (104/150) ve CCQ-MED (147/150)
alt kategorilerini hedefin altinda birakti (muhtemel neden: bu turlerde
critical_fact belgeye BIREBIR yerlesmeme veya QC/duplicate red orani
daha yuksek). Bu script mevcut CCQ_dataset.json'daki 701 soruyu KORUR,
her alt kategori icin sadece EKSIK kadar yeni soru uretir ve birlestirir.

Cakisma onleme: mevcut dataset'teki TUM redacted_value/question degerleri
baslangicta seen_facts/seen_questions kumelerine yuklenir, boylece yeni
uretilen sorular onceki 701 ile cakismaz (ccq_generator.py'nin
build_one/run_qc/qc_duplicate mekanizmasi aynen kullanilir).

Kullanim:
    python ccq_topup.py --in CCQ_dataset.json --out CCQ_dataset.json
"""

import os
import json
import random
import logging
import argparse
from dataclasses import asdict

from ccq_generator import CCQ_SUBCATEGORIES, build_one, SEED

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ccq_topup")


def topup(in_path: str, out_path: str, target_per_subcategory: int = 150) -> list:
    with open(in_path, encoding="utf-8") as f:
        existing = json.load(f)

    seen_facts = {it["redacted_value"].strip().lower() for it in existing}
    seen_questions = {it["question"].strip().lower() for it in existing}

    counts, max_idx = {}, {}
    for subcode in CCQ_SUBCATEGORIES:
        counts[subcode] = 0
        max_idx[subcode] = 0
    for it in existing:
        sub = it["subcategory"]
        counts[sub] = counts.get(sub, 0) + 1
        idx = int(it["question_id"].rsplit("-", 1)[-1])
        max_idx[sub] = max(max_idx.get(sub, 0), idx)

    new_items = []
    for subcode in CCQ_SUBCATEGORIES:
        need = target_per_subcategory - counts.get(subcode, 0)
        if need <= 0:
            logger.info("Subcategory %s already at target (%d/%d)",
                        subcode, counts.get(subcode, 0), target_per_subcategory)
            continue
        logger.info("Subcategory %s needs %d more (currently %d/%d)",
                    subcode, need, counts.get(subcode, 0), target_per_subcategory)
        collected, attempts = 0, 0
        # onceki --full calistirmasinda bu kategoriler icin basari orani
        # dusuk cikti (orn. CCQ-LEG ~%17); eksigi TAMAMLAMAYI garanti
        # etmek icin buyuk bir deneme payi birakiyoruz.
        max_attempts = max(need * 15, 150)
        next_idx = max_idx.get(subcode, 0)
        while collected < need and attempts < max_attempts:
            attempts += 1
            next_idx += 1
            item = build_one(subcode, next_idx, seen_facts, seen_questions)
            if item and item.qc_passed:
                seen_facts.add(item.redacted_value.strip().lower())
                seen_questions.add(item.question.strip().lower())
                new_items.append(item)
                collected += 1
        logger.info("Subcategory %s: +%d/%d collected in %d attempts",
                    subcode, collected, need, attempts)
        if collected < need:
            logger.warning("Subcategory %s STILL short by %d after top-up "
                           "-- re-run this script to try again", subcode, need - collected)

    merged = existing + [asdict(it) for it in new_items]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d total CCQ items -> %s (was %d, added %d)",
                len(merged), out_path, len(existing), len(new_items))
    return merged


if __name__ == "__main__":
    random.seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="CCQ_dataset.json",
                        help="Mevcut (eksik) CCQ dataset dosyasi")
    parser.add_argument("--out", dest="out_path", default="CCQ_dataset.json",
                        help="Cikti dosyasi (varsayilan: ayni dosyanin uzerine yazar)")
    parser.add_argument("--target", type=int, default=150,
                        help="Alt kategori basina hedef soru sayisi (varsayilan 150)")
    args = parser.parse_args()

    if not os.environ.get("ASU_CREATEAI_TOKEN"):
        logger.info("ASU_CREATEAI_TOKEN yok; import OK.")
    else:
        topup(args.in_path, args.out_path, target_per_subcategory=args.target)
