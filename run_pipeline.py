"""
EHQ-3000 Evaluation Pipeline
==============================
20 model x 3000 soru (FEQ+PCQ+HNQ+CCQ) uzerinde EHQ1/EHQ2/EHQ3/EHQ
hesaplayan ana calistirma scripti.

EHQ v1 makalesinde kullanilan classifier/scorer/exporter/visualizer
mantigi (RAR arsivinden cikarilan src/*.py + config.py) framework/
altina portlandi; bu script (run_pipeline.py, arsivde YOKTU) o
modulleri ASU CreateAI + DeepSeek uzerinden calistiran YENI orkestrasyon
katmanidir.

Kullanim:
  python run_pipeline.py --models all --categories all
  python run_pipeline.py --models "Claude-4.5-Haiku,GPT-5-mini" --categories all
  python run_pipeline.py --models all --categories FEQ,CCQ --limit 40
  python run_pipeline.py --dry-run --limit 10     # API cagrisi yok, sadece yapi testi

Ortam degiskenleri:
  ASU_CREATEAI_TOKEN   -> type: "asu" modeller icin (18 model)
  DEEPSEEK_API_KEY     -> type: "deepseek" modeller icin (2 model)
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from framework import config_scoring, classifier, scorer, model_client, prompt_builder, exporter, visualizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ehq_pipeline")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(path: str, categories: list, limit: int = None, seed: int = 42) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if categories:
        data = [d for d in data if d["category"] in categories]
    if limit:
        data = list(data)
        random.Random(seed).shuffle(data)
        data = data[:limit]
    return data


def select_models(all_models: list, wanted: str) -> list:
    if wanted == "all":
        return all_models
    names = {n.strip() for n in wanted.split(",")}
    selected = [m for m in all_models if m["name"] in names]
    missing = names - {m["name"] for m in selected}
    if missing:
        raise ValueError(f"config'te bulunamayan model adi/adlari: {sorted(missing)}")
    return selected


def evaluate_item(model_cfg: dict, item: dict, dry_run: bool) -> dict:
    prompt = prompt_builder.build_prompt(item)

    if dry_run:
        out = {"answer": "[DRY_RUN] This is a placeholder answer.", "conf_raw": "50"}
    else:
        out = model_client.query_with_confidence(model_cfg, prompt)

    result = classifier.process_response(
        response=out["answer"],
        confidence_response=out["conf_raw"],
        correct_answer=item["correct_answer"],
        category=item["category"],
        abstain_patterns=config_scoring.ALL_ABSTAIN_PATTERNS,
        hedge_patterns=config_scoring.ALL_HEDGE_PATTERNS,
        scale_max=config_scoring.CONFIDENCE_SCALE_MAX,
    )
    result["rubric_score"] = scorer.get_rubric_score(
        result["response_type"], item["category"],
        config_scoring.RUBRIC, config_scoring.CATEGORY_INDEX)
    result.update({
        "question_id": item["question_id"],
        "model": model_cfg["name"],
        "category": item["category"],
        "subcategory": item["subcategory"],
    })
    return result


def evaluate_model(model_cfg: dict, dataset: list, dry_run: bool,
                   results_dir: Path, timestamp: str, categories: list) -> dict:
    name = model_cfg["name"]
    max_workers = model_cfg.get("params", {}).get("max_workers", 4)
    logger.info("=== %s | %d soru | %d worker ===", name, len(dataset), max_workers)

    raw_results = [None] * len(dataset)
    n_skipped = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(evaluate_item, model_cfg, item, dry_run): i
                  for i, item in enumerate(dataset)}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            item = dataset[i]
            try:
                r = fut.result()
                if r["response_raw"] == "":   # process_response: None yanit -> ""
                    n_skipped += 1
            except Exception as e:
                logger.error("[%s] item %d (%s) basarisiz: %s",
                             name, i, item.get("question_id", "?"), e)
                r = {
                    "question_id": item["question_id"], "model": name,
                    "category": item["category"], "subcategory": item["subcategory"],
                    "response_type": "ABSTAIN", "is_correct": False,
                    "confidence": 0.5, "response_raw": "[API_ERROR]",
                    "conf_raw": "", "rubric_score": 1.0,
                }
                n_skipped += 1
            raw_results[i] = r
            done += 1
            if done % 50 == 0 or done == len(dataset):
                logger.info("[%s] %d/%d tamamlandi (%.0fs)",
                            name, done, len(dataset), time.time() - t0)

    ehq = scorer.compute_ehq(raw_results, config_scoring.EHQ_WEIGHTS, config_scoring.CONFIDENCE_BINS)
    summary = {
        "model": name,
        "model_type": model_cfg["type"],
        "n_questions": len(dataset),
        "n_skipped": n_skipped,
        **ehq,
        "EHQ2_by_category": scorer.compute_ehq2_by_category(raw_results, categories),
        "response_distribution": scorer.compute_response_distribution(raw_results),
        "raw_results": raw_results,
    }

    interim_path = results_dir / f"interim_{name}_{timestamp}.json"
    with open(interim_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("[%s] EHQ=%.3f (EHQ1=%.3f EHQ2=%.3f EHQ3=%.3f) -> %s",
               name, ehq["EHQ"], ehq["EHQ1"], ehq["EHQ2"], ehq["EHQ3"], interim_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="EHQ-3000 Evaluation Pipeline")
    parser.add_argument("--config", default="config_ehq_20models.yaml")
    parser.add_argument("--dataset", default=None,
                        help="Varsayilan: config'teki datasets[0].params.path")
    parser.add_argument("--models", default="all",
                        help="Virgulle ayrilmis model adi listesi veya 'all'")
    parser.add_argument("--categories", default="all",
                        help="Virgulle ayrilmis kategori listesi (FEQ,PCQ,HNQ,CCQ) veya 'all'")
    parser.add_argument("--limit", type=int, default=None,
                        help="Toplam soru sayisini sinirlar (hizli test icin)")
    parser.add_argument("--output-dir", default=None,
                        help="Varsayilan: config'teki experiment.output_dir")
    parser.add_argument("--dry-run", action="store_true",
                        help="API cagrisi yapmadan yapi/akisi test et")
    args = parser.parse_args()

    cfg = load_config(args.config)
    categories = config_scoring.CATEGORIES if args.categories == "all" \
        else [c.strip() for c in args.categories.split(",")]
    unknown = set(categories) - set(config_scoring.CATEGORIES)
    if unknown:
        raise ValueError(f"Bilinmeyen kategori(ler): {sorted(unknown)} "
                         f"(gecerli: {config_scoring.CATEGORIES})")

    dataset_path = args.dataset or cfg["datasets"][0]["params"]["path"]
    dataset = load_dataset(dataset_path, categories, limit=args.limit,
                           seed=cfg.get("experiment", {}).get("seed", 42))
    if not dataset:
        raise ValueError(f"Dataset bos: {dataset_path} (kategoriler: {categories})")

    models = select_models(cfg["models"], args.models)
    if not models:
        raise ValueError("Secilen kriterlere uyan model yok.")

    output_dir = Path(args.output_dir or cfg.get("experiment", {}).get("output_dir", "outputs_ehq"))
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("DRY RUN: API cagrisi YAPILMAYACAK.")
    else:
        for m in models:
            if m["type"] == "asu" and not os.environ.get("ASU_CREATEAI_TOKEN"):
                raise RuntimeError(f"ASU_CREATEAI_TOKEN tanimli degil (gerekli: {m['name']})")
            if m["type"] == "deepseek" and not os.environ.get(
                    m["params"].get("api_key_env", "DEEPSEEK_API_KEY")):
                raise RuntimeError(f"DEEPSEEK_API_KEY tanimli degil (gerekli: {m['name']})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("EHQ-3000 Pipeline | %d model | %d soru (%s) | timestamp=%s",
               len(models), len(dataset), ",".join(categories), timestamp)

    all_results = {}
    for model_cfg in models:
        try:
            all_results[model_cfg["name"]] = evaluate_model(
                model_cfg, dataset, args.dry_run, results_dir, timestamp, categories)
        except Exception as e:
            logger.error("Model %s TAMAMEN basarisiz: %s", model_cfg["name"], e, exc_info=True)

    full_path = results_dir / f"ehq_full_{timestamp}.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    summary_only = {
        name: {k: v for k, v in d.items() if k != "raw_results"}
        for name, d in all_results.items()
    }
    summary_path = results_dir / f"ehq_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_only, f, indent=2, ensure_ascii=False)

    excel_path = output_dir / f"EHQ_3000_results_{timestamp}.xlsx"
    exporter.export_to_excel(all_results, str(excel_path), categories)
    visualizer.generate_all_figures(all_results, str(figures_dir / f"figures_{timestamp}"), categories)

    logger.info("TAMAMLANDI. Sonuclar: %s | Ozet: %s | Excel: %s",
               full_path, summary_path, excel_path)


if __name__ == "__main__":
    main()
