"""
EHQ-3000 evaluation orchestrator.

Architecture adapted from senolali/RQEval (evaluation/evaluator.py):
item-level checkpointing (utils/checkpoint.ItemCheckpoint) so a crash
partway through a model's 3000 items resumes from where it left off
instead of re-paying for every call, a ThreadPoolExecutor sized per
model config, and an incremental per-model save (Excel + figures
regenerated after each model finishes, not just at the very end).

What's EHQ-specific (no RQEval equivalent): the actual per-item
work -- build the category-aware prompt (framework/prompt_builder.py),
run the 2-turn answer+confidence protocol, classify the response
(framework/classifier.py), and score EHQ1/EHQ2/EHQ3 (framework/scorer.py)
-- replaces RQEval's 6-metric (correctness/consistency/robustness/
logical-coherence/efficiency/stability) computation.
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from models.base_model import BaseModel
from framework import classifier, scorer, config_scoring, prompt_builder
from utils.logger import get_logger
from utils.experiment_tracker import ExperimentTracker
from utils.checkpoint import ItemCheckpoint, item_key as ck_item_key

logger = get_logger(__name__)


class Evaluator:
    """Full pipeline: load -> generate (2-turn) -> classify -> score -> export."""

    def __init__(
        self,
        config: Dict[str, Any],
        categories: List[str],
        output_dir: str = "outputs_ehq",
        experiment_id: Optional[str] = None,
    ):
        self.config        = config
        self.categories    = categories
        self.output_dir    = output_dir
        self.experiment_id = experiment_id or f"ehq_{int(time.time())}"
        # Top-level config key (sibling of "experiment:"), matching
        # senolali/RQEval's convention -- one pool size for all API
        # models (not per-model params.max_workers, which config files
        # may still document for humans but the evaluator ignores).
        self.max_workers    = config.get("max_workers", 4)
        self.tracker        = ExperimentTracker(
            experiment_id=self.experiment_id, output_dir=output_dir)

    # ------------------------------------------------------------------
    # Per-item work
    # ------------------------------------------------------------------

    def _process_item(self, model: BaseModel, item: Dict[str, Any],
                      idx: int, total: int, log_lock: threading.Lock) -> Dict[str, Any]:
        def log(msg):
            with log_lock:
                logger.info(msg)

        prompt = prompt_builder.build_prompt(item)
        failed = False
        try:
            out = model.generate_with_confidence(prompt)
            answer, conf_raw = out["answer"], out["conf_raw"]
        except Exception as e:
            log(f"    [{idx+1}/{total}] {item['question_id']} FAILED: {e}")
            failed = True
            answer, conf_raw = None, None

        result = classifier.process_response(
            response=answer,
            confidence_response=conf_raw,
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
            "model": model.name,
            "category": item["category"],
            "subcategory": item["subcategory"],
            "generation_failed": failed,
        })
        if not failed and (idx + 1) % 50 == 0:
            log(f"    [{idx+1}/{total}] {result['response_type']}")
        return result

    # ------------------------------------------------------------------
    # Checkpoint fingerprint
    # ------------------------------------------------------------------

    def _get_checkpoint(self, model: BaseModel) -> Optional[ItemCheckpoint]:
        """Stable path: <output_dir>/checkpoints/<experiment_name>/<model>.jsonl
        Fingerprint covers everything that changes item content or scoring
        rules; a mismatch invalidates the old checkpoint automatically (no
        manual version bump to remember). Disable via config:
        experiment: { checkpoint: false }"""
        exp_cfg = self.config.get("experiment", {})
        if exp_cfg.get("checkpoint", True) is False:
            return None
        import re
        exp_name = exp_cfg.get("name", "ehq_experiment")
        safe_exp = re.sub(r"[^A-Za-z0-9_\-]", "_", exp_name)
        safe_mod = re.sub(r"[^A-Za-z0-9_\-]", "_", model.name)
        path = os.path.join(self.output_dir, "checkpoints", safe_exp, f"{safe_mod}.jsonl")

        scalar_model_params = {}
        for attr in ("model_id", "model_name", "model_provider", "base_url",
                    "max_tokens", "temperature", "_temperature",
                    "deterministic", "system_prompt"):
            value = getattr(model, attr, None)
            if isinstance(value, (str, int, float, bool)) or value is None:
                scalar_model_params[attr] = value

        fingerprint = {
            "model": model.name,
            "model_params": scalar_model_params,
            "categories": sorted(self.categories),
            "seed": exp_cfg.get("seed"),
            # Any scoring-rule change (rubric, abstain/hedge patterns,
            # weights) automatically invalidates old checkpoints --
            # no manual "protocol version" string to remember to bump.
            "ehq_weights": config_scoring.EHQ_WEIGHTS,
            "rubric": config_scoring.RUBRIC,
            "abstain_patterns": config_scoring.ALL_ABSTAIN_PATTERNS,
            "hedge_patterns": config_scoring.ALL_HEDGE_PATTERNS,
            "confidence_scale_max": config_scoring.CONFIDENCE_SCALE_MAX,
        }
        return ItemCheckpoint(path, fingerprint, enabled=True)

    # ------------------------------------------------------------------
    # Single model, all items
    # ------------------------------------------------------------------

    def evaluate_model(self, model: BaseModel, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        name = model.name
        total = len(dataset)
        max_workers = self.max_workers
        logger.info(f"\n{'='*60}")
        logger.info(f"  Model : {name}")
        logger.info(f"  Items : {total} | workers={max_workers}")
        logger.info(f"{'='*60}")

        ckpt = self._get_checkpoint(model)
        keys = [ck_item_key(item, idx) for idx, item in enumerate(dataset)]

        raw_results: List[Optional[Dict[str, Any]]] = [None] * total
        pending = []
        for idx in range(total):
            cached_entry = ckpt.get(keys[idx]) if ckpt is not None else None
            if cached_entry is not None:
                raw_results[idx] = cached_entry
            else:
                pending.append(idx)

        if ckpt is not None and len(pending) < total:
            logger.info(f"  [Checkpoint] {total - len(pending)}/{total} items restored "
                       f"— running remaining {len(pending)}.")

        log_lock = threading.Lock()
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._process_item, model, dataset[idx], idx, total, log_lock): idx
                for idx in pending
            }
            done = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    entry = fut.result()
                except Exception as e:
                    item = dataset[idx]
                    logger.error(f"  [Worker error idx={idx} {item.get('question_id','?')}]: {e}")
                    entry = {
                        "question_id": item["question_id"], "model": name,
                        "category": item["category"], "subcategory": item["subcategory"],
                        "response_type": "ABSTAIN", "is_correct": False,
                        "confidence": 0.5, "response_raw": "", "conf_raw": "",
                        "rubric_score": 1.0, "generation_failed": True,
                    }
                raw_results[idx] = entry
                if ckpt is not None:
                    ckpt.add(keys[idx], entry)
                done += 1
                if done % 100 == 0 or done == len(pending):
                    logger.info(f"  [{name}] {done}/{len(pending)} new items "
                               f"({time.time()-t0:.0f}s)")
        if ckpt is not None:
            ckpt.close()

        n_skipped = sum(1 for r in raw_results if r.get("generation_failed"))
        ehq = scorer.compute_ehq(raw_results, config_scoring.EHQ_WEIGHTS, config_scoring.CONFIDENCE_BINS)
        result = {
            "model": name,
            "model_type": getattr(model, "config", {}).get("type", "-"),
            "n_questions": total,
            "n_skipped": n_skipped,
            **ehq,
            "EHQ2_by_category": scorer.compute_ehq2_by_category(raw_results, self.categories),
            "response_distribution": scorer.compute_response_distribution(raw_results),
            "raw_results": raw_results,
        }
        logger.info(f"  → EHQ={ehq['EHQ']:.3f} (EHQ1={ehq['EHQ1']:.3f} "
                   f"EHQ2={ehq['EHQ2']:.3f} EHQ3={ehq['EHQ3']:.3f}) | skipped={n_skipped}")

        self.tracker.log_model_result(name, result)
        self._save_incremental()
        return result

    # ------------------------------------------------------------------
    # Incremental per-model save (Excel + figures, cumulative)
    # ------------------------------------------------------------------

    def _save_incremental(self) -> None:
        """Regenerate Excel/figures after each model completes -- files are
        overwritten each call so outputs/ always reflects the latest
        cumulative results, safe to inspect mid-run."""
        try:
            completed = self.tracker._model_results
            self.tracker.export_excel(completed, "EHQ_3000_results.xlsx", self.categories)
            self.tracker.export_figures(completed, self.categories)
            logger.info(f"  [Saved] EHQ_3000_results.xlsx + figures/ ({len(completed)} model(s) so far)")
        except Exception as e:
            logger.warning(f"  [Incremental save failed] {e}")

    # ------------------------------------------------------------------
    # Multi-model pipeline
    # ------------------------------------------------------------------

    def evaluate_all(self, models: List[BaseModel],
                     dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"\nEHQ-3000 evaluation pipeline")
        logger.info(f"  Models  : {[m.name for m in models]}")
        logger.info(f"  Samples : {len(dataset)}")

        results = []
        for i, model in enumerate(models):
            logger.info(f"\n[{i+1}/{len(models)}] {model.name}")
            results.append(self.evaluate_model(model, dataset))

        self.tracker.save_summary(results)
        return results
