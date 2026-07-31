"""Experiment tracker: per-model incremental JSON saves + Excel/figure export.

Adapted from senolali/RQEval (utils/experiment_tracker.py). RQEval's
version builds its own 4-sheet Excel report inline (6 reasoning-quality
metrics); EHQ already has a validated, EHQ-specific report generator
(framework/exporter.py, framework/visualizer.py, ported from the EHQ v1
paper's src/exporter.py + src/visualizer.py) so this tracker delegates
to those instead of re-implementing sheet layout. What's kept from
RQEval: per-model JSON saved to a stable experiment directory
IMMEDIATELY after each model finishes (not just at the very end), and
a final summary.json across all models.
"""

import json
import os
import time
from typing import Any, Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentTracker:

    def __init__(self, experiment_id: str, output_dir: str = "outputs_ehq"):
        self.experiment_id = experiment_id
        self.output_dir    = output_dir
        self.exp_dir       = os.path.join(output_dir, experiment_id)
        os.makedirs(self.exp_dir, exist_ok=True)
        self._model_results: List[Dict[str, Any]] = []
        self._start_time    = time.time()

    def log_model_result(self, model_name: str, result: Dict[str, Any]) -> None:
        """Append a model's result and save it to disk immediately -- if the
        run is killed partway through the model list, everything completed
        so far is still on disk (not just held in RAM until the end)."""
        self._model_results.append(result)
        os.makedirs(self.exp_dir, exist_ok=True)
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        path = os.path.join(self.exp_dir, f"{safe_name}_result.json")
        # raw_results can be large (up to 3000 items); still worth persisting
        # per-model for post-hoc inspection/debugging.
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {path}")

    def save_summary(self, results: List[Dict[str, Any]]) -> str:
        summary = {
            "experiment_id":    self.experiment_id,
            "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_seconds": round(time.time() - self._start_time, 2),
            "num_models":       len(results),
            "results": [
                {k: v for k, v in r.items() if k != "raw_results"}
                for r in results
            ],
        }
        path = os.path.join(self.exp_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved: {path}")
        return path

    # ------------------------------------------------------------------
    # Delegated report generation (framework/exporter.py, framework/visualizer.py)
    # ------------------------------------------------------------------

    def export_excel(self, results: List[Dict[str, Any]], filename: str,
                     categories: List[str]) -> str:
        from framework.exporter import export_to_excel
        all_results = {r["model"]: r for r in results}
        path = os.path.join(self.exp_dir, filename)
        return export_to_excel(all_results, path, categories)

    def export_figures(self, results: List[Dict[str, Any]],
                       categories: List[str]) -> list:
        from framework.visualizer import generate_all_figures
        all_results = {r["model"]: r for r in results}
        figures_dir = os.path.join(self.exp_dir, "figures")
        return generate_all_figures(all_results, figures_dir, categories)
