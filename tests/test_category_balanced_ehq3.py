import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "analyze_category_balanced_ehq3.py"
SPEC = importlib.util.spec_from_file_location("analyze_category_balanced_ehq3", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(model: str, category: str, confidence: float, correct: bool):
    label = "CONFIDENT_CORRECT" if correct else "CONFIDENT_WRONG"
    return {
        "model": model,
        "category": category,
        "k": 0,
        "valid_for_ehq12": True,
        "valid_for_ehq3": True,
        "confidence_terminal": False,
        "parsed_confidence": confidence,
        "classification": {"label": label, "is_correct": correct},
    }


class CategoryBalancedEHQ3Tests(unittest.TestCase):
    def test_equal_category_weighting_differs_from_pooled_weighting(self):
        records = []
        for model in ("Model-A", "Model-B"):
            for category in MODULE.CATEGORIES:
                records.append(_row(model, category, 0.9, category == "HNQ"))
        # Model A contributes many additional well-calibrated HNQ answers, so
        # its official pooled EHQ3 is pulled toward HNQ while the balanced
        # sensitivity still assigns one quarter to each category.
        records.extend(_row("Model-A", "HNQ", 0.9, True) for _ in range(20))
        summary, rows = MODULE.analyse(
            records,
            weights={"ehq1": 0.30, "ehq2": 0.45, "ehq3": 0.25},
            n_bins=10,
            excluded_models=set(),
        )
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["n_models"], 2)
        model_a = next(row for row in rows if row["model"] == "Model-A")
        self.assertNotAlmostEqual(
            model_a["official_EHQ3"], model_a["category_balanced_EHQ3"]
        )
        self.assertEqual(model_a["HNQ_n_calibration"], 21)


if __name__ == "__main__":
    unittest.main()
