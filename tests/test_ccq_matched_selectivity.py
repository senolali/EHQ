import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "analyze_ccq_matched_selectivity.py"
SPEC = importlib.util.spec_from_file_location("analyze_ccq_matched_selectivity", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CCQMatchedSelectivityTests(unittest.TestCase):
    def test_matched_outcomes_are_counted(self):
        full = [
            {
                "model": "Model-A",
                "category": "CCQ",
                "question_id": "CCQ-1",
                "valid_for_ehq12": True,
                "classification": {"label": "ABSTAIN"},
            },
            {
                "model": "Model-A",
                "category": "CCQ",
                "question_id": "CCQ-2",
                "valid_for_ehq12": True,
                "classification": {"label": "CONFIDENT_WRONG"},
            },
        ]
        restored = [
            {
                "model": "Model-A",
                "source_question_id": "CCQ-1",
                "scored": True,
                "is_correct": True,
            },
            {
                "model": "Model-A",
                "source_question_id": "CCQ-2",
                "scored": True,
                "is_correct": False,
            },
        ]
        rows = MODULE.analyse(full, restored, excluded_models=set())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_paired"], 2)
        self.assertEqual(rows[0]["both_pass_count"], 1)
        self.assertEqual(rows[0]["both_fail_count"], 1)
        self.assertEqual(rows[0]["joint_pass_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
