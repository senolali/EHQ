import unittest

from ehq.analysis.construct import build_construct_scope_sensitivity


def record(model, category, label, confidence):
    return {
        "model": model,
        "category": category,
        "k": 0,
        "valid_for_ehq12": True,
        "valid_for_ehq3": True,
        "parsed_confidence": confidence,
        "confidence_terminal": False,
        "classification": {
            "label": label,
            "is_correct": label == "CONFIDENT_CORRECT",
        },
    }


class ConstructScopeSensitivityTests(unittest.TestCase):
    def test_reports_scope_correlations_and_hnq_contribution(self):
        rows = [
            record("A", "FEQ", "HEDGE", 0.1),
            record("A", "PCQ", "CONFIDENT_CORRECT", 0.8),
            record("A", "HNQ", "CONFIDENT_CORRECT", 0.9),
            record("A", "CCQ", "CONFIDENT_WRONG", 0.3),
            record("B", "FEQ", "CONFIDENT_WRONG", 0.9),
            record("B", "PCQ", "CONFIDENT_CORRECT", 0.6),
            record("B", "HNQ", "CONFIDENT_WRONG", 0.8),
            record("B", "CCQ", "CONFIDENT_WRONG", 0.6),
            record("C", "FEQ", "ABSTAIN", 0.0),
            record("C", "PCQ", "CONFIDENT_WRONG", 0.7),
            record("C", "HNQ", "CONFIDENT_CORRECT", 0.5),
            record("C", "CCQ", "CONFIDENT_WRONG", 0.9),
        ]
        result = build_construct_scope_sensitivity(
            rows,
            models=["A", "B", "C"],
            weights={"ehq1": 0.30, "ehq2": 0.45, "ehq3": 0.25},
        )
        by_id = {row["scenario"]: row for row in result["scenarios"]}
        self.assertEqual(len(by_id), 5)
        self.assertEqual(by_id["official"]["pearson_with_official"], 1.0)
        self.assertEqual(
            by_id["strict_unavailability"]["categories"],
            ["FEQ", "PCQ", "CCQ"],
        )
        self.assertEqual(result["confident_correct_by_category"]["HNQ"], 2)
        self.assertAlmostEqual(result["hnq_share_of_confident_correct"], 0.5)
        self.assertEqual(
            len(by_id["context_boundary"]["model_scores_and_ranks"]), 3
        )

    def test_rejects_missing_model_records(self):
        with self.assertRaisesRegex(ValueError, "No retained records"):
            build_construct_scope_sensitivity(
                [record("A", "CCQ", "CONFIDENT_WRONG", 0.5)],
                models=["A", "B"],
                weights={"ehq1": 0.30, "ehq2": 0.45, "ehq3": 0.25},
            )


if __name__ == "__main__":
    unittest.main()
