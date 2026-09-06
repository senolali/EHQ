import unittest

from ehq.analysis.report import build_analysis_report
from ehq.types import ModelSpec


def _result(ehq, ehq1, ehq2, ehq3):
    scores = {"EHQ": ehq, "EHQ1": ehq1, "EHQ2": ehq2, "EHQ3": ehq3}
    return {
        "scores": scores,
        "category_scores": {"FEQ": scores},
    }


def _category_result(value, offset=0.0):
    scores = {"EHQ": value, "EHQ1": value, "EHQ2": value, "EHQ3": value}
    return {
        "scores": scores,
        "category_scores": {
            "FEQ": {**scores, "EHQ": value},
            "PCQ": {**scores, "EHQ": value + offset},
            "HNQ": {**scores, "EHQ": 1.0 - value},
            "CCQ": {**scores, "EHQ": 0.2 + offset},
        },
    }


class AnalysisReportTests(unittest.TestCase):
    def test_rq1_and_rq2_are_derived_without_claim_invention(self):
        models = [
            ModelSpec("A-old", "asu", "a1", pair="A", generation="old"),
            ModelSpec("A-new", "asu", "a2", pair="A", generation="new"),
            ModelSpec("B-old", "asu", "b1", pair="B", generation="old"),
            ModelSpec("B-new", "asu", "b2", pair="B", generation="new"),
        ]
        results = {
            "A-old": _result(0.4, 0.3, 0.5, 0.4),
            "A-new": _result(0.6, 0.5, 0.7, 0.6),
            "B-old": _result(0.5, 0.4, 0.6, 0.5),
            "B-new": _result(0.7, 0.6, 0.8, 0.7),
        }
        without_cq = build_analysis_report(results, models, seed=42)
        self.assertEqual(
            without_cq["rq1_capability_relationship"]["status"],
            "capability_scores_not_provided",
        )
        self.assertEqual(
            without_cq["rq2_generational_pairs"]["summary"]["improved_pairs"],
            2,
        )
        component = without_cq["component_correlations"]["EHQ1_vs_EHQ2"]
        self.assertEqual(component["inference_unit"], "model")
        self.assertIn("holm_adjusted_p", component["pearson"])
        self.assertIn("permutation_test", component["spearman"])
        with_cq = build_analysis_report(
            results,
            models,
            seed=42,
            capability_scores={
                "A-old": 0.9,
                "A-new": 0.8,
                "B-old": 0.7,
                "B-new": 0.6,
            },
        )
        self.assertEqual(with_cq["rq1_capability_relationship"]["status"], "ok")

    def test_category_correlations_are_formal_and_exploratory(self):
        models = [ModelSpec(str(i), "asu", str(i)) for i in range(4)]
        results = {
            str(i): _category_result(0.2 + i * 0.15, i * 0.01)
            for i in range(4)
        }
        report = build_analysis_report(results, models, seed=42)
        self.assertEqual(len(report["category_correlations"]), 6)
        row = report["category_correlations"]["FEQ_vs_PCQ"]
        self.assertEqual(row["n"], 4)
        self.assertIn("holm_adjusted_p", row["pearson"])
        self.assertEqual(
            report["category_correlation_multiplicity"]["status"],
            "exploratory",
        )
        sensitivity = report["composite_weight_sensitivity"]
        self.assertEqual(sensitivity["status"], "descriptive_sensitivity_analysis")
        self.assertEqual(len(sensitivity["scenarios"]), 3)
        self.assertIn(
            "maximum_absolute_rank_shift", sensitivity["scenarios"][0]
        )

    def test_report_handles_models_without_a_defined_composite(self):
        models = [
            ModelSpec("A", "asu", "a"),
            ModelSpec("B", "asu", "b"),
        ]
        results = {
            "A": _result(None, 1.0, 1.0, None),
            "B": _result(None, 0.9, 0.9, None),
        }
        report = build_analysis_report(results, models, seed=42)
        self.assertEqual(report["n_models_with_valid_ehq"], 0)
        for scenario in report["composite_weight_sensitivity"]["scenarios"]:
            self.assertEqual(scenario["status"], "no_complete_scores")
            self.assertIsNone(scenario["maximum_absolute_rank_shift"])
