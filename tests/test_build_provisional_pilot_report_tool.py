import importlib.util
from pathlib import Path
import unittest

from ehq.evaluation.scoring import compute_ehq_scores


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_provisional_pilot_report",
    ROOT / "tools" / "build_provisional_pilot_report.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BuildProvisionalPilotReportToolTests(unittest.TestCase):
    def test_stratified_bootstrap_is_deterministic_and_stays_within_strata(self):
        strata = ["A", "A", "B", "B"]
        first = MODULE._stratified_bootstrap_indices(
            strata, n_resamples=20, seed=7
        )
        second = MODULE._stratified_bootstrap_indices(
            strata, n_resamples=20, seed=7
        )
        self.assertTrue((first == second).all())
        self.assertTrue(((first[:, :2] == 0) | (first[:, :2] == 1)).all())
        self.assertTrue(((first[:, 2:] == 2) | (first[:, 2:] == 3)).all())

    def test_percentile_interval_contains_center(self):
        low, high = MODULE._percentile_interval([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertLess(low, 0.3)
        self.assertGreater(high, 0.3)


WEIGHTS = {"ehq1": 0.30, "ehq2": 0.45, "ehq3": 0.25}


def _record(model, index, label, confidence, *, technical=False, no_confidence=False):
    row = {
        "model": model,
        "question_id": f"FEQ-ORG-{index:03d}",
        "category": "FEQ",
        "subcategory": f"FEQ-S{index % 4}",
        "k": 0,
        "answer_response": {"ok": not technical, "text": None if technical else "a"},
        "confidence_response": {"ok": True, "text": "50"},
        "parsed_confidence": None if (technical or no_confidence) else confidence,
        "confidence_parse_reason": "multiple_numeric_tokens" if no_confidence else None,
        "confidence_terminal": bool(no_confidence),
        "classification": None if technical else {
            "label": label,
            "is_correct": label == "CONFIDENT_CORRECT",
        },
        "valid_for_ehq12": not technical,
        "valid_for_ehq3": not (technical or no_confidence),
        "exclusion_reason": "response_format" if technical else None,
    }
    if technical:
        row["answer_response"]["error_type"] = "response_format"
    return row


def _panel():
    """Two models over the same 20 items; the second has incomplete coverage."""

    labels = ["CONFIDENT_CORRECT", "CONFIDENT_WRONG", "ABSTAIN", "HEDGE"]
    rows = []
    for index in range(20):
        label = labels[index % 4]
        confidence = 0.9 if label.startswith("CONFIDENT") else 0.1
        rows.append(_record("Complete", index, label, confidence))
        rows.append(
            _record(
                "Partial",
                index,
                label,
                confidence,
                technical=(index == 5),
                no_confidence=(index == 9),
            )
        )
    return rows


class IncompleteCoverageTests(unittest.TestCase):
    def test_coverage_tables_document_every_unscored_record(self):
        exclusions, coverage = MODULE._coverage_tables(
            _panel(), ["Complete", "Partial"]
        )
        self.assertEqual(len(exclusions), 2)
        by_reason = {row["reason"]: row for row in exclusions}
        self.assertEqual(
            by_reason["technical_failure"]["excluded_from"], "EHQ1, EHQ2, EHQ3"
        )
        self.assertEqual(by_reason["technical_failure"]["detail"], "response_format")
        self.assertEqual(
            by_reason["terminal_missing_confidence"]["excluded_from"], "EHQ3"
        )
        self.assertEqual(
            by_reason["terminal_missing_confidence"]["detail"],
            "multiple_numeric_tokens",
        )
        self.assertTrue(all(row["model"] == "Partial" for row in exclusions))

        complete, partial = coverage
        self.assertEqual(complete["n_items"], 20)
        self.assertEqual(complete["n_excluded_ehq12"], 0)
        self.assertEqual(complete["n_excluded_ehq3_only"], 0)
        self.assertEqual(partial["n_scored_ehq12"], 19)
        self.assertEqual(partial["n_excluded_ehq12"], 1)
        self.assertEqual(partial["n_excluded_ehq3_only"], 1)

    def test_bootstrap_denominators_match_the_point_estimate(self):
        rows = _panel()
        boot, _ = MODULE._bootstrap_scores(
            rows,
            ["Complete", "Partial"],
            weights=WEIGHTS,
            n_bins=10,
            n_resamples=4000,
            seed=42,
        )
        for model in ("Complete", "Partial"):
            point = compute_ehq_scores(
                [row for row in rows if row["model"] == model],
                weights=WEIGHTS,
                n_bins=10,
            )
            for field in ("EHQ1", "EHQ2"):
                self.assertAlmostEqual(
                    float(boot[model][field].mean()),
                    float(point[field]),
                    delta=0.03,
                    msg=f"{model} {field} bootstrap mean drifted from point estimate",
                )
            self.assertTrue((boot[model]["EHQ3"] <= 1.0).all())

    def test_incomplete_coverage_shifts_only_the_affected_model(self):
        rows = _panel()
        boot, _ = MODULE._bootstrap_scores(
            rows,
            ["Complete", "Partial"],
            weights=WEIGHTS,
            n_bins=10,
            n_resamples=500,
            seed=42,
        )
        # The complete model is unaffected by its neighbour's missing records.
        point = compute_ehq_scores(
            [row for row in rows if row["model"] == "Complete"],
            weights=WEIGHTS,
            n_bins=10,
        )
        self.assertAlmostEqual(
            float(boot["Complete"]["EHQ1"].mean()), float(point["EHQ1"]), delta=0.03
        )
