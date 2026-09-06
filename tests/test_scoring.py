import unittest

from ehq.evaluation.scoring import (
    compute_ehq_scores,
    expected_calibration_error,
)


def _record(label, correct, confidence, valid3=True):
    return {
        "k": 0,
        "valid_for_ehq12": True,
        "valid_for_ehq3": valid3,
        "parsed_confidence": confidence,
        "classification": {"label": label, "is_correct": correct},
    }


class ScoringTests(unittest.TestCase):
    def test_ehq3_calibrates_over_substantive_answers_only(self):
        records = [
            _record("ABSTAIN", False, 0.0),
            _record("HEDGE", False, 0.2),
            _record("CONFIDENT_CORRECT", True, 0.9),
            _record("CONFIDENT_WRONG", False, 0.8),
        ]
        result = compute_ehq_scores(
            records,
            weights={"ehq1": 0.30, "ehq2": 0.45, "ehq3": 0.25},
            n_bins=10,
        )
        self.assertEqual(result["EHQ1"], 0.5)
        self.assertEqual(result["EHQ2"], 0.75)
        # Confidence coverage still counts every record that produced a
        # parsable confidence value.
        self.assertEqual(result["n_valid_ehq3"], 4)
        # Calibration is restricted to CONFIDENT_CORRECT / CONFIDENT_WRONG.
        self.assertEqual(result["n_ehq3_calibration"], 2)
        self.assertEqual(result["ehq3_protocol"], "confidence_substantive_only_v1")
        self.assertAlmostEqual(result["EHQ3"], 0.55)
        self.assertAlmostEqual(result["EHQ"], 0.625)

    def test_abstain_and_hedge_confidences_do_not_move_ehq3(self):
        substantive = [
            _record("CONFIDENT_CORRECT", True, 0.9),
            _record("CONFIDENT_WRONG", False, 0.8),
        ]
        weights = {"ehq1": 0.30, "ehq2": 0.45, "ehq3": 0.25}
        baseline = compute_ehq_scores(substantive, weights=weights, n_bins=10)
        with_restraint = compute_ehq_scores(
            [
                *substantive,
                _record("ABSTAIN", False, 0.95),
                _record("HEDGE", False, 0.05),
            ],
            weights=weights,
            n_bins=10,
        )
        self.assertAlmostEqual(baseline["EHQ3"], with_restraint["EHQ3"])

    def test_ehq3_is_undefined_without_any_substantive_answer(self):
        result = compute_ehq_scores(
            [_record("ABSTAIN", False, 0.0), _record("HEDGE", False, 0.1)],
            weights={"ehq1": 0.30, "ehq2": 0.45, "ehq3": 0.25},
            n_bins=10,
        )
        self.assertEqual(result["EHQ1"], 1.0)
        self.assertEqual(result["EHQ2"], 1.0)
        self.assertEqual(result["n_valid_ehq3"], 2)
        self.assertEqual(result["n_ehq3_calibration"], 0)
        self.assertIsNone(result["EHQ3"])
        self.assertIsNone(result["EHQ"])

    def test_missing_confidence_is_reported_not_imputed(self):
        records = [
            {
                **_record("ABSTAIN", False, None, valid3=False),
                "confidence_terminal": True,
            },
            _record("CONFIDENT_WRONG", False, 0.8),
        ]
        result = compute_ehq_scores(
            records,
            weights={"ehq1": 0.30, "ehq2": 0.45, "ehq3": 0.25},
        )
        self.assertEqual(result["n_missing_confidence"], 1)
        self.assertEqual(result["confidence_coverage"], 0.5)
        self.assertEqual(result["n_terminal_missing_confidence"], 1)
        self.assertEqual(result["n_retryable_records"], 0)

    def test_technical_failure_is_excluded_from_denominators(self):
        records = [
            _record("ABSTAIN", False, 0.0),
            {
                "k": 0,
                "valid_for_ehq12": False,
                "valid_for_ehq3": False,
                "classification": None,
            },
        ]
        result = compute_ehq_scores(
            records,
            weights={"ehq1": 0.30, "ehq2": 0.45, "ehq3": 0.25},
        )
        self.assertEqual(result["n_valid_ehq12"], 1)
        self.assertEqual(result["technical_failures"], 1)
        self.assertEqual(result["n_retryable_records"], 1)
        self.assertEqual(result["EHQ1"], 1.0)

    def test_ece_rejects_empty_inputs(self):
        with self.assertRaises(ValueError):
            expected_calibration_error([], [])
