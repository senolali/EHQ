import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {"ehq1": 0.30, "ehq2": 0.45, "ehq3": 0.25}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SENSITIVITY = _load("truncation_sensitivity")
AUDIT = _load("audit_truncated_responses")


def _record(model, question_id, label, text, *, confidence=0.9, correct=False):
    return {
        "model": model,
        "question_id": question_id,
        "category": "FEQ",
        "subcategory": "FEQ-ORG",
        "k": 0,
        "valid_for_ehq12": True,
        "valid_for_ehq3": True,
        "confidence_terminal": False,
        "parsed_confidence": confidence,
        "answer_response": {"ok": True, "text": text, "usage": {"completion_tokens": 40}},
        "classification": {
            "label": label,
            "is_correct": correct,
            "abstention_detected": label == "ABSTAIN",
            "hedge_detected": label == "HEDGE",
            "substantive_answer_detected": label.startswith("CONFIDENT"),
            "reasons": [],
        },
    }


COMPLETE = "The institute was founded in Ankara in 1967."
CUT = "the information requested is not available. The document mentions the acquisition of"


class SharedDetectorTests(unittest.TestCase):
    """The audit and the sensitivity analysis must agree on what is truncated."""

    def test_cut_clause_is_suspected_and_complete_sentence_is_not(self):
        self.assertTrue(AUDIT.signals(CUT, 40, 512)["suspected"])
        self.assertFalse(AUDIT.signals(COMPLETE, 40, 512)["suspected"])

    def test_token_limit_alone_is_enough(self):
        self.assertTrue(AUDIT.signals(COMPLETE, 512, 512)["suspected"])

    def test_sensitivity_reads_the_records_through_the_same_rule(self):
        self.assertTrue(
            SENSITIVITY._is_suspected(
                _record("M", "Q1", "CONFIDENT_WRONG", CUT), 512
            )
        )
        self.assertFalse(
            SENSITIVITY._is_suspected(
                _record("M", "Q1", "CONFIDENT_WRONG", COMPLETE), 512
            )
        )


class ScenarioTests(unittest.TestCase):
    def _analyse(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "records.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            return SENSITIVITY.analyse(
                run_dir, answer_max_tokens=512, weights=WEIGHTS
            )

    def test_an_untruncated_model_scores_identically_under_every_scenario(self):
        rows = [
            _record("Clean", f"Q{i}", "CONFIDENT_WRONG", COMPLETE) for i in range(5)
        ] + [
            _record("Clean", f"Q{i}", "CONFIDENT_CORRECT", COMPLETE, correct=True)
            for i in range(5, 10)
        ]
        report = self._analyse(rows)
        stats = report["models"]["Clean"]
        self.assertEqual(stats["n_suspected"], 0)
        self.assertEqual(stats["ehq_span"], 0.0)
        self.assertEqual(stats["rank_interval"], [1, 1])

    def test_truncated_confident_wrong_lifts_ehq1_and_ehq2_under_restraint(self):
        rows = [
            _record("Cut", f"Q{i}", "CONFIDENT_WRONG", CUT) for i in range(4)
        ] + [
            _record("Cut", f"Q{i}", "CONFIDENT_CORRECT", COMPLETE, correct=True)
            for i in range(4, 10)
        ]
        stats = self._analyse(rows)["models"]["Cut"]
        measured = stats["scores"]["measured"]
        restraint = stats["scores"]["restraint"]

        self.assertEqual(stats["n_suspected"], 4)
        self.assertEqual(stats["n_suspected_confident_wrong"], 4)
        # Measured: four wrong answers out of ten, no restraint at all.
        self.assertAlmostEqual(measured["EHQ1"], 0.0)
        self.assertAlmostEqual(measured["EHQ2"], 0.6)
        # Restraint: the same four records read as hedges.
        self.assertAlmostEqual(restraint["EHQ1"], 0.4)
        self.assertAlmostEqual(restraint["EHQ2"], 1.0)
        self.assertEqual(measured["n_valid_ehq12"], restraint["n_valid_ehq12"])

    def test_relabelled_records_leave_the_calibration_set(self):
        rows = [
            _record("Cut", f"Q{i}", "CONFIDENT_WRONG", CUT) for i in range(4)
        ] + [
            _record("Cut", f"Q{i}", "CONFIDENT_CORRECT", COMPLETE, correct=True)
            for i in range(4, 10)
        ]
        stats = self._analyse(rows)["models"]["Cut"]
        self.assertEqual(stats["scores"]["measured"]["n_ehq3_calibration"], 10)
        self.assertEqual(stats["scores"]["restraint"]["n_ehq3_calibration"], 6)
        # EHQ3 is recomputed on the reduced set, not carried over unchanged.
        self.assertNotAlmostEqual(
            stats["scores"]["measured"]["EHQ3"],
            stats["scores"]["restraint"]["EHQ3"],
        )

    def test_dropping_removes_the_records_from_every_denominator(self):
        rows = [
            _record("Cut", f"Q{i}", "CONFIDENT_WRONG", CUT) for i in range(4)
        ] + [
            _record("Cut", f"Q{i}", "CONFIDENT_CORRECT", COMPLETE, correct=True)
            for i in range(4, 10)
        ]
        dropped = self._analyse(rows)["models"]["Cut"]["scores"]["dropped"]
        self.assertEqual(dropped["n_valid_ehq12"], 6)
        self.assertAlmostEqual(dropped["EHQ2"], 1.0)

    def test_a_cut_model_can_overtake_a_clean_one_and_both_intervals_widen(self):
        rows = []
        # A clean model that answers correctly and confidently throughout.
        rows += [
            _record("Clean", f"C{i}", "CONFIDENT_CORRECT", COMPLETE, correct=True)
            for i in range(10)
        ]
        # A cut model that looks worse than it is.
        rows += [_record("Cut", f"X{i}", "CONFIDENT_WRONG", CUT) for i in range(8)]
        rows += [
            _record("Cut", f"X{i}", "CONFIDENT_CORRECT", COMPLETE, correct=True)
            for i in range(8, 10)
        ]
        report = self._analyse(rows)
        clean, cut = report["models"]["Clean"], report["models"]["Cut"]

        self.assertEqual(cut["ranks"]["measured"], 2)
        self.assertEqual(cut["ranks"]["restraint"], 1)
        self.assertGreater(cut["ehq_span"], 0.0)
        # A rank is a statement about the panel, not about one model: the clean
        # model's own scores never move, yet its position is uncertain because a
        # peer's is. Reporting rank intervals for the panel is therefore the
        # honest presentation, not just for the models that were truncated.
        self.assertEqual(clean["ehq_span"], 0.0)
        self.assertEqual(clean["rank_interval"], [1, 2])

    def test_technical_failures_are_not_counted_as_truncation(self):
        rows = [
            _record("Broken", "Q0", "CONFIDENT_WRONG", COMPLETE),
            {
                **_record("Broken", "Q1", "CONFIDENT_WRONG", COMPLETE),
                "valid_for_ehq12": False,
                "answer_response": {"ok": False, "text": None},
            },
        ]
        stats = self._analyse(rows)["models"]["Broken"]
        self.assertEqual(stats["n_suspected"], 0)


class RenderingTests(unittest.TestCase):
    def _report(self):
        rows = [_record("Cut", f"X{i}", "CONFIDENT_WRONG", CUT) for i in range(4)] + [
            _record("Cut", f"X{i}", "CONFIDENT_CORRECT", COMPLETE, correct=True)
            for i in range(4, 10)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "records.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            return SENSITIVITY.analyse(
                run_dir, answer_max_tokens=512, weights=WEIGHTS
            )

    def test_csv_has_one_row_per_model_and_a_cell_for_every_scenario(self):
        text = SENSITIVITY._csv(self._report())
        lines = [line for line in text.strip().splitlines() if line]
        self.assertEqual(len(lines), 2)
        header, row = lines
        self.assertEqual(len(header.split(",")), len(row.split(",")))
        for scenario in SENSITIVITY.SCENARIOS:
            self.assertIn(f"EHQ_{scenario}", header)

    def test_latex_table_is_self_contained(self):
        text = SENSITIVITY._latex(self._report())
        self.assertIn("\\begin{table}", text)
        self.assertIn("\\label{tab:truncation-sensitivity}", text)
        self.assertIn("\\end{table}", text)
        self.assertEqual(text.count("\\toprule"), 1)


if __name__ == "__main__":
    unittest.main()
