import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMPLIANCE = _load("abstention_confidence_compliance")

COMPLETE = "I don't have reliable information about that."
CUT = "the information requested is not available. The document mentions the acquisition of"


def _record(model, label, confidence, *, text=COMPLETE, valid3=True):
    return {
        "model": model,
        "question_id": "Q",
        "valid_for_ehq12": True,
        "valid_for_ehq3": valid3,
        "parsed_confidence": confidence,
        "answer_response": {
            "ok": True,
            "text": text,
            "usage": {"completion_tokens": 40},
        },
        "classification": {"label": label, "is_correct": False},
    }


class ComplianceTests(unittest.TestCase):
    def _profile(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "records.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
            )
            options = {
                "answer_max_tokens": 512,
                "excluded": [],
                "serving": {"A": "aws", "B": "aws"},
            }
            options.update(kwargs)
            return COMPLIANCE.profile(run, **options)

    def test_compliance_counts_only_exact_zero(self):
        rows = [_record("A", "ABSTAIN", 0.0) for _ in range(6)]
        rows += [_record("A", "ABSTAIN", 0.9) for _ in range(4)]
        stats = self._profile(rows)["models"]["A"]
        self.assertEqual(stats["n_abstain"], 10)
        self.assertAlmostEqual(stats["compliance"], 0.6)
        self.assertAlmostEqual(stats["high_confidence_rate"], 0.4)
        self.assertAlmostEqual(stats["max_confidence_rate"], 0.0)

    def test_maximum_confidence_is_reported_separately(self):
        rows = [_record("A", "ABSTAIN", 1.0) for _ in range(5)]
        rows += [_record("A", "ABSTAIN", 0.0) for _ in range(5)]
        stats = self._profile(rows)["models"]["A"]
        self.assertAlmostEqual(stats["max_confidence_rate"], 0.5)
        self.assertAlmostEqual(stats["high_confidence_rate"], 0.5)
        self.assertAlmostEqual(stats["mean_confidence"], 0.5)

    def test_truncated_abstentions_are_dropped_and_counted(self):
        rows = [_record("A", "ABSTAIN", 0.0) for _ in range(4)]
        rows += [_record("A", "ABSTAIN", 0.9, text=CUT) for _ in range(6)]
        stats = self._profile(rows)["models"]["A"]
        # Only the intact records are profiled; the cut ones are reported.
        self.assertEqual(stats["n_abstain"], 4)
        self.assertEqual(stats["dropped_truncated"], 6)
        self.assertAlmostEqual(stats["compliance"], 1.0)

    def test_excluded_models_do_not_appear_at_all(self):
        rows = [_record("A", "ABSTAIN", 0.0), _record("B", "ABSTAIN", 1.0)]
        report = self._profile(rows, excluded=["B"])
        self.assertIn("A", report["models"])
        self.assertNotIn("B", report["models"])
        self.assertEqual(report["excluded_models"], ["B"])

    def test_hedges_are_profiled_separately_from_abstentions(self):
        rows = [_record("A", "ABSTAIN", 0.0) for _ in range(3)]
        rows += [_record("A", "HEDGE", 0.6) for _ in range(2)]
        stats = self._profile(rows)["models"]["A"]
        self.assertEqual(stats["n_abstain"], 3)
        self.assertEqual(stats["n_hedge"], 2)
        self.assertAlmostEqual(stats["hedge_mean_confidence"], 0.6)
        # A hedge may carry a substantive answer, so it must not dilute the
        # instruction-compliance figure.
        self.assertAlmostEqual(stats["compliance"], 1.0)

    def test_a_switch_and_a_scale_are_told_apart(self):
        # Both models are equally non-compliant; only one is using the scale.
        rows = [_record("A", "ABSTAIN", 1.0) for _ in range(8)]
        rows += [_record("A", "ABSTAIN", 0.0) for _ in range(2)]
        rows += [_record("B", "ABSTAIN", 0.95) for _ in range(5)]
        rows += [_record("B", "ABSTAIN", 0.97) for _ in range(3)]
        rows += [_record("B", "ABSTAIN", 0.0) for _ in range(2)]
        report = self._profile(rows)
        switch, scale = report["models"]["A"], report["models"]["B"]

        self.assertAlmostEqual(switch["compliance"], 0.2)
        self.assertAlmostEqual(scale["compliance"], 0.2)
        self.assertAlmostEqual(switch["max_confidence_rate"], 0.8)
        self.assertAlmostEqual(switch["graded_high_rate"], 0.0)
        self.assertEqual(switch["scale_use"], "binary")
        self.assertAlmostEqual(scale["max_confidence_rate"], 0.0)
        self.assertAlmostEqual(scale["graded_high_rate"], 0.8)
        self.assertEqual(scale["scale_use"], "graded")

    def test_graded_excludes_exactly_one(self):
        rows = [_record("A", "ABSTAIN", 1.0) for _ in range(10)]
        stats = self._profile(rows)["models"]["A"]
        self.assertAlmostEqual(stats["high_confidence_rate"], 1.0)
        self.assertAlmostEqual(stats["graded_high_rate"], 0.0)

    def test_substantive_answers_are_ignored(self):
        rows = [_record("A", "CONFIDENT_WRONG", 0.9) for _ in range(5)]
        rows += [_record("A", "ABSTAIN", 0.0)]
        stats = self._profile(rows)["models"]["A"]
        self.assertEqual(stats["n_abstain"], 1)

    def test_records_without_a_parsed_confidence_are_skipped(self):
        rows = [
            _record("A", "ABSTAIN", 0.0),
            _record("A", "ABSTAIN", None, valid3=False),
        ]
        stats = self._profile(rows)["models"]["A"]
        self.assertEqual(stats["n_abstain"], 1)

    def test_developer_summary_spans_the_models_it_contains(self):
        rows = [_record("Claude-5-Sonnet", "ABSTAIN", 0.0) for _ in range(10)]
        rows += [_record("GPT-4o", "ABSTAIN", 1.0) for _ in range(10)]
        report = self._profile(rows)
        self.assertAlmostEqual(report["developers"]["Anthropic"]["mean"], 1.0)
        self.assertAlmostEqual(report["developers"]["OpenAI"]["mean"], 0.0)

    def test_the_developer_is_not_taken_from_the_serving_route(self):
        # Claude, LLaMA and Nova all arrive through `aws`; grouping on the
        # serving route would report them as one family.
        rows = [_record("Claude-5-Sonnet", "ABSTAIN", 0.0)]
        rows += [_record("LLaMA-3-70B", "ABSTAIN", 0.0)]
        rows += [_record("Nova-Pro", "ABSTAIN", 0.0)]
        report = self._profile(
            rows,
            serving={
                "Claude-5-Sonnet": "aws",
                "LLaMA-3-70B": "aws",
                "Nova-Pro": "aws",
            },
        )
        self.assertEqual(
            sorted(report["developers"]), ["Amazon", "Anthropic", "Meta"]
        )
        for stats in report["models"].values():
            self.assertEqual(stats["serving"], "aws")

    def test_an_unrecognised_name_is_not_forced_into_a_family(self):
        report = self._profile([_record("Zephyr-9", "ABSTAIN", 0.0)])
        self.assertEqual(report["models"]["Zephyr-9"]["developer"], "unknown")


class RenderingTests(unittest.TestCase):
    def _report(self):
        rows = [_record("A", "ABSTAIN", 0.0) for _ in range(7)]
        rows += [_record("A", "ABSTAIN", 1.0) for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "records.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
            )
            return COMPLIANCE.profile(
                run,
                answer_max_tokens=512,
                excluded=["Gemini-2.5-Pro"],
                serving={"A": "aws"},
            )

    def test_csv_header_matches_its_rows(self):
        lines = [l for l in COMPLIANCE._csv(self._report()).strip().splitlines() if l]
        self.assertEqual(len(lines), 2)
        self.assertEqual(len(lines[0].split(",")), len(lines[1].split(",")))

    def test_latex_names_the_excluded_model(self):
        text = COMPLIANCE._latex(self._report())
        self.assertIn("Gemini-2.5-Pro", text)
        self.assertIn("Graded", text)
        self.assertIn("\\label{tab:abstention-confidence-compliance}", text)
        self.assertEqual(text.count("\\toprule"), 1)


if __name__ == "__main__":
    unittest.main()
