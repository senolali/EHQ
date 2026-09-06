import unittest

from ehq.analysis import build_analysis_report
from ehq.cli import _parse_model_exclusions
from ehq.types import ModelSpec


def _result(value):
    return {
        "scores": {
            "EHQ": value,
            "EHQ1": value,
            "EHQ2": value,
            "EHQ3": value,
            "n_valid_ehq12": 3000,
            "response_distribution": {},
        },
        "records": [],
        "category_scores": {},
        "subcategory_scores": {},
    }


PANEL = {
    "Claude-5-Sonnet": 0.8131,
    "GPT-4o": 0.4831,
    "Nova-Micro": 0.4147,
    "Gemini-2.5-Pro": 0.2935,
}
MODELS = [ModelSpec(name=n, provider="asu", provider_model=n) for n in PANEL]
REASON = "33% of answers truncated; rank undetermined across nine positions"


class ExclusionParsingTests(unittest.TestCase):
    def test_a_reason_is_required(self):
        for bad in ("Gemini-2.5-Pro", "Gemini-2.5-Pro=", "=reason", ""):
            with self.assertRaises(SystemExit):
                _parse_model_exclusions([bad])

    def test_name_and_reason_are_kept_verbatim_after_trimming(self):
        parsed = _parse_model_exclusions([" A = cut off mid-clause "])
        self.assertEqual(parsed, {"A": "cut off mid-clause"})

    def test_a_reason_may_contain_equals_signs(self):
        parsed = _parse_model_exclusions(["A=rate=33%"])
        self.assertEqual(parsed, {"A": "rate=33%"})


class ExclusionEffectTests(unittest.TestCase):
    def _report(self, excluded=None):
        return build_analysis_report(
            {name: _result(value) for name, value in PANEL.items()},
            MODELS,
            seed=42,
            excluded_models=excluded,
        )

    def test_an_excluded_model_leaves_the_confirmatory_panel(self):
        report = self._report({"Gemini-2.5-Pro": REASON})
        self.assertNotIn("Gemini-2.5-Pro", report["confirmatory_panel"])
        self.assertEqual(len(report["confirmatory_panel"]), 3)

    def test_its_scores_and_reason_are_still_reported(self):
        report = self._report({"Gemini-2.5-Pro": REASON})
        excluded = report["excluded_models"]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["model"], "Gemini-2.5-Pro")
        self.assertEqual(excluded[0]["reason"], REASON)
        # Withheld from the analyses, not deleted from the record.
        self.assertAlmostEqual(excluded[0]["EHQ"], 0.2935)

    def test_component_correlations_are_computed_without_it(self):
        with_all = self._report()
        without = self._report({"Gemini-2.5-Pro": REASON})
        pair = "EHQ1_vs_EHQ2"
        self.assertEqual(with_all["component_correlations"][pair]["n"], 4)
        self.assertEqual(without["component_correlations"][pair]["n"], 3)

    def test_excluding_an_absent_model_is_an_error_not_a_no_op(self):
        with self.assertRaises(ValueError):
            self._report({"Not-In-This-Run": REASON})

    def test_excluding_everything_is_refused(self):
        with self.assertRaises(ValueError):
            self._report({name: REASON for name in PANEL})

    def test_no_exclusions_leaves_the_panel_whole(self):
        report = self._report()
        self.assertEqual(len(report["confirmatory_panel"]), 4)
        self.assertEqual(report["excluded_models"], [])


if __name__ == "__main__":
    unittest.main()


class RecomputeExclusionTests(unittest.TestCase):
    """The in-place regeneration path must accept the same exclusion."""

    def test_the_tool_passes_exclusions_to_the_analysis(self):
        import importlib.util
        import inspect
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "recompute_ehq3_protocol",
            root / "tools" / "recompute_ehq3_protocol.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        signature = inspect.signature(module.recompute)
        self.assertIn("excluded_models", signature.parameters)
        source = inspect.getsource(module.recompute)
        self.assertIn("excluded_models=excluded_models", source)
