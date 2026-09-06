import tempfile
import unittest
from pathlib import Path

from ehq.analysis import load_capability_counts

ROOT_TOOLS = Path(__file__).resolve().parents[1] / "tools"
from ehq.analysis.statistics import homogeneity_test


CEILING = {
    "Claude-3-Haiku": (259, 259),
    "Claude-4.5-Haiku": (258, 259),
    "DeepSeek-V4": (258, 259),
    "Claude-5-Sonnet": (258, 259),
    "Claude-4-Sonnet": (257, 259),
    "Gemini-2.5-Flash": (257, 259),
    "Gemma-4-31B": (257, 259),
    "LLaMA-4-Maverick": (257, 259),
    "Nova-Pro": (257, 259),
    "GPT-4o-mini": (256, 259),
    "LLaMA-3-70B": (255, 259),
    "GPT-4o": (254, 259),
    "Nova-Micro": (254, 259),
}


class HomogeneityTests(unittest.TestCase):
    def test_a_panel_at_ceiling_is_not_distinguished_from_one_rate(self):
        report = homogeneity_test(CEILING, resamples=2000, seed=7)
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["homogeneous"])
        self.assertGreater(report["p_value"], 0.05)
        # The spread is what resampling one rate would produce anyway.
        self.assertLess(abs(report["sd_ratio"] - 1.0), 0.25)

    def test_one_genuinely_worse_model_is_detected(self):
        counts = dict(CEILING)
        counts["Gemini-2.5-Pro"] = (196, 259)
        report = homogeneity_test(counts, resamples=2000, seed=7)
        self.assertFalse(report["homogeneous"])
        self.assertLess(report["p_value"], 0.01)

    def test_a_measure_that_separates_models_is_not_suppressed(self):
        counts = {
            "A": (240, 259),
            "B": (200, 259),
            "C": (160, 259),
            "D": (120, 259),
        }
        report = homogeneity_test(counts, resamples=2000, seed=7)
        self.assertFalse(report["homogeneous"])

    def test_fewer_than_two_models_is_reported_not_guessed(self):
        report = homogeneity_test({"A": (250, 259)}, resamples=100, seed=7)
        self.assertEqual(report["status"], "insufficient_models")

    def test_the_result_is_reproducible_from_the_seed(self):
        first = homogeneity_test(CEILING, resamples=1000, seed=11)
        second = homogeneity_test(CEILING, resamples=1000, seed=11)
        self.assertEqual(first["p_value"], second["p_value"])


class CapabilityCountsLoadingTests(unittest.TestCase):
    def _csv(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capability_scores.csv"
            path.write_text(text, encoding="utf-8")
            return load_capability_counts(path)

    def test_counts_are_read_when_present(self):
        counts = self._csv(
            "model,capability_score,n_correct,n_scored\n"
            "GPT-4o,0.980695,254,259\n"
            "Nova-Pro,0.992278,257,259\n"
        )
        self.assertEqual(counts, {"GPT-4o": (254, 259), "Nova-Pro": (257, 259)})

    def test_a_file_without_the_columns_yields_nothing(self):
        # Score files written before the columns existed must still load; the
        # analysis then simply cannot run the homogeneity test.
        self.assertEqual(
            self._csv("model,capability_score\nGPT-4o,0.980695\n"), {}
        )

    def test_unparseable_counts_are_skipped_not_fatal(self):
        counts = self._csv(
            "model,capability_score,n_correct,n_scored\n"
            "GPT-4o,0.98,not-a-number,259\n"
            "Nova-Pro,0.99,257,259\n"
        )
        self.assertEqual(counts, {"Nova-Pro": (257, 259)})


if __name__ == "__main__":
    unittest.main()


class BackfillTests(unittest.TestCase):
    def _run(self, models):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "backfill_capability_counts",
            ROOT_TOOLS / "backfill_capability_counts.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tmp = tempfile.mkdtemp()
        run = Path(tmp)
        (run / "capability_summary.json").write_text(
            __import__("json").dumps({"models": models}), encoding="utf-8"
        )
        module.backfill(run)
        return run / "capability_scores.csv"

    def test_unequal_denominators_survive_the_round_trip(self):
        # The reason the counts cannot be recovered from the rate alone.
        path = self._run(
            [
                {"model": "A", "capability_score": 0.996139, "n_correct": 258,
                 "n_scored": 259},
                {"model": "B", "capability_score": 0.996124, "n_correct": 257,
                 "n_scored": 258},
            ]
        )
        counts = load_capability_counts(path)
        self.assertEqual(counts, {"A": (258, 259), "B": (257, 258)})

    def test_impossible_counts_are_refused(self):
        with self.assertRaises(SystemExit):
            self._run(
                [{"model": "A", "capability_score": 1.5, "n_correct": 300,
                  "n_scored": 259}]
            )
