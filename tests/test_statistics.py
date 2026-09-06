import unittest

from ehq.analysis.statistics import (
    bootstrap_correlation_ci,
    correlation_permutation_test,
    holm_adjust,
    paired_generation_analysis,
    pearson_correlation,
    spearman_correlation,
)


class StatisticsTests(unittest.TestCase):
    def test_correlations(self):
        x = [1, 2, 3, 4]
        self.assertAlmostEqual(pearson_correlation(x, [2, 4, 6, 8]), 1.0)
        self.assertAlmostEqual(spearman_correlation(x, [10, 30, 20, 40]), 0.8)

    def test_bootstrap_is_deterministic(self):
        x = [1, 2, 3, 4, 5]
        y = [1, 3, 2, 5, 4]
        first = bootstrap_correlation_ci(x, y, n_resamples=100, seed=7)
        second = bootstrap_correlation_ci(x, y, n_resamples=100, seed=7)
        self.assertEqual(first, second)

    def test_exact_correlation_permutation_test(self):
        result = correlation_permutation_test(
            [1, 2, 3, 4], [2, 4, 6, 8], method="pearson"
        )
        self.assertEqual(result["method"], "exact")
        self.assertEqual(result["n_permutations"], 24)
        self.assertAlmostEqual(result["p_value"], 2 / 24)

    def test_holm_adjustment_is_monotone_in_sorted_p_values(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.06)

    def test_paired_generation_summary(self):
        result = paired_generation_analysis(
            [
                {"old": 0.5, "new": 0.6},
                {"old": 0.7, "new": 0.65},
                {"old": 0.4, "new": 0.5},
            ]
        )
        self.assertEqual(result["n_pairs"], 3)
        self.assertEqual(result["improved_pairs"], 2)
        self.assertEqual(result["declined_pairs"], 1)

    def test_single_pair_does_not_report_zero_effect_or_p_value(self):
        result = paired_generation_analysis([{"old": 0.432, "new": 0.4338}])
        self.assertEqual(result["inferential_status"], "insufficient_pairs")
        self.assertIsNone(result["cohens_dz"])
        self.assertEqual(result["cohens_dz_status"], "insufficient_pairs")
        self.assertIsNone(result["exact_sign_permutation_p"])
        self.assertEqual(
            result["exact_sign_permutation_status"], "insufficient_pairs"
        )
