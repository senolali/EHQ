from unittest.mock import patch
import unittest

from ehq.reporting import build_summary, check_reporting_dependencies


class ReportingTests(unittest.TestCase):
    def test_missing_optional_dependency_is_detected_before_run(self):
        with patch("ehq.reporting.importlib.util.find_spec", return_value=None):
            with self.assertRaises(RuntimeError):
                check_reporting_dependencies(excel=True, figures=False)

    def test_no_optional_output_requires_no_extra_dependency(self):
        with patch("ehq.reporting.importlib.util.find_spec", return_value=None):
            check_reporting_dependencies(excel=False, figures=False)

    def test_equal_scores_receive_average_tied_ranks(self):
        results = {
            name: {
                "scores": {
                    "EHQ": 0.7,
                    "EHQ1": 0.7,
                    "EHQ2": 0.7,
                    "EHQ3": 0.7,
                }
            }
            for name in ("A", "B")
        }
        rows = build_summary(results)["models"]
        self.assertEqual([row["rank"] for row in rows], [1.5, 1.5])
