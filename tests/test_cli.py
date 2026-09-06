import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ehq import cli
from ehq.cli import (
    _make_model_progress_reporter,
    _make_request_event_reporter,
    _missing_runtime_dependencies,
    build_parser,
)


class CLITests(unittest.TestCase):
    def test_real_http_provider_requires_requests(self):
        models = [SimpleNamespace(provider="openai")]
        with patch("ehq.cli.importlib.util.find_spec", return_value=None):
            self.assertEqual(_missing_runtime_dependencies(models), ["requests"])

    def test_offline_only_provider_declares_no_http_dependency(self):
        models = [SimpleNamespace(provider="offline")]
        with patch("ehq.cli.importlib.util.find_spec", return_value=None):
            self.assertEqual(_missing_runtime_dependencies(models), [])

    def test_progress_reporter_emits_first_interval_and_final_updates(self):
        reporter = _make_model_progress_reporter("Example Model")
        base = {
            "phase": "evaluating",
            "total_items": 100,
            "resumed_records": 0,
            "valid_ehq12": 1,
            "valid_ehq3": 1,
            "terminal_missing_confidence": 0,
            "retryable_records": 0,
            "technical_failures": 0,
            "elapsed_seconds": 1.0,
            "items_per_minute": 60.0,
            "eta_seconds": 99.0,
        }
        with patch("ehq.cli._emit_event") as emit:
            for completed in (1, 2, 10, 100):
                reporter(
                    {
                        **base,
                        "completed_items": completed,
                        "completed_this_session": completed,
                    }
                )
        self.assertEqual(emit.call_count, 3)
        self.assertEqual(emit.call_args_list[0].args[0], "model_progress")
        self.assertEqual(emit.call_args_list[-1].kwargs["percent"], 100.0)

    def test_request_reporter_exposes_retry_without_prompt_content(self):
        reporter = _make_request_event_reporter("Example Model")
        with patch("ehq.cli._emit_event") as emit:
            reporter(
                {
                    "phase": "retry_scheduled",
                    "purpose": "answer",
                    "attempt": 2,
                    "next_attempt": 3,
                    "max_retries": 5,
                    "wait_seconds": 2,
                    "error_type": "timeout",
                    "prompt": "must not be logged",
                }
            )
        self.assertEqual(emit.call_args.args[0], "provider_retry_scheduled")
        self.assertEqual(emit.call_args.kwargs["model"], "Example Model")
        self.assertNotIn("prompt", emit.call_args.kwargs)

    def test_simple_command_surface_has_safe_defaults(self):
        parser = build_parser()
        pilot = parser.parse_args(["pilot", "--model", "Nova-Pro"])
        self.assertEqual(pilot.limit, 20)
        self.assertFalse(pilot.excel)
        full = parser.parse_args(["full", "--all-verified"])
        self.assertIsNone(full.limit)
        self.assertTrue(full.excel)
        self.assertTrue(full.figures)


if __name__ == "__main__":
    unittest.main()


class ConsoleOutputTests(unittest.TestCase):
    def test_progress_event_renders_a_readable_line(self):
        line = cli._format_event(
            "model_progress",
            {
                "model": "Claude-5-Sonnet",
                "phase": "evaluating",
                "completed": 2100,
                "total": 3000,
                "percent": 70.0,
                "valid_ehq12": 2100,
                "valid_ehq3": 2098,
                "technical_failures": 2,
                "items_per_minute": 8787.0,
                "eta_seconds": 65,
            },
        )
        self.assertIn("[Claude-5-Sonnet] 2,100/3,000 (70.0%)", line)
        self.assertIn("valid: 2,100", line)
        self.assertIn("fail: 2", line)
        self.assertIn("ETA 1m 05s", line)

    def test_retry_and_failure_events_name_the_provider_error(self):
        retry = cli._format_event(
            "provider_retry_scheduled",
            {
                "model": "Claude-5-Sonnet",
                "purpose": "answer",
                "next_attempt": 4,
                "max_retries": 5,
                "wait_seconds": 8,
                "error_type": "response_format",
            },
        )
        self.assertIn("[RETRY] Claude-5-Sonnet answer (response_format)", retry)
        self.assertIn("attempt 4/5", retry)
        failure = cli._format_event(
            "provider_request_exhausted",
            {
                "model": "Claude-5-Sonnet",
                "purpose": "answer",
                "error_type": "response_format",
                "error_message": "ASU response text is missing",
            },
        )
        self.assertIn("[FAILED]", failure)
        self.assertIn("ASU response text is missing", failure)

    def test_undefined_ehq_is_reported_not_formatted_as_a_number(self):
        line = cli._format_event(
            "model_completed",
            {"model": "M", "ehq": None, "retryable": 0},
        )
        self.assertIn("EHQ: undefined", line)

    def test_run_summary_table_reports_scores_and_coverage(self):
        rendered = cli._render_run_summary(
            {
                "run_dir": "/runs/demo",
                "fingerprint": "abc123",
                "dataset_policy": "NON_PUBLISHABLE_CANDIDATE",
                "ehq3_protocol": "confidence_substantive_only_v1",
                "artifact_count": 9,
                "artifacts_verified": True,
                "analysis_status": {"rq1": "ok", "rq2": "ok"},
                "summary": {
                    "models": [
                        {
                            "rank": 1,
                            "model": "Claude-5-Sonnet",
                            "EHQ1": 0.7364,
                            "EHQ2": 0.8395,
                            "EHQ3": 0.8574,
                            "EHQ": 0.8130,
                            "n_valid_ehq12": 2997,
                            "n_ehq3_calibration": 790,
                            "n_missing_confidence": 0,
                            "technical_failures": 3,
                        }
                    ]
                },
            }
        )
        self.assertIn("Evaluation complete", rendered)
        self.assertIn("confidence_substantive_only_v1", rendered)
        self.assertIn("verified: yes", rendered)
        self.assertIn("0.8574", rendered)
        self.assertIn("2,997", rendered)
        self.assertIn("Incomplete coverage", rendered)
        self.assertIn("3 technical failure(s)", rendered)


class AutomaticReportTests(unittest.TestCase):
    def test_full_builds_the_publication_package_by_default(self):
        parser = build_parser()
        full = parser.parse_args(["full", "--all-verified"])
        self.assertTrue(full.report)
        pilot = parser.parse_args(["pilot", "--all-verified"])
        self.assertFalse(pilot.report)
        # The low-level runners stay opt-in.
        run = parser.parse_args(["run"])
        self.assertFalse(run.report)

    def test_report_can_be_disabled_and_resamples_configured(self):
        parser = build_parser()
        args = parser.parse_args(
            ["full", "--all-verified", "--no-report", "--report-resamples", "500"]
        )
        self.assertFalse(args.report)
        self.assertEqual(args.report_resamples, 500)

    def test_skipped_report_is_reported_without_failing_the_run(self):
        rendered = cli._render_run_summary(
            {
                "run_dir": "/runs/demo",
                "fingerprint": "abc",
                "dataset_policy": "NON_PUBLISHABLE_CANDIDATE",
                "ehq3_protocol": "confidence_substantive_only_v1",
                "artifact_count": 3,
                "artifacts_verified": True,
                "analysis_status": {"rq1": "ok", "rq2": "ok"},
                "summary": {"models": []},
                "report": {"status": "skipped", "reason": "no substantive answers"},
            }
        )
        self.assertIn("Publication pkg  : SKIPPED - no substantive answers", rendered)

    def test_report_dependencies_are_checked_before_the_run(self):
        from ehq.reporting import check_reporting_dependencies

        with patch("ehq.reporting.importlib.util.find_spec", return_value=None):
            with self.assertRaises(RuntimeError):
                check_reporting_dependencies(excel=False, figures=False, report=True)
            check_reporting_dependencies(excel=False, figures=False, report=False)
