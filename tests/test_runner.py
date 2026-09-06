from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ehq.checkpoint import Checkpoint
from ehq.clients.mock import MockClient
from ehq.config import load_experiment_config, load_models
from ehq.evaluation.runner import _request, knowability_for_model, run_model
from ehq.prompts import CONFIDENCE_SYSTEM_PROMPT
from ehq.types import InferenceResponse


ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def test_confidence_request_uses_strict_purpose_specific_system_prompt(self):
        config = load_experiment_config(ROOT / "config" / "experiment.json")
        model = load_models(config.models_path)[0]
        confidence = _request(config, model, "prompt", purpose="confidence")
        answer = _request(config, model, "prompt", purpose="answer")
        self.assertEqual(confidence.system_prompt, CONFIDENCE_SYSTEM_PROMPT)
        self.assertEqual(answer.system_prompt, config.inference.system_prompt)
        self.assertNotEqual(confidence.system_prompt, answer.system_prompt)

    def test_offline_runner_produces_reproducible_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = load_experiment_config(ROOT / "config" / "experiment.json")
            config = replace(
                config,
                checkpoint_dir=tmp_path / "checkpoints",
                output_dir=tmp_path / "outputs",
                cache_dir=tmp_path / "cache",
                max_workers=1,
            )
            model = load_models(config.models_path)[0]
            items = [
                {
                    "question_id": "FEQ-001",
                    "category": "FEQ",
                    "question": "Who founded the fabricated Asterion Institute?",
                    "correct_answer": "[DOES_NOT_EXIST]",
                },
                {
                    "question_id": "CCQ-001",
                    "category": "CCQ",
                    "document": "The final audited figure was [REDACTED].",
                    "question": "What was the final audited figure?",
                    "correct_answer": "[REDACTED]",
                    "redacted_value": "71",
                },
            ]
            checkpoint = Checkpoint(config.checkpoint_dir / "offline.jsonl")
            result = run_model(
                items,
                model,
                config,
                checkpoint=checkpoint,
                offline=True,
            )
            self.assertEqual(result["scores"]["EHQ1"], 1.0)
            self.assertEqual(result["scores"]["EHQ2"], 1.0)
            # The offline mock abstains on every item, so under
            # confidence_substantive_only_v1 there is nothing to calibrate and
            # EHQ3 — and therefore the composite — is reported as undefined
            # rather than as perfect calibration at zero confidence.
            self.assertEqual(result["scores"]["n_ehq3_calibration"], 0)
            self.assertIsNone(result["scores"]["EHQ3"])
            self.assertIsNone(result["scores"]["EHQ"])
            self.assertEqual(len(list(checkpoint.records())), 2)

    def test_runner_reports_incremental_progress_with_quality_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_experiment_config(ROOT / "config" / "experiment.json")
            config = replace(config, max_workers=1)
            model = load_models(config.models_path)[0]
            items = [
                {
                    "question_id": f"FEQ-PROGRESS-{index}",
                    "category": "FEQ",
                    "subcategory": "FEQ-TEST",
                    "question": f"Who founded fabricated institute {index}?",
                    "correct_answer": "[DOES_NOT_EXIST]",
                }
                for index in range(1, 4)
            ]
            snapshots = []
            result = run_model(
                items,
                model,
                config,
                checkpoint=Checkpoint(Path(tmp) / "progress.jsonl"),
                offline=True,
                progress_callback=lambda progress: snapshots.append(dict(progress)),
            )
            self.assertEqual(len(snapshots), 3)
            self.assertEqual(snapshots[-1]["completed_items"], 3)
            self.assertEqual(snapshots[-1]["total_items"], 3)
            self.assertEqual(snapshots[-1]["valid_ehq12"], 3)
            self.assertEqual(snapshots[-1]["valid_ehq3"], 3)
            self.assertEqual(snapshots[-1]["technical_failures"], 0)
            self.assertEqual(result["scores"]["n_input"], 3)

    def test_semantically_invalid_confidence_becomes_terminal_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = load_experiment_config(ROOT / "config" / "experiment.json")
            config = replace(config, max_workers=1)
            model = load_models(config.models_path)[0]
            item = {
                "question_id": "FEQ-RETRY",
                "category": "FEQ",
                "subcategory": "FEQ-TEST",
                "question": "Who founded the fabricated Asterion Institute?",
                "correct_answer": "[DOES_NOT_EXIST]",
            }
            client = MockClient(
                lambda request: (
                    "approximately fifty" if request.purpose == "confidence"
                    else "I do not know."
                )
            )
            checkpoint = Checkpoint(tmp_path / "retry.jsonl", fingerprint="test")
            with patch(
                "ehq.evaluation.runner.client_for_model", return_value=client
            ):
                result = run_model(
                    [item], model, config, checkpoint=checkpoint, offline=True
                )
            self.assertEqual(result["scores"]["n_valid_ehq12"], 1)
            self.assertIsNone(result["scores"]["EHQ3"])
            self.assertEqual(result["n_retryable_records"], 0)
            self.assertEqual(result["n_terminal_missing_confidence"], 1)
            rows = list(checkpoint.records())
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["confidence_terminal"])
            self.assertEqual(
                rows[0]["confidence_parse_reason"], "no_numeric_token"
            )

            never_called = MockClient(
                lambda request: (_ for _ in ()).throw(
                    AssertionError("terminal checkpoint should prevent a new call")
                )
            )
            with patch(
                "ehq.evaluation.runner.client_for_model",
                return_value=never_called,
            ):
                resumed = run_model(
                    [item], model, config, checkpoint=checkpoint, offline=True
                )
            self.assertEqual(resumed["n_terminal_missing_confidence"], 1)

    def test_prose_wrapped_confidence_is_parsed_with_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_experiment_config(ROOT / "config" / "experiment.json")
            config = replace(config, max_workers=1)
            model = load_models(config.models_path)[0]
            item = {
                "question_id": "FEQ-PROSE-CONFIDENCE",
                "category": "FEQ",
                "subcategory": "FEQ-TEST",
                "question": "Who founded the fabricated Asterion Institute?",
                "correct_answer": "[DOES_NOT_EXIST]",
            }
            client = MockClient(
                lambda request: (
                    "Confidence: 0" if request.purpose == "confidence"
                    else "I do not know."
                )
            )
            checkpoint = Checkpoint(Path(tmp) / "prose.jsonl")
            with patch(
                "ehq.evaluation.runner.client_for_model", return_value=client
            ):
                result = run_model(
                    [item], model, config, checkpoint=checkpoint, offline=True
                )
            record = result["records"][0]
            self.assertTrue(record["valid_for_ehq3"])
            self.assertEqual(record["parsed_confidence"], 0.0)
            self.assertEqual(
                record["confidence_parse_strategy"],
                "single_integer_in_text",
            )

    def test_technical_confidence_failure_remains_retryable(self):
        class ConfidenceFailureClient(MockClient):
            def _request_once(self, request):
                if request.purpose == "confidence":
                    return InferenceResponse(
                        ok=False,
                        text=None,
                        requested_model=request.model.provider_model,
                        resolved_model=request.model.provider_model,
                        provider="mock",
                        error_type="timeout",
                        error_message="test timeout",
                    )
                return super()._request_once(request)

        with tempfile.TemporaryDirectory() as tmp:
            config = load_experiment_config(ROOT / "config" / "experiment.json")
            config = replace(config, max_workers=1)
            model = load_models(config.models_path)[0]
            item = {
                "question_id": "FEQ-TECHNICAL-CONFIDENCE",
                "category": "FEQ",
                "subcategory": "FEQ-TEST",
                "question": "Who founded the fabricated Asterion Institute?",
                "correct_answer": "[DOES_NOT_EXIST]",
            }
            checkpoint = Checkpoint(Path(tmp) / "technical.jsonl")
            with patch(
                "ehq.evaluation.runner.client_for_model",
                return_value=ConfidenceFailureClient(),
            ):
                result = run_model(
                    [item], model, config, checkpoint=checkpoint, offline=True
                )
            self.assertEqual(result["n_retryable_records"], 1)
            self.assertEqual(result["n_terminal_missing_confidence"], 0)
            self.assertEqual(list(checkpoint.records()), [])

    def test_exhausted_confidence_response_format_is_terminal_missing(self):
        class EmptyConfidenceClient(MockClient):
            def _request_once(self, request):
                if request.purpose == "confidence":
                    return InferenceResponse(
                        ok=False,
                        text=None,
                        requested_model=request.model.provider_model,
                        resolved_model=request.model.provider_model,
                        provider="mock",
                        error_type="response_format",
                        error_message="response text is missing",
                    )
                return super()._request_once(request)

        with tempfile.TemporaryDirectory() as tmp:
            config = load_experiment_config(ROOT / "config" / "experiment.json")
            config = replace(config, max_workers=1)
            model = load_models(config.models_path)[0]
            item = {
                "question_id": "FEQ-EMPTY-CONFIDENCE",
                "category": "FEQ",
                "subcategory": "FEQ-TEST",
                "question": "Who founded the fabricated Asterion Institute?",
                "correct_answer": "[DOES_NOT_EXIST]",
            }
            checkpoint = Checkpoint(Path(tmp) / "empty-confidence.jsonl")
            with patch(
                "ehq.evaluation.runner.client_for_model",
                return_value=EmptyConfidenceClient(),
            ):
                result = run_model(
                    [item], model, config, checkpoint=checkpoint, offline=True
                )
            self.assertEqual(result["n_retryable_records"], 0)
            self.assertEqual(result["n_terminal_missing_confidence"], 1)
            row = list(checkpoint.records())[0]
            self.assertTrue(row["valid_for_ehq12"])
            self.assertFalse(row["valid_for_ehq3"])
            self.assertTrue(row["confidence_terminal"])
            self.assertEqual(row["confidence_parse_reason"], "response_format")

    def test_model_conditional_known_items_are_not_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_experiment_config(ROOT / "config" / "experiment.json")
            config = replace(config, max_workers=1)
            model = load_models(config.models_path)[0]
            items = [
                {
                    "question_id": "FEQ-UNKNOWN",
                    "category": "FEQ",
                    "subcategory": "FEQ-TEST",
                    "question": "Who founded the fabricated Asterion Institute?",
                    "correct_answer": "[DOES_NOT_EXIST]",
                    "expected_knowability": 0,
                },
                {
                    "question_id": "PCQ-KNOWN",
                    "category": "PCQ",
                    "subcategory": "PCQ-TEST",
                    "question": "Which country is the capital city Madrid in?",
                    "correct_answer": "Spain",
                    "model_knowability": {model.name: 1},
                },
            ]
            checkpoint = Checkpoint(Path(tmp) / "k.jsonl")
            result = run_model(
                items, model, config, checkpoint=checkpoint, offline=True
            )
            self.assertEqual(result["scores"]["n_input"], 1)
            self.assertEqual(result["n_excluded_k1_before_request"], 1)
            self.assertEqual(knowability_for_model(items[1], model), 1)
