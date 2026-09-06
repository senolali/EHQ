import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ehq.analysis import build_analysis_report
from ehq.artifacts import verify_run_artifacts, write_artifact_catalog, write_json
from ehq.constants import EHQ3_PROTOCOL, FRAMEWORK_VERSION, RESPONSE_LABELS
from ehq.evaluation.scoring import compute_ehq_scores, compute_grouped_scores
from ehq.reporting import write_standard_reports
from ehq.types import ModelSpec


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "recompute_ehq3_protocol", ROOT / "tools" / "recompute_ehq3_protocol.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

SNAPSHOT = {
    "seed": 42,
    "weights": {"ehq1": 0.30, "ehq2": 0.45, "ehq3": 0.25},
    "confidence": {"n_bins": 10},
}
WEIGHTS = {"ehq1": 0.30, "ehq2": 0.45, "ehq3": 0.25}


def _record(model, question_id, label, confidence):
    return {
        "question_id": question_id,
        "model": model,
        "category": "FEQ",
        "subcategory": "FEQ-ORG",
        "k": 0,
        "answer_response": {"ok": True, "text": "answer"},
        "confidence_response": {"ok": True, "text": str(int(confidence * 100))},
        "parsed_confidence": confidence,
        "confidence_terminal": False,
        "classification": {
            "label": label,
            "is_correct": label == "CONFIDENT_CORRECT",
            "abstention_detected": label == "ABSTAIN",
            "hedge_detected": label == "HEDGE",
            "substantive_answer_detected": label.startswith("CONFIDENT"),
            "reasons": [],
            "automated_label": label,
            "adjudicated_label": None,
        },
        "valid_for_ehq12": True,
        "valid_for_ehq3": True,
        "exclusion_reason": None,
    }


def _model_records(model):
    return [
        _record(model, "FEQ-ORG-001", "ABSTAIN", 0.05),
        _record(model, "FEQ-ORG-002", "HEDGE", 0.20),
        _record(model, "FEQ-ORG-003", "CONFIDENT_CORRECT", 0.90),
        _record(model, "FEQ-ORG-004", "CONFIDENT_WRONG", 0.80),
    ]


class RecomputeEhq3ProtocolToolTests(unittest.TestCase):
    def _legacy_run(self, run_dir: Path, model_names):
        """Write a completed run scored under the superseded EHQ3 definition."""

        models = [
            ModelSpec(
                name=name,
                provider="asu",
                provider_model=name.lower(),
                pair="test-pair",
                generation=generation,
            )
            for name, generation in zip(model_names, ("old", "new"))
        ]
        results = {}
        # The superseded definition calibrated over every valid EHQ3 record,
        # including ABSTAIN and HEDGE, and did not stamp the protocol name.
        def _legacy_shape(scores):
            scores.pop("ehq3_protocol", None)
            scores.pop("n_ehq3_calibration", None)
            return scores

        with patch(
            "ehq.evaluation.scoring.EHQ3_SUBSTANTIVE_LABELS", RESPONSE_LABELS
        ):
            for model in models:
                rows = _model_records(model.name)
                results[model.name] = {
                    "model": model.name,
                    "provider_endpoint": "https://example.test/query",
                    "completed_at_utc": "2026-08-03T00:00:00+00:00",
                    "scores": _legacy_shape(
                        compute_ehq_scores(rows, weights=WEIGHTS, n_bins=10)
                    ),
                    "category_scores": {
                        key: _legacy_shape(value)
                        for key, value in compute_grouped_scores(
                            rows, group_field="category", weights=WEIGHTS, n_bins=10
                        ).items()
                    },
                    "subcategory_scores": {
                        key: _legacy_shape(value)
                        for key, value in compute_grouped_scores(
                            rows,
                            group_field="subcategory",
                            weights=WEIGHTS,
                            n_bins=10,
                        ).items()
                    },
                    "n_retryable_records": 0,
                    "n_terminal_missing_confidence": 0,
                    "n_excluded_k1_before_request": 0,
                    "records": rows,
                }

        manifest = {
            "experiment_name": "EHQ-test-experiment",
            "framework_version": "0.2.12",
            "protocol_version": "1.1.4",
            "config": {"path": "config/experiment.json", "snapshot": SNAPSHOT},
            "dataset": {"sha256": "dataset-hash"},
            "models": [
                {
                    "name": model.name,
                    "provider": model.provider,
                    "provider_model": model.provider_model,
                    "pair": model.pair,
                    "generation": model.generation,
                }
                for model in models
            ],
            "run": {
                "run_id": "20260803T000000Z",
                "mode": "real",
                "fingerprint": "parent-fingerprint",
                "dataset_policy": "NON_PUBLISHABLE_CANDIDATE",
            },
        }
        run_dir.mkdir(parents=True)
        write_json(run_dir / "manifest.json", manifest)
        for name, result in results.items():
            write_json(run_dir / "models" / f"{name}.json", result)
        write_standard_reports(run_dir, results)
        write_json(
            run_dir / "analysis.json",
            build_analysis_report(results, models, seed=42),
        )
        write_artifact_catalog(run_dir)
        return results

    def test_regenerates_official_outputs_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "real_20260803T000000Z"
            legacy = self._legacy_run(run_dir, ["Old", "New"])
            legacy_scores = {
                name: dict(result["scores"]) for name, result in legacy.items()
            }
            records_before = (run_dir / "records.jsonl").read_bytes()
            entries_before = {path.name for path in root.iterdir()}

            outcome = MODULE.recompute(run_dir, excel=False, figures=False)

            # No sibling output directory is created.
            self.assertEqual({path.name for path in root.iterdir()}, entries_before)
            self.assertEqual(outcome["run_dir"], str(run_dir.resolve()))
            self.assertEqual(outcome["api_calls"], 0)
            self.assertEqual(outcome["cache_hits"], 0)
            self.assertEqual(outcome["models_processed"], 2)
            self.assertFalse(outcome["ehq1_changed"])
            self.assertFalse(outcome["ehq2_changed"])
            self.assertTrue(outcome["ehq3_changed"])
            self.assertTrue(outcome["artifacts_verified"])
            self.assertTrue(verify_run_artifacts(run_dir)["valid"])

            # Raw evaluation records are inputs, not outputs.
            self.assertEqual((run_dir / "records.jsonl").read_bytes(), records_before)

            for name, previous in legacy_scores.items():
                current = json.loads(
                    (run_dir / "models" / f"{name}.json").read_text(encoding="utf-8")
                )["scores"]
                self.assertEqual(current["EHQ1"], previous["EHQ1"])
                self.assertEqual(current["EHQ2"], previous["EHQ2"])
                self.assertNotEqual(current["EHQ3"], previous["EHQ3"])
                self.assertAlmostEqual(current["EHQ3"], 0.55)
                self.assertEqual(current["n_ehq3_calibration"], 2)
                self.assertEqual(current["ehq3_protocol"], EHQ3_PROTOCOL)

            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["ehq3_protocol"], EHQ3_PROTOCOL)

            analysis = json.loads(
                (run_dir / "analysis.json").read_text(encoding="utf-8")
            )
            self.assertEqual(analysis["ehq3_protocol"], EHQ3_PROTOCOL)

            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["ehq3_protocol"], EHQ3_PROTOCOL)
            self.assertEqual(manifest["framework_version"], FRAMEWORK_VERSION)
            # The experiment identity is preserved: the model responses behind
            # this run did not change.
            self.assertEqual(manifest["experiment_name"], "EHQ-test-experiment")
            self.assertEqual(manifest["run"]["run_id"], "20260803T000000Z")
            self.assertEqual(manifest["run"]["fingerprint"], "parent-fingerprint")
            recomputation = manifest["run"]["ehq3_protocol_recomputation"]
            self.assertEqual(recomputation["provider_calls_made"], 0)
            self.assertEqual(
                recomputation["provider_run_framework_version"], "0.2.12"
            )
            self.assertFalse(recomputation["records_rewritten"])

            # Regeneration is idempotent and never loses the provider-run
            # provenance, so it is safe to re-run over official outputs.
            second = MODULE.recompute(run_dir, excel=False, figures=False)
            self.assertFalse(second["ehq3_changed"])
            self.assertTrue(second["artifacts_verified"])
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["run"]["ehq3_protocol_recomputation"][
                    "provider_run_framework_version"
                ],
                "0.2.12",
            )
            self.assertEqual((run_dir / "records.jsonl").read_bytes(), records_before)

    def test_record_order_follows_the_original_file_not_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "real_20260803T000000Z"
            self._legacy_run(run_dir, ["Old", "New"])
            records_before = (run_dir / "records.jsonl").read_bytes()

            # A manifest whose model order differs from the written record
            # order must not silently reshuffle records.jsonl.
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["models"].reverse()
            write_json(manifest_path, manifest)
            write_artifact_catalog(run_dir)

            outcome = MODULE.recompute(run_dir, excel=False, figures=False)
            self.assertTrue(outcome["ehq3_changed"])
            self.assertEqual((run_dir / "records.jsonl").read_bytes(), records_before)

    def test_refuses_a_run_whose_artifacts_do_not_match_its_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "real_20260803T000000Z"
            self._legacy_run(run_dir, ["Old", "New"])
            (run_dir / "summary.csv").write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                MODULE.recompute(run_dir, excel=False, figures=False)


if __name__ == "__main__":
    unittest.main()


class CapabilityScorePreservationTests(RecomputeEhq3ProtocolToolTests):
    """RQ1 must survive an EHQ3-only regeneration."""

    def _run_with_capability(self, root, rows="Old,0.4\nNew,0.9\n"):
        from ehq.artifacts import write_artifact_catalog, write_json
        from ehq.hashing import sha256_file

        run = root / "real_capability"
        self._legacy_run(run, ["Old", "New"])
        scores = root / "capability_scores.csv"
        scores.write_text(f"model,capability_score\n{rows}", encoding="utf-8")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        manifest["run"]["capability_scores_path"] = str(scores)
        manifest["run"]["capability_scores_sha256"] = sha256_file(scores)
        write_json(run / "manifest.json", manifest)
        write_artifact_catalog(run)
        return run, scores

    def test_capability_scores_are_reloaded_not_dropped(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            run, _ = self._run_with_capability(Path(tmp))
            MODULE.recompute(run, excel=False, figures=False)
            analysis = json.loads(
                (run / "analysis.json").read_text(encoding="utf-8")
            )
            # Present and evaluated rather than reported as never supplied.
            self.assertNotEqual(
                analysis["rq1_capability_relationship"]["status"],
                "capability_scores_not_provided",
            )
            manifest = json.loads(
                (run / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                manifest["run"]["ehq3_protocol_recomputation"][
                    "capability_scores_reloaded"
                ]
            )

    def test_edited_capability_file_is_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            run, scores = self._run_with_capability(Path(tmp))
            scores.write_text(
                "model,capability_score\nOld,0.9\nNew,0.4\n", encoding="utf-8"
            )
            # Silently rescoring RQ1 against different numbers would be worse
            # than refusing to regenerate at all.
            with self.assertRaises(SystemExit):
                MODULE.recompute(run, excel=False, figures=False)

    def test_missing_capability_file_is_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            run, scores = self._run_with_capability(Path(tmp))
            scores.unlink()
            with self.assertRaises(SystemExit):
                MODULE.recompute(run, excel=False, figures=False)
