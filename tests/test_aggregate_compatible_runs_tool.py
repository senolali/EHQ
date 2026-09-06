import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from ehq.artifacts import verify_run_artifacts, write_artifact_catalog, write_json, write_jsonl
from ehq.constants import FRAMEWORK_VERSION, PROTOCOL_VERSION
from ehq.hashing import sha256_json, sha256_tree


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aggregate_compatible_runs", ROOT / "tools" / "aggregate_compatible_runs.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _record(model, question_id, label, confidence):
    return {
        "question_id": question_id,
        "model": model,
        "category": "FEQ",
        "subcategory": "FEQ-ORG",
        "k": 0,
        "started_at_utc": "2026-08-02T00:00:00+00:00",
        "completed_at_utc": "2026-08-02T00:00:01+00:00",
        "answer_response": {"ok": True, "text": "Unknown"},
        "confidence_response": {"ok": True, "text": str(confidence)},
        "parsed_confidence": confidence / 100,
        "confidence_parse_strategy": "exact_integer",
        "confidence_parse_reason": None,
        "confidence_terminal": False,
        "classification": {
            "label": label,
            "is_correct": label == "CONFIDENT_CORRECT",
            "abstention_detected": label == "ABSTAIN",
            "hedge_detected": label == "HEDGE",
            "substantive_answer_detected": False,
            "reasons": [],
            "automated_label": label,
            "adjudicated_label": None,
        },
        "valid_for_ehq12": True,
        "valid_for_ehq3": True,
        "exclusion_reason": None,
    }


class AggregateCompatibleRunsToolTests(unittest.TestCase):
    def _source(self, root, name, model_name, generation):
        path = root / name
        path.mkdir()
        snapshot = {
            "seed": 42,
            "weights": {"ehq1": 0.3, "ehq2": 0.45, "ehq3": 0.25},
            "confidence": {"n_bins": 10},
        }
        selection = {
            "n_selected": 1,
            "question_ids": ["FEQ-ORG-001"],
            "question_ids_sha256": sha256_json(["FEQ-ORG-001"]),
        }
        model = {
            "name": model_name,
            "provider": "asu",
            "provider_model": model_name.lower(),
            "pair": "test-pair",
            "generation": generation,
        }
        manifest = {
            "framework_version": FRAMEWORK_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "dataset": {"sha256": "dataset-hash"},
            "config": {"sha256": "config-hash", "snapshot": snapshot},
            "models": [model],
            "run": {
                "framework_source_sha256": sha256_tree(ROOT / "src" / "ehq"),
                "selection": selection,
                "answer_prompt_version": "answer-v1",
                "confidence_prompt_version": "confidence-v2",
                "dataset_policy": "NON_PUBLISHABLE_CANDIDATE",
                "publication_blockers": ["test_only"],
                "provider_endpoints": {model_name: "https://example.test/query"},
            },
        }
        write_json(path / "manifest.json", manifest)
        # A substantive record keeps EHQ3 (and therefore the composite) defined
        # under the confidence_substantive_only_v1 protocol, so the aggregate
        # still exercises the RQ2 pairing path.
        write_jsonl(
            path / "records.jsonl",
            [_record(model_name, "FEQ-ORG-001", "CONFIDENT_CORRECT", 80)],
        )
        write_artifact_catalog(path)
        return path

    def test_aggregate_verified_compatible_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = self._source(root, "old", "Old", "old")
            new = self._source(root, "new", "New", "new")
            output = root / "aggregate"
            result = MODULE.aggregate_runs([old, new], output)
            self.assertTrue(result["artifacts_verified"])
            self.assertEqual(result["n_models"], 2)
            self.assertEqual(result["n_records"], 2)
            self.assertEqual(result["provider_calls_made"], 0)
            self.assertTrue(verify_run_artifacts(output)["valid"])
            analysis = json.loads((output / "analysis.json").read_text())
            summary = analysis["rq2_generational_pairs"]["summary"]
            self.assertEqual(summary["inferential_status"], "insufficient_pairs")

    def test_rejects_incompatible_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = self._source(root, "old", "Old", "old")
            new = self._source(root, "new", "New", "new")
            manifest_path = new / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["run"]["selection"]["question_ids_sha256"] = "different"
            write_json(manifest_path, manifest)
            write_artifact_catalog(new)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                MODULE.aggregate_runs([old, new], root / "aggregate")
